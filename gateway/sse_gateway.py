#!/usr/bin/env python3
import os
import codecs
import hashlib
import json
import re
import shlex
import shutil
import struct
import sys
import time
import threading
import urllib.request
import urllib.error
import uuid
from collections import OrderedDict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

UPSTREAM_BASE = os.environ.get("LLAMA_BACKEND_URL", os.environ.get("UPSTREAM_BASE", "http://127.0.0.1:8789"))
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
_DEFAULT_CORS_HOST = "127.0.0.1" if GATEWAY_HOST == "0.0.0.0" else GATEWAY_HOST
GATEWAY_CORS_ORIGIN = os.environ.get("GATEWAY_CORS_ORIGIN", f"http://{_DEFAULT_CORS_HOST}:{os.environ.get('GATEWAY_PORT', '8787')}")
ALLOWED_POST_PATHS = frozenset({
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/chat/completions/input_tokens",
    "/v1/completions",
    "/completions",
    "/v1/embeddings",
    "/embeddings",
    "/v1/complete",
    "/complete",
    "/tokenize",
})
ALLOWED_GET_PATHS = frozenset({
    "/health",
    "/v1/models",
    "/props",
})
_ROOT_ROUTE_ALIASES = {
    "/chat/completions": "/v1/chat/completions",
    "/chat/completions/input_tokens": "/v1/chat/completions/input_tokens",
    "/models": "/v1/models",
    "/messages": "/v1/messages",
    "/messages/count_tokens": "/v1/messages/count_tokens",
    "/v1/health": "/health",
}
_DOT_SEGMENT_RE = re.compile(r"(^|/)\.{1,2}(/|$)")
try:
    with open(os.path.abspath(__file__), "rb") as _gwf:
        GATEWAY_BUILD_HASH = hashlib.sha256(_gwf.read()).hexdigest()[:16]
except OSError:
    GATEWAY_BUILD_HASH = "unknown"


def _normalize_gateway_path(raw_path: str):
    """Canonicalize OpenAI/Anthropic routes; return None on dot-segments, backslashes, or percent-encoded paths."""
    path_only, sep, query = (raw_path or "/").partition("?")
    if not path_only.startswith("/") or "%" in path_only or "\\" in path_only:
        return None
    norm = re.sub(r"/{2,}", "/", path_only)
    if _DOT_SEGMENT_RE.search(norm):
        return None
    while norm == "/v1/v1" or norm.startswith("/v1/v1/"):
        norm = norm[3:]
    if len(norm) > 1:
        norm = norm.rstrip("/")
    norm = _ROOT_ROUTE_ALIASES.get(norm, norm)
    return f"{norm}{sep}{query}" if sep else norm
_GATEWAY_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_GATEWAY_DIR)
_KV_CACHE_DIR = os.path.join(_REPO_ROOT, "kv_cache")
if _KV_CACHE_DIR not in sys.path:
    sys.path.insert(0, _KV_CACHE_DIR)


def _resolve_workspace_dir() -> str:
    env_ws = os.environ.get("WORKSPACE_DIR", "").strip()
    if env_ws:
        return env_ws
    marker = os.path.join(_REPO_ROOT, ".modchip_workspace")
    if os.path.isfile(marker):
        try:
            with open(marker, "r", encoding="utf-8") as mf:
                line = mf.readline().strip()
                if line:
                    return line
        except OSError:
            pass
    return os.path.join(_REPO_ROOT, "workspace")


_WORKSPACE_DIR = _resolve_workspace_dir()
_EUID = getattr(os, "geteuid", getattr(os, "getuid", lambda: 0))()
_HARP_SHM_PATH = f"/dev/shm/modchip_harp_cache_{_EUID}.bin"

_active_model_cache = {"name": "", "ts": 0.0}
_active_model_lock = threading.Lock()


def _detect_active_model(target_upstream: str = UPSTREAM_BASE, fallback: str = "modchip-local") -> str:
    """Dynamically detects the active served model alias (env -> /v1/models -> /props -> default_model)."""
    env_name = os.environ.get("MODCHIP_MODEL_NAME", "").strip()
    if env_name:
        return env_name
    now = time.monotonic()
    with _active_model_lock:
        if _active_model_cache["name"] and (now - _active_model_cache["ts"]) < 5.0:
            return _active_model_cache["name"]
    detected = ""
    try:
        mreq = urllib.request.Request(f"{target_upstream}/v1/models", method="GET")
        with urllib.request.urlopen(mreq, timeout=0.35) as mresp:
            mdata = json.loads(mresp.read().decode("utf-8", errors="replace"))
            mlist = mdata.get("data") or mdata.get("models") or []
            if mlist and isinstance(mlist[0], dict):
                detected = os.path.basename(str(mlist[0].get("id") or mlist[0].get("name") or "")).strip()
    except Exception:
        pass
    if not detected:
        try:
            preq = urllib.request.Request(f"{target_upstream}/props", method="GET")
            with urllib.request.urlopen(preq, timeout=0.35) as presp:
                pdata = json.loads(presp.read().decode("utf-8", errors="replace"))
                detected = os.path.basename(str(pdata.get("model_path") or "")).strip()
        except Exception:
            pass
    if not detected:
        def_mfile = os.path.join(_resolve_workspace_dir(), "run", "default_model")
        if os.path.isfile(def_mfile):
            try:
                with open(def_mfile, "r", encoding="utf-8") as dmf:
                    detected = os.path.basename(dmf.readline().strip())
            except OSError:
                pass
    if detected:
        with _active_model_lock:
            _active_model_cache["name"] = detected
            _active_model_cache["ts"] = now
        return detected
    return fallback or "modchip-local"


_harp_engine = None
if os.environ.get("HARP_CACHE", "1") != "0" and os.path.isdir("/dev/shm"):
    try:
        from harp_cache import HARPCacheEngine, hash_token_block
        _harp_engine = HARPCacheEngine(shm_path=_HARP_SHM_PATH)
        print(f"[Gateway] RouteCrack-KV: HARP-Cache Shared Memory Engine initialized ({_HARP_SHM_PATH}).", flush=True)
    except Exception as _harp_init_err:
        _harp_engine = None
        print(f"[Gateway] Notice: HARP-Cache not loaded: {_harp_init_err}", flush=True)
else:
    print("[Gateway] Notice: HARP-Cache disabled (no /dev/shm or HARP_CACHE=0).", flush=True)

# Thread-safe session tracking (LRU up to 128 concurrent sessions)
_state_lock = threading.Lock()
_session_modes = OrderedDict()
_default_think_mode = "no"

_slot0_restore_lock = threading.Lock()
_last_endpoint_was_chat = True
SLOT0_SNAPSHOT_NAME = os.environ.get("LLAMA_SLOT_SNAPSHOT_NAME", "claude_40tools_slot0.bin")

def _copy_file_secure_0600(src_path: str, dst_path: str) -> None:
    """Copies src_path -> dst_path safely without following symlinks and enforcing mode 0600 + UID ownership."""
    import stat as _stat
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(dst_path, open_flags, 0o600)
    try:
        st = os.fstat(fd)
        if not _stat.S_ISREG(st.st_mode) or st.st_uid != _EUID:
            raise PermissionError(f"Unsafe SHM snapshot target: {dst_path}")
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as out_f:
            fd = -1
            with open(src_path, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=1 << 20)
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def _ensure_slot0_restored(target_upstream: str) -> None:
    """Auto-restores the per-UID Slot 0 snapshot (22.2K tokens for Nemotron or 2.94K compact tokens for DeepSeek-V4) in ~19-32 ms if Slot 0 is cold or was evicted."""
    ws_dir = _resolve_workspace_dir()
    base_name, base_ext = os.path.splitext(SLOT0_SNAPSHOT_NAME)
    uid_snap_name = f"{base_name}_{_EUID}{base_ext}"
    shm_snap = f"/dev/shm/{uid_snap_name}"
    restore_filename = uid_snap_name
    custom_cache_dir = os.environ.get("MODCHIP_CACHE_DIR", "")
    if not os.path.exists(shm_snap):
        for disk_cand in (
            os.path.join(custom_cache_dir, SLOT0_SNAPSHOT_NAME) if custom_cache_dir else "",
            os.path.join(ws_dir, "cache", SLOT0_SNAPSHOT_NAME),
            os.path.join(ws_dir, "slot_cache", SLOT0_SNAPSHOT_NAME),
        ):
            if disk_cand and os.path.exists(disk_cand):
                try:
                    _copy_file_secure_0600(disk_cand, shm_snap)
                    if os.path.exists(disk_cand + ".ckpt") and not os.path.exists(shm_snap + ".ckpt"):
                        _copy_file_secure_0600(disk_cand + ".ckpt", shm_snap + ".ckpt")
                    print(f"[Gateway] [Slot0-Guard] Copied {disk_cand} -> {shm_snap} (mode 0600)", flush=True)
                except Exception:
                    pass
                break

    if not os.path.exists(shm_snap) and custom_cache_dir and os.path.exists(os.path.join(custom_cache_dir, SLOT0_SNAPSHOT_NAME)):
        shm_snap = os.path.join(custom_cache_dir, SLOT0_SNAPSHOT_NAME)
        restore_filename = SLOT0_SNAPSHOT_NAME

    if not os.path.exists(shm_snap):
        return

    min_tok_threshold = 2500 if "dsv4" in SLOT0_SNAPSHOT_NAME.lower() else 21500
    with _slot0_restore_lock:
        try:
            req = urllib.request.Request(f"{target_upstream}/slots")
            with urllib.request.urlopen(req, timeout=0.25) as resp:
                slots = json.loads(resp.read().decode("utf-8", errors="replace"))
            if not isinstance(slots, list) or not slots:
                return
            s0 = slots[0]
            if s0.get("is_processing", False):
                return
            n_tokens = int(s0.get("n_prompt_tokens") or s0.get("n_past") or 0)
            if n_tokens >= min_tok_threshold:
                return
            t0 = time.perf_counter()
            restore_req = urllib.request.Request(
                f"{target_upstream}/slots/0?action=restore",
                data=json.dumps({"filename": restore_filename}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(restore_req, timeout=10.0) as r_resp:
                res = json.loads(r_resp.read().decode("utf-8", errors="replace"))
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(
                f"[Gateway] [Slot0-Guard] ⚡ Auto-restored {res.get('n_restored', min_tok_threshold)} tokens "
                f"from {shm_snap} into Slot 0 in {dt_ms:.1f} ms (was {n_tokens} tok)!",
                flush=True,
            )
        except Exception as e:
            print(f"[Gateway] [Slot0-Guard] Notice: {e}", flush=True)


def get_session_mode(session_id: str, init_mode: str = "") -> str:
    with _state_lock:
        if session_id not in _session_modes:
            mode = init_mode if init_mode in ("no", "med", "max") else _default_think_mode
            _session_modes[session_id] = mode
            if len(_session_modes) > 128:
                _session_modes.popitem(last=False)
        else:
            _session_modes.move_to_end(session_id)
        return _session_modes[session_id]

def set_session_mode(session_id: str, new_mode: str) -> None:
    global _default_think_mode
    with _state_lock:
        _session_modes[session_id] = new_mode
        _session_modes.move_to_end(session_id)
        if session_id in ("default", "all"):
            _default_think_mode = new_mode

_MAX_XML_PARAM_JSON_BYTES = 65536
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "").strip()
GATEWAY_STRIP_REMINDERS = os.environ.get("GATEWAY_STRIP_REMINDERS", "1") != "0"

def _coerce_xml_param(raw, prop):
    """Convert an XML <parameter> value to the JSON type that the tool schema declares."""
    ptype = prop.get("type") if isinstance(prop, dict) else None
    if ptype == "string":
        return raw
    s = raw.strip()
    if len(s) > _MAX_XML_PARAM_JSON_BYTES:
        return raw if ptype is not None else s
    try:
        if ptype == "integer":
            return int(s)
        if ptype == "number":
            return float(s) if any(c in s for c in ".eE") else int(s)
        if ptype == "boolean":
            return s.lower() == "true"
        if ptype in ("array", "object") or (ptype is None and s[:1] in "[{"):
            return json.loads(s)
        if ptype is None and (s.lower() in ("true", "false") or re.match(r"^-?\d+(\.\d+)?$", s)):
            return json.loads(s.lower())
    except Exception:
        pass
    return raw if ptype is not None else s

def _write_as_bash(name, args, schemas=None):
    """Optionally rewrite Write calls into base64 Bash commands ONLY when explicitly enabled."""
    if os.environ.get("MODCHIP_EXPERIMENTAL_BASH_WRITE") != "1":
        return name, args
    if schemas is not None and "Bash" not in schemas:
        return name, args
    if name == "Write" and isinstance(args, dict) and args.get("file_path"):
        import base64
        b64 = base64.b64encode(str(args.get("content", "")).encode("utf-8")).decode("ascii")
        return "Bash", {"command": f"echo '{b64}' | base64 -d > {shlex.quote(str(args['file_path']))}"}
    return name, args

# DeepSeek DSML tool calls (advertised by the compact-tools prompt on dsv4 / MODCHIP_COMPACT_TOOLS).
# Official DeepSeek V3.2/V4 templates and sse_gateway.py emit the U+FF5C marker in BOTH opening and closing tags
# (`</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>`), while stripped variants emit plain `</...>`.
# Both closing conventions are accepted in linear O(N) time.
DSML_MARKER = "｜DSML｜"
DSML_TOOL_OPEN = f"<{DSML_MARKER}tool_calls>"
DSML_TOOL_CLOSE_FULL = f"</{DSML_MARKER}tool_calls>"
DSML_TOOL_CLOSE_PLAIN = "</tool_calls>"
DSML_TOOL_CLOSE = DSML_TOOL_CLOSE_FULL
DSML_INVOKE_OPEN_PREFIX = f"<{DSML_MARKER}invoke"
DSML_INVOKE_CLOSE_FULL = f"</{DSML_MARKER}invoke>"
DSML_INVOKE_CLOSE_PLAIN = "</invoke>"
DSML_PARAM_OPEN_PREFIX = f"<{DSML_MARKER}parameter"
DSML_PARAM_CLOSE_FULL = f"</{DSML_MARKER}parameter>"
DSML_PARAM_CLOSE_PLAIN = "</parameter>"
_DSML_ATTR_RE = re.compile(r'([A-Za-z_][\w-]*)\s*=\s*"([^"]*)"')
# Longest marker minus one (19 - 1 = 18): tail kept in buffers so tags split across SSE deltas are never missed.
_TAG_HOLDBACK = max(len(DSML_TOOL_OPEN), len(DSML_TOOL_CLOSE_FULL)) - 1
_MAX_TOOL_BUF_BYTES = 262144
_MAX_RAW_SSE_BUF_BYTES = 4194304


def _iter_dsml_elements(text, open_prefix, close_full, close_plain):
    """Linear-time O(N) cursor scanner for `<｜DSML｜tag attrs>body</｜DSML｜tag>` (or plain `</tag>`)."""
    n = len(text)
    pos = 0
    plen = len(open_prefix)
    while pos < n:
        start = text.find(open_prefix, pos)
        if start == -1:
            break
        after_prefix = start + plen
        if after_prefix < n and text[after_prefix] not in " \t\r\n>":
            pos = after_prefix
            continue
        gt = text.find(">", after_prefix)
        if gt == -1:
            break
        attr_str = text[after_prefix:gt]
        body_start = gt + 1
        end_full = text.find(close_full, body_start)
        end_plain = text.find(close_plain, body_start)
        if end_full != -1 and (end_plain == -1 or end_full <= end_plain):
            body_end = end_full
            next_pos = end_full + len(close_full)
        elif end_plain != -1:
            body_end = end_plain
            next_pos = end_plain + len(close_plain)
        else:
            break
        yield attr_str, text[body_start:body_end]
        pos = next_pos


def parse_dsml_tool_calls(text, schemas=None):
    """Parse all DeepSeek DSML `<｜DSML｜invoke ...>` tool calls in `text` in linear O(N) time."""
    if DSML_TOOL_OPEN in text:
        body = text.split(DSML_TOOL_OPEN, 1)[1]
        close_full = body.find(DSML_TOOL_CLOSE_FULL)
        close_plain = body.find(DSML_TOOL_CLOSE_PLAIN)
        if close_full != -1 and (close_plain == -1 or close_full <= close_plain):
            body = body[:close_full]
        elif close_plain != -1:
            body = body[:close_plain]
    else:
        body = text
    calls = []
    for inv_attrs_raw, inv_body in _iter_dsml_elements(
        body, DSML_INVOKE_OPEN_PREFIX, DSML_INVOKE_CLOSE_FULL, DSML_INVOKE_CLOSE_PLAIN
    ):
        attrs = dict(_DSML_ATTR_RE.findall(inv_attrs_raw))
        name = attrs.get("name", "").strip()
        # Enforce tool whitelist strictly (including empty schemas == {} on tool-less requests).
        if not name or (schemas is not None and name not in schemas):
            continue
        props = (schemas or {}).get(name, {})
        args = {}
        for param_attrs_raw, raw in _iter_dsml_elements(
            inv_body, DSML_PARAM_OPEN_PREFIX, DSML_PARAM_CLOSE_FULL, DSML_PARAM_CLOSE_PLAIN
        ):
            pattrs = dict(_DSML_ATTR_RE.findall(param_attrs_raw))
            key = pattrs.get("name", "").strip()
            if not key:
                continue
            prop = props.get(key, {})
            ptype = prop.get("type") if isinstance(prop, dict) else None
            if raw.startswith("\n"):
                raw = raw[1:]
            if raw.endswith("\n"):
                raw = raw[:-1]
            if pattrs.get("string") == "true" and ptype in (None, "string"):
                args[key] = raw
            elif pattrs.get("string") == "false" and not ptype:
                args[key] = _coerce_xml_param(raw, {})
            else:
                args[key] = _coerce_xml_param(raw, prop)
        name, args = _write_as_bash(name, args, schemas)
        calls.append({"name": name, "input": args})
    return calls


def parse_tool_calls(text, schemas=None):
    """Every tool call in `text`: each DSML invoke, or the single call of the older formats."""
    calls = parse_dsml_tool_calls(text, schemas)
    if calls:
        return calls
    parsed = try_parse_tool_call(text, schemas)
    return [parsed] if parsed else []


def _tool_use_block(parsed):
    return {
        "type": "tool_use",
        "id": "call_" + uuid.uuid4().hex[:12],
        "name": parsed["name"],
        "input": parsed["input"],
    }


def try_parse_tool_call(text, schemas=None):
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t).strip()

    if DSML_TOOL_OPEN in t or DSML_INVOKE_OPEN_PREFIX in t:
        calls = parse_dsml_tool_calls(t, schemas)
        return calls[0] if calls else None

    m = re.search(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", t)
    if m:
        t = m.group(1).strip()

    fn_m = re.search(r"<function=\s*([^>\n\s]+)\s*>([\s\S]*?)(?:</function>|$)", t)
    if fn_m:
        name = fn_m.group(1).strip()
        # Security hardening: reject tool names not in allowed schemas (including empty schemas == {})
        if schemas is not None and name not in schemas:
            return None
        body = fn_m.group(2)
        props = (schemas or {}).get(name, {})
        args = {}
        for pm in re.finditer(r"<parameter=\s*([^>\n\s]+)\s*>\n?([\s\S]*?)\n?</parameter>", body):
            k = pm.group(1).strip()
            args[k] = _coerce_xml_param(pm.group(2), props.get(k, {}))
        name, args = _write_as_bash(name, args, schemas)
        return {"name": name, "input": args}

    if t.startswith("{") and t.endswith("}"):
        try:
            data = json.loads(t)
            if isinstance(data, dict):
                if "name" in data and ("arguments" in data or "parameters" in data or "input" in data):
                    name = data["name"]
                    if schemas is not None and name not in schemas:
                        return None
                    args = data.get("arguments", data.get("parameters", data.get("input", {})))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    name, args = _write_as_bash(name, args, schemas)
                    return {"name": name, "input": args if isinstance(args, dict) else {}}
                if "function" in data and isinstance(data["function"], dict):
                    fn = data["function"]
                    name = fn.get("name")
                    if schemas is not None and name not in schemas:
                        return None
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    name, args = _write_as_bash(name, args, schemas)
                    return {"name": name, "input": args if isinstance(args, dict) else {}}
        except Exception:
            pass
    return None

class ThinkBlockFilter:
    """
    Robust SSE stream normalizer for Claude Code (V3):
    - Uses --skip-chat-parsing from llama-server to completely prevent C++ PEG grammar aborts.
    - Intercepts <think>...</think> and converts into native Anthropic thinking/thinking_delta blocks.
    - Intercepts <tool_call>...</tool_call> and DeepSeek DSML tool-call blocks and converts them into native
      Anthropic tool_use blocks.
    - Resilient to tag ordering errors (tool_call inside think, missing </think>, stray tags).
    - Deduplicates identical tool calls.
    - Enforces hard cap of max_tool_calls (default 2) per turn.
    - Ensures valid, sequential Anthropic block lifecycles (start -> deltas -> stop).
    """
    def __init__(self, mode="no", max_tool_calls=2, schemas=None):
        self.mode = mode
        self.max_tool_calls = max_tool_calls
        self.schemas = schemas
        self.next_idx = 0
        self.state = "INIT"  # INIT, THINKING, TEXT, TOOL
        self.buf = ""
        self.tool_buf = ""
        self.seen_tools = set()
        self.tool_count = 0
        self.tool_is_dsml = False
        self.has_content_block = False
        self.has_tool_call = False
        self.raw_sse_buf = ""
        self.thinking_buf = ""
        self.text_buf = ""
        self.native_text_open = False

    def feed(self, chunk: str) -> str:
        self.raw_sse_buf += chunk.replace("\r\n", "\n")
        blocks = self.raw_sse_buf.split("\n\n")
        self.raw_sse_buf = blocks[-1]
        if len(self.raw_sse_buf) > _MAX_RAW_SSE_BUF_BYTES:
            self.raw_sse_buf = self.raw_sse_buf[-_MAX_RAW_SSE_BUF_BYTES:]
        out = []
        for b in blocks[:-1]:
            if not b.strip():
                continue
            for r in self._process_sse_block(b.strip()):
                out.append(r + "\n\n")
        return "".join(out)

    def flush(self) -> str:
        out = []
        if self.raw_sse_buf.strip():
            for r in self._process_sse_block(self.raw_sse_buf.strip()):
                out.append(r + "\n\n")
            self.raw_sse_buf = ""
        for r in self._flush_normalizer():
            out.append(r + "\n\n")
        return "".join(out)

    def _process_sse_block(self, block: str):
        lines = block.split("\n")
        data_lines = [line[6:].strip() for line in lines if line.startswith("data: ")]
        data_str = "\n".join(data_lines).strip()
        if not data_str or data_str == "[DONE]":
            return [block]
        try:
            ev = json.loads(data_str)
        except Exception:
            return [block]

        t = ev.get("type")
        if t == "message_start":
            return [block]

        if t == "content_block_start":
            cb = ev.get("content_block", {})
            cb_type = cb.get("type", "")
            self.upstream_cb_type = cb_type
            if cb_type == "tool_use":
                out = []
                if self.state == "TEXT":
                    if self.buf:
                        clean = self.buf.replace("</think>", "").replace("<think>", "")
                        if clean:
                            self.text_buf += clean
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "text_delta", "text": clean}
                            }))
                        self.buf = ""
                    out.append("event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": self.next_idx
                    }))
                    self.next_idx += 1
                    self.state = "INIT"
                    self.native_text_open = False
                elif self.state == "THINKING":
                    if self.mode != "no":
                        out.append("event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop",
                            "index": self.next_idx
                        }))
                        self.next_idx += 1
                    self.state = "INIT"
                self.has_content_block = True
                self.has_tool_call = True
                self.upstream_tool_idx = self.next_idx
                self.next_idx += 1
                ev["index"] = self.upstream_tool_idx
                out.append("event: content_block_start\ndata: " + json.dumps(ev))
                return out
            elif cb_type == "thinking":
                # A native thinking block may follow a text block that _feed_text
                # opened lazily. Close it first so the two blocks never share an index.
                out = []
                if self.state == "TEXT" and self.native_text_open:
                    if self.buf:
                        clean = self.buf.replace("</think>", "").replace("<think>", "")
                        if clean:
                            self.text_buf += clean
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "text_delta", "text": clean}
                            }))
                        self.buf = ""
                    # The text block was already emitted (even in mode "no"), so it
                    # must always be closed and its index retired regardless of mode.
                    out.append("event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": self.next_idx
                    }))
                    self.next_idx += 1
                    self.native_text_open = False
                    self.state = "INIT"
                self.upstream_thinking_open = True
                if self.mode != "no":
                    self.upstream_thinking_idx = self.next_idx
                    ev["index"] = self.upstream_thinking_idx
                    out.append("event: content_block_start\ndata: " + json.dumps(ev))
                return out
            return []

        if t == "content_block_delta":
            delta = ev.get("delta", {})
            d_type = delta.get("type", "")
            if d_type == "input_json_delta" and getattr(self, "upstream_cb_type", "") == "tool_use":
                ev["index"] = getattr(self, "upstream_tool_idx", 0)
                return ["event: content_block_delta\ndata: " + json.dumps(ev)]
            if d_type == "thinking_delta" or "thinking" in delta:
                th_txt = delta.get("thinking", "")
                if th_txt:
                    self.thinking_buf += th_txt
                    if self.mode != "no":
                        ev["index"] = getattr(self, "upstream_thinking_idx", self.next_idx)
                        return ["event: content_block_delta\ndata: " + json.dumps(ev)]
                return []
            txt = delta.get("text", "")
            if not txt:
                return []
            return self._feed_text(txt)

        if t == "content_block_stop":
            cb_type = getattr(self, "upstream_cb_type", "")
            self.upstream_cb_type = ""
            if cb_type == "tool_use":
                return ["event: content_block_stop\ndata: " + json.dumps({
                    "type": "content_block_stop",
                    "index": getattr(self, "upstream_tool_idx", 0)
                })]
            if cb_type == "thinking" and getattr(self, "upstream_thinking_open", False):
                self.upstream_thinking_open = False
                if self.mode != "no":
                    idx = getattr(self, "upstream_thinking_idx", self.next_idx)
                    self.next_idx = idx + 1
                    return ["event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": idx
                    })]
            return []

        if t in ("message_delta", "message_stop"):
            out = []
            flush_evs = self._flush_normalizer()
            out.extend(flush_evs)
            if t == "message_delta":
                if self.has_tool_call:
                    ev.setdefault("delta", {})["stop_reason"] = "tool_use"
                elif ev.get("delta", {}).get("stop_reason") == "tool_use":
                    ev["delta"]["stop_reason"] = "end_turn"
            out.append("event: " + t + "\ndata: " + json.dumps(ev))
            return out

        return [block]

    def _emit_tool_call(self, out, parsed):
        name = parsed.get("name")
        args = parsed.get("input", {})
        sig = (name, json.dumps(args, sort_keys=True))
        if sig in self.seen_tools or self.tool_count >= self.max_tool_calls:
            return
        self.seen_tools.add(sig)
        self.tool_count += 1
        self.has_content_block = True
        self.has_tool_call = True
        cid = "call_" + uuid.uuid4().hex[:12]
        out.append("event: content_block_start\ndata: " + json.dumps({
            "type": "content_block_start",
            "index": self.next_idx,
            "content_block": {"type": "tool_use", "id": cid, "name": name, "input": {}}
        }))
        out.append("event: content_block_delta\ndata: " + json.dumps({
            "type": "content_block_delta",
            "index": self.next_idx,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(args)}
        }))
        out.append("event: content_block_stop\ndata: " + json.dumps({
            "type": "content_block_stop",
            "index": self.next_idx
        }))
        self.next_idx += 1

    def _feed_text(self, text):
        out = []
        self.buf += text
        tools_allowed = self.schemas is None or bool(self.schemas)

        while self.buf:
            if self.state == "INIT":
                if self.buf:
                    self.buf = self.buf.lstrip(" \t\r\n")
                if not self.buf:
                    break

                if self.buf.startswith("</think>"):
                    self.buf = self.buf[8:]
                    continue
                if self.buf.startswith("</tool_call>"):
                    self.buf = self.buf[12:]
                    continue
                if self.buf.startswith(DSML_TOOL_CLOSE_FULL):
                    self.buf = self.buf[len(DSML_TOOL_CLOSE_FULL):]
                    continue
                if self.buf.startswith(DSML_TOOL_CLOSE_PLAIN):
                    self.buf = self.buf[len(DSML_TOOL_CLOSE_PLAIN):]
                    continue

                if self.buf.startswith("<think>"):
                    self.buf = self.buf[7:]
                    self.state = "THINKING"
                    if self.mode != "no":
                        out.append("event: content_block_start\ndata: " + json.dumps({
                            "type": "content_block_start",
                            "index": self.next_idx,
                            "content_block": {"type": "thinking", "thinking": ""}
                        }))
                elif tools_allowed and self.buf.startswith(DSML_TOOL_OPEN):
                    self.buf = self.buf[len(DSML_TOOL_OPEN):]
                    self.state = "TOOL"
                    self.tool_buf = ""
                    self.tool_is_dsml = True
                elif tools_allowed and self.buf.startswith("<tool_call>"):
                    self.buf = self.buf[11:]
                    self.state = "TOOL"
                    self.tool_buf = ""
                    self.tool_is_dsml = False
                else:
                    self.state = "TEXT"
                    self.has_content_block = True
                    self.native_text_open = True
                    out.append("event: content_block_start\ndata: " + json.dumps({
                        "type": "content_block_start",
                        "index": self.next_idx,
                        "content_block": {"type": "text", "text": ""}
                    }))

            elif self.state == "THINKING":
                end_think = self.buf.find("</think>")
                tool_call = -1
                tool_dsml = False
                if tools_allowed:
                    tool_call = self.buf.find("<tool_call>")
                    dsml_call = self.buf.find(DSML_TOOL_OPEN)
                    if dsml_call != -1 and (tool_call == -1 or dsml_call < tool_call):
                        tool_call = dsml_call
                        tool_dsml = True

                if tool_call != -1 and (end_think == -1 or tool_call < end_think):
                    chunk = self.buf[:tool_call]
                    if chunk:
                        self.thinking_buf += chunk
                        if self.mode != "no":
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "thinking_delta", "thinking": chunk}
                            }))
                    if self.mode != "no":
                        out.append("event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop",
                            "index": self.next_idx
                        }))
                        self.next_idx += 1
                    self.buf = self.buf[tool_call + (len(DSML_TOOL_OPEN) if tool_dsml else 11):]
                    self.state = "TOOL"
                    self.tool_buf = ""
                    self.tool_is_dsml = tool_dsml
                elif end_think != -1:
                    chunk = self.buf[:end_think]
                    if chunk:
                        self.thinking_buf += chunk
                        if self.mode != "no":
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "thinking_delta", "thinking": chunk}
                            }))
                    if self.mode != "no":
                        out.append("event: content_block_stop\ndata: " + json.dumps({
                            "type": "content_block_stop",
                            "index": self.next_idx
                        }))
                        self.next_idx += 1
                    self.buf = self.buf[end_think + 8:]
                    self.state = "INIT"
                else:
                    if len(self.buf) > _TAG_HOLDBACK:
                        chunk = self.buf[:-_TAG_HOLDBACK]
                        self.buf = self.buf[-_TAG_HOLDBACK:]
                        self.thinking_buf += chunk
                        if self.mode != "no":
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "thinking_delta", "thinking": chunk}
                            }))
                    break

            elif self.state == "TOOL":
                # Scan accumulated tool_buf from len(longest_closer) - 1 back so any closing tag
                # split across SSE deltas is detected without consuming trailing text.
                prev_len = len(self.tool_buf)
                self.tool_buf += self.buf
                if self.tool_is_dsml:
                    scan_from = max(0, prev_len - (len(DSML_TOOL_CLOSE_FULL) - 1))
                    idx_full = self.tool_buf.find(DSML_TOOL_CLOSE_FULL, scan_from)
                    idx_plain = self.tool_buf.find(DSML_TOOL_CLOSE_PLAIN, scan_from)
                    if idx_full != -1 and (idx_plain == -1 or idx_full <= idx_plain):
                        end_tool, closer_len = idx_full, len(DSML_TOOL_CLOSE_FULL)
                    elif idx_plain != -1:
                        end_tool, closer_len = idx_plain, len(DSML_TOOL_CLOSE_PLAIN)
                    else:
                        end_tool, closer_len = -1, 0
                else:
                    scan_from = max(0, prev_len - 11)
                    end_tool = self.tool_buf.find("</tool_call>", scan_from)
                    closer_len = 12

                if end_tool != -1:
                    raw_tool = self.tool_buf[:end_tool].strip()
                    self.buf = self.tool_buf[end_tool + closer_len:]
                    self.tool_buf = ""
                    self.state = "INIT"
                    for parsed in parse_tool_calls(raw_tool, self.schemas):
                        self._emit_tool_call(out, parsed)
                else:
                    self.buf = ""
                    if len(self.tool_buf) > _MAX_TOOL_BUF_BYTES:
                        raw_tool = self.tool_buf.strip()
                        self.tool_buf = ""
                        parsed_overflow = parse_tool_calls(raw_tool, self.schemas)
                        if parsed_overflow:
                            self.state = "INIT"
                            for parsed in parsed_overflow:
                                self._emit_tool_call(out, parsed)
                        else:
                            self.state = "TEXT"
                            self.has_content_block = True
                            out.append("event: content_block_start\ndata: " + json.dumps({
                                "type": "content_block_start",
                                "index": self.next_idx,
                                "content_block": {"type": "text", "text": ""}
                            }))
                    break

            elif self.state == "TEXT":
                tc_pos = self.buf.find("<tool_call>")
                if tc_pos != -1:
                    prefix_before_tc = self.buf[:tc_pos]
                    if prefix_before_tc and not prefix_before_tc.endswith(("\n", "\r", " ", "\t")):
                        tc_pos = -1
                dsml_pos = self.buf.find(DSML_TOOL_OPEN)
                if dsml_pos != -1:
                    prefix_before_dsml = self.buf[:dsml_pos]
                    if prefix_before_dsml and prefix_before_dsml.endswith("`"):
                        dsml_pos = -1
                if not self.schemas:
                    tc_pos = -1
                    dsml_pos = -1
                if tc_pos != -1 and dsml_pos != -1:
                    tool_call = min(tc_pos, dsml_pos)
                else:
                    tool_call = tc_pos if tc_pos != -1 else dsml_pos

                think = self.buf.find("<think>")
                split_pos = -1

                if tool_call != -1 and think != -1:
                    split_pos = min(tool_call, think)
                elif tool_call != -1:
                    split_pos = tool_call
                elif think != -1:
                    split_pos = think

                if split_pos != -1:
                    chunk = self.buf[:split_pos]
                    is_inter_tool_commentary = (
                        self.has_tool_call
                        and split_pos == tool_call
                        and len(chunk.strip()) < 350
                        and any(w in chunk.lower() for w in ["tool", "call", "output", "directive", "format", "assistant", "must", "xml"])
                    )
                    clean_chunk = chunk.replace("</think>", "").replace("<think>", "")
                    if clean_chunk and not is_inter_tool_commentary:
                        self.text_buf += clean_chunk
                        out.append("event: content_block_delta\ndata: " + json.dumps({
                            "type": "content_block_delta",
                            "index": self.next_idx,
                            "delta": {"type": "text_delta", "text": clean_chunk}
                        }))
                    out.append("event: content_block_stop\ndata: " + json.dumps({
                        "type": "content_block_stop",
                        "index": self.next_idx
                    }))
                    self.next_idx += 1
                    self.buf = self.buf[split_pos:]
                    self.state = "INIT"
                else:
                    if len(self.buf) > _TAG_HOLDBACK:
                        chunk = self.buf[:-_TAG_HOLDBACK]
                        self.buf = self.buf[-_TAG_HOLDBACK:]
                        clean_chunk = chunk.replace("</think>", "").replace("<think>", "")
                        if clean_chunk:
                            self.text_buf += clean_chunk
                            out.append("event: content_block_delta\ndata: " + json.dumps({
                                "type": "content_block_delta",
                                "index": self.next_idx,
                                "delta": {"type": "text_delta", "text": clean_chunk}
                            }))
                    break

        return out

    def _flush_normalizer(self):
        if getattr(self, "flushed_at_end", False):
            return []
        self.flushed_at_end = True
        out = []
        if self.state == "THINKING":
            if self.buf and self.mode != "no":
                self.thinking_buf += self.buf
                out.append("event: content_block_delta\ndata: " + json.dumps({
                    "type": "content_block_delta",
                    "index": self.next_idx,
                    "delta": {"type": "thinking_delta", "thinking": self.buf}
                }))
            if self.mode != "no":
                out.append("event: content_block_stop\ndata: " + json.dumps({
                    "type": "content_block_stop",
                    "index": self.next_idx
                }))
                self.next_idx += 1
        elif self.state == "TOOL":
            self.tool_buf += self.buf
            for parsed in parse_tool_calls(self.tool_buf.strip(), self.schemas):
                self._emit_tool_call(out, parsed)
        elif self.state == "TEXT":
            if self.buf:
                clean = self.buf.replace("</think>", "").replace("<think>", "")
                if clean:
                    self.text_buf += clean
                    out.append("event: content_block_delta\ndata: " + json.dumps({
                        "type": "content_block_delta",
                        "index": self.next_idx,
                        "delta": {"type": "text_delta", "text": clean}
                    }))
            out.append("event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop",
                "index": self.next_idx
            }))
            self.next_idx += 1
            self.native_text_open = False

        self.buf = ""
        self.tool_buf = ""
        self.state = "INIT"

        if not self.has_content_block:
            # THOUGHT CANNIBALIZATION RESCUE:
            # If the model placed its entire answer inside <think> and exited without emitting text or tools,
            # emit the actual thinking analysis instead of a bare placeholder.
            fallback = self.thinking_buf.strip()
            if not fallback:
                fallback = "Operation completed."

            out.append("event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start",
                "index": self.next_idx,
                "content_block": {"type": "text", "text": ""}
            }))
            out.append("event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta",
                "index": self.next_idx,
                "delta": {"type": "text_delta", "text": fallback}
            }))
            out.append("event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop",
                "index": self.next_idx
            }))
            self.next_idx += 1
            self.has_content_block = True
        elif not self.has_tool_call and self.text_buf.strip() in ("Completed.", "Done.", "Operation completed.", "Completato.", "Completato", "Fatto.", "Fatto", "Ecco.", "Finito.") and len(self.thinking_buf.strip()) > 80:
            # Model wrote detailed analysis in thinking and emitted only "Completato." in text!
            rescue_text = "\n\n" + self.thinking_buf.strip()
            out.append("event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start",
                "index": self.next_idx,
                "content_block": {"type": "text", "text": ""}
            }))
            out.append("event: content_block_delta\ndata: " + json.dumps({
                "type": "content_block_delta",
                "index": self.next_idx,
                "delta": {"type": "text_delta", "text": rescue_text}
            }))
            out.append("event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop",
                "index": self.next_idx
            }))
            self.next_idx += 1

        return out

def parse_slash_command(target_text: str):
    text_clean = target_text.strip()

    # 1. XML format from Claude Code slash command plugin
    if "<command-name>" in text_clean:
        cmd_m = re.search(r"<command-name>\s*([^<]+)\s*</command-name>", text_clean)
        cmd_name = cmd_m.group(1).strip() if cmd_m else ""
        args_m = re.search(r"<command-args>\s*([^<]*)\s*</command-args>", text_clean)
        args_raw = args_m.group(1).strip() if args_m else ""

        if "think" in cmd_name:
            tokens = args_raw.split(None, 1)
            first_tok = tokens[0].lower() if tokens else ""
            remainder = tokens[1] if len(tokens) > 1 else ""
            if first_tok in ("no", "med", "max"):
                return ("SET_MODE", first_tok, remainder)
            return ("STATUS", "", "")

    # 2. Plain text format: REQUIRE prefix [/!:] to avoid false positives
    m_set = re.search(r"^\s*[/!:]\s*think(?::think)?\s+(no|med|max)\b", text_clean, re.IGNORECASE)
    if m_set:
        mode = m_set.group(1).lower()
        remainder = text_clean[m_set.end():].strip()
        return ("SET_MODE", mode, remainder)

    m_status = re.search(r"^\s*[/!:]\s*think(?::think)?(?:\s+status|\s+help)?\s*$", text_clean, re.IGNORECASE)
    if m_status:
        return ("STATUS", "", "")

    return ("NORMAL", "", text_clean)

_READ_LINE_NUM_RE = re.compile(r"(?m)^[ \t]*\d+[\t→]")

def normalize_anthropic_messages(messages):
    """
    Preserves native Anthropic tool_use and tool_result blocks for llama-server /v1/messages
    while stripping Claude Code Read tool line-number prefixes (`1\\t`, `1→`) so source files
    enter the KV cache and RouteCrack 16K Hot Zone as verbatim code tokens.
    """
    if not isinstance(messages, list):
        return
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "tool_result":
                    tr_c = part.get("content")
                    if isinstance(tr_c, str) and ("1\t" in tr_c[:32] or "1→" in tr_c[:32]):
                        part["content"] = _READ_LINE_NUM_RE.sub("", tr_c)
    return

def sanitize_non_streaming_response(resp_dict, schemas=None, max_tool_calls=2):
    content = resp_dict.get("content", [])
    if not isinstance(content, list):
        return resp_dict
    new_content = []
    thinking_extra = []
    has_tool_call = False
    seen_tools = set()
    tool_count = 0

    def _append_parsed_calls(parsed_calls):
        nonlocal has_tool_call, tool_count
        for parsed in parsed_calls:
            name = parsed.get("name")
            args = parsed.get("input", {})
            sig = (name, json.dumps(args, sort_keys=True))
            if sig in seen_tools or tool_count >= max_tool_calls:
                continue
            seen_tools.add(sig)
            tool_count += 1
            has_tool_call = True
            new_content.append(_tool_use_block(parsed))

    for block in content:
        b_type = block.get("type")
        if b_type == "text":
            txt = block.get("text", "")
            if "</think>" in txt:
                parts = txt.split("</think>", 1)
                thinking_extra.append(parts[0].replace("<think>", "").strip())
                rem = parts[1].strip()
                if rem:
                    parsed_calls = parse_tool_calls(rem, schemas)
                    if parsed_calls:
                        _append_parsed_calls(parsed_calls)
                    else:
                        block["text"] = rem
                        new_content.append(block)
            elif "<think>" in txt:
                parts = txt.split("<think>", 1)
                if parts[0].strip():
                    new_content.append({"type": "text", "text": parts[0].strip()})
                thinking_extra.append(parts[1].strip())
            else:
                parsed_calls = parse_tool_calls(txt, schemas)
                if parsed_calls:
                    _append_parsed_calls(parsed_calls)
                else:
                    new_content.append(block)
        elif b_type == "tool_use":
            has_tool_call = True
            tool_count += 1
            new_content.append(block)
        else:
            new_content.append(block)

    if thinking_extra:
        found_th = False
        for b in new_content:
            if b.get("type") == "thinking":
                b["thinking"] += "\n" + "\n".join(thinking_extra)
                found_th = True
                break
        if not found_th:
            new_content.insert(0, {"type": "thinking", "thinking": "\n".join(thinking_extra), "signature": ""})

    if has_tool_call:
        resp_dict["stop_reason"] = "tool_use"

    resp_dict["content"] = new_content
    return resp_dict


def _build_modchip_telemetry_json() -> dict:
    """Collects live hardware, per-UID Modchip SHM, and llama-server /slots telemetry."""
    import stat
    import struct
    cpu_model = "CPU"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass

    anon_hp_gib = 0.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("AnonHugePages:"):
                    anon_hp_gib = round(int(line.split()[1]) / (1024 * 1024), 2)
                    break
    except Exception:
        pass

    shm_info = {
        "active": False,
        "l3_window": 2048,
        "attn_window": 0,
        "moe_top_k": 0,
        "rc_enabled": 1,
        "rc_gates": "84/92/96",
    }
    shm_candidates = []
    custom_shm = os.environ.get("MODCHIP_SHM_PATH", "").strip()
    if custom_shm:
        shm_candidates.append(custom_shm)
    xdg_dir = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if xdg_dir:
        shm_candidates.append(os.path.join(xdg_dir, "llama_modchip.bin"))
    euid = getattr(os, "geteuid", getattr(os, "getuid", lambda: 0))()
    shm_candidates.append(f"/dev/shm/llama_modchip_{euid}.bin")
    for shm_path in shm_candidates:
        if os.path.exists(shm_path):
            fd = -1
            try:
                fd = os.open(shm_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_uid != euid or (st.st_mode & 0o022) != 0:
                    continue
                buf = os.read(fd, 48)
                if len(buf) >= 48:
                    vals = struct.unpack("<IIiiiiiiiiii", buf[:48])
                    if vals[0] in (0x4D4F4443, 0x4D434850):
                        shm_info = {
                            "active": True,
                            "l3_window": vals[2],
                            "attn_window": vals[3],
                            "attn_sinks": vals[4],
                            "swa_stride": vals[5],
                            "rc_enabled": vals[7],
                            "rc_gates": f"{vals[8]}/{vals[9]}/{vals[10]}",
                            "moe_top_k": vals[11],
                        }
                        break
            except Exception:
                pass
            finally:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    slot_tokens = 0
    slot_state = "OFFLINE"
    backend_up = False
    active_model = ""
    try:
        req = urllib.request.Request(f"{UPSTREAM_BASE}/slots", method="GET")
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            slots = json.loads(resp.read().decode("utf-8"))
            backend_up = True
            if isinstance(slots, list) and slots:
                s0 = slots[0]
                slot_tokens = int(s0.get("n_past", s0.get("n_prompt_tokens", s0.get("n_decoded", 0))) or 0)
                slot_state = "GENERATING" if s0.get("is_processing") else "READY"
    except Exception:
        pass

    if backend_up:
        try:
            preq = urllib.request.Request(f"{UPSTREAM_BASE}/props", method="GET")
            with urllib.request.urlopen(preq, timeout=0.5) as presp:
                pdata = json.loads(presp.read().decode("utf-8"))
                active_model = os.path.basename(str(pdata.get("model_path") or ""))
        except Exception:
            pass
        if not active_model:
            try:
                mreq = urllib.request.Request(f"{UPSTREAM_BASE}/v1/models", method="GET")
                with urllib.request.urlopen(mreq, timeout=0.5) as mresp:
                    mdata = json.loads(mresp.read().decode("utf-8"))
                    mlist = mdata.get("data") or mdata.get("models") or []
                    if mlist and isinstance(mlist[0], dict):
                        active_model = os.path.basename(str(mlist[0].get("id") or mlist[0].get("name") or ""))
            except Exception:
                pass
    if not active_model:
        def_mfile = os.path.join(_resolve_workspace_dir(), "run", "default_model")
        if os.path.isfile(def_mfile):
            try:
                with open(def_mfile, "r", encoding="utf-8") as dmf:
                    active_model = os.path.basename(dmf.readline().strip())
            except Exception:
                pass

    return {
        "cpu_model": cpu_model,
        "anon_hugepages_gib": anon_hp_gib,
        "shm": shm_info,
        "slot0_tokens": slot_tokens,
        "slot0_state": slot_state,
        "backend_up": backend_up,
        "active_model": active_model,
        "upstream": UPSTREAM_BASE,
    }


WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>llama-modchip v3.1.0 — MaxQ 16GB/32GB/64GB CPU + RouteCrack 2.0 HUD</title>
<style>
  :root {
    --bg: #0b0e14;
    --panel: #121722;
    --panel-alt: #171e2c;
    --border: #253047;
    --text: #e6edf3;
    --muted: #8b949e;
    --accent: #3fb950;
    --accent-cyan: #58a6ff;
    --accent-gold: #d29922;
    --accent-purple: #bc8cff;
    --font-mono: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
  }
  header {
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    padding: 10px 18px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 15px;
    color: var(--text);
  }
  .badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 99px;
    background: rgba(63, 185, 80, 0.15);
    color: var(--accent);
    border: 1px solid rgba(63, 185, 80, 0.4);
    font-family: var(--font-mono);
  }
  .hud {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
    gap: 8px;
    background: #0d121b;
    border-bottom: 1px solid var(--border);
    padding: 8px 18px;
    font-family: var(--font-mono);
  }
  .hud-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
  }
  .hud-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .hud-val { font-size: 15px; font-weight: 700; margin-top: 2px; color: var(--accent); }
  .hud-val.cyan { color: var(--accent-cyan); }
  .hud-val.gold { color: var(--accent-gold); }
  .hud-val.purple { color: var(--accent-purple); }
  .hud-sub { font-size: 10px; color: var(--muted); margin-top: 1px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 8px 18px;
    background: var(--panel-alt);
    border-bottom: 1px solid var(--border);
    align-items: center;
  }
  .presets span { font-size: 12px; color: var(--muted); font-family: var(--font-mono); }
  .preset-btn {
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 4px 10px;
    font-size: 12px;
    font-family: var(--font-mono);
    cursor: pointer;
    transition: all 0.15s;
  }
  .preset-btn:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
  #chat {
    flex: 1;
    overflow-y: auto;
    padding: 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .msg {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
    padding: 12px 16px;
    border-radius: 8px;
    line-height: 1.55;
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg.user {
    background: #1b2538;
    border: 1px solid #2e4063;
    align-self: flex-end;
  }
  .msg.assistant {
    background: var(--panel);
    border: 1px solid var(--border);
  }
  .msg-meta {
    margin-top: 8px;
    padding-top: 6px;
    border-top: 1px dashed var(--border);
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
  }
  details.think-box {
    background: #0d121a;
    border: 1px solid #232d42;
    border-radius: 6px;
    padding: 6px 10px;
    margin-bottom: 8px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--muted);
  }
  details.think-box summary { cursor: pointer; color: var(--accent-purple); font-weight: 600; }
  footer {
    background: var(--panel);
    border-top: 1px solid var(--border);
    padding: 12px 18px;
  }
  .input-row {
    max-width: 900px;
    margin: 0 auto;
    display: flex;
    gap: 8px;
  }
  textarea {
    flex: 1;
    background: #0b0e14;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 12px;
    font-family: inherit;
    font-size: 14px;
    resize: none;
    height: 46px;
  }
  textarea:focus { outline: none; border-color: var(--accent-cyan); }
  select, button.send-btn {
    background: var(--accent);
    color: #061108;
    font-weight: 700;
    font-family: var(--font-mono);
    border: none;
    border-radius: 6px;
    padding: 0 16px;
    cursor: pointer;
    font-size: 13px;
  }
  select {
    background: var(--panel-alt);
    color: var(--text);
    border: 1px solid var(--border);
    font-weight: 500;
  }
  button.send-btn:disabled { opacity: 0.5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span>♟️ llama-modchip v3.1.0</span>
    <span class="badge" id="status-badge">CONNECTING...</span>
  </div>
  <div style="font-family: var(--font-mono); font-size: 12px; color: var(--muted);" id="cpu-banner">
    Probing CPU &amp; per-UID Modchip SHM...
  </div>
</header>

<section class="hud">
  <div class="hud-card">
    <div class="hud-label">⚡ Decode Speed</div>
    <div class="hud-val" id="hud-tps">-- tok/s</div>
    <div class="hud-sub" id="hud-peak">Peak burst: -- tok/s</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">⏱️ TTFT &amp; Prefill</div>
    <div class="hud-val cyan" id="hud-ttft">-- ms</div>
    <div class="hud-sub" id="hud-pp">Prefill: -- tok/s</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">♟️ RouteCrack 2.0</div>
    <div class="hud-val purple" id="hud-spec">Active (L1 Arbiter)</div>
    <div class="hud-sub" id="hud-spec-sub">MTP/RC + 2-Zone + Zobrist TT</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">💾 KV &amp; Mamba-2 Slot 0</div>
    <div class="hud-val gold" id="hud-slot">0 tok</div>
    <div class="hud-sub" id="hud-hp">2MB THP: -- GiB</div>
  </div>
  <div class="hud-card">
    <div class="hud-label">🔧 Modchip Locks</div>
    <div class="hud-val cyan" id="hud-locks">MoE Top-K=Auto</div>
    <div class="hud-sub" id="hud-l3">L3 Win: 2048t | Full 256K</div>
  </div>
</section>

<div class="presets">
  <span>1-Click Benchmarks:</span>
  <button class="preset-btn" onclick="runPreset('Write a high-performance C++20 lock-free ring buffer using std::atomic with cache-line alignment (alignas(64)) to prevent false sharing, and include a brief usage example.')">⚡ C++20 Lock-Free Ring Buffer</button>
  <button class="preset-btn" onclick="runPreset('Refactor the code above to add batch push_bulk() and pop_bulk() methods with zero-copy span views, preserving the exact class structure and comments.')">♟️ In-Context Refactor (RouteCrack 2.0 Burst)</button>
  <button class="preset-btn" onclick="runPreset('Explain in 5 concise bullet points why 2MB Transparent HugePages, AVX-512 VNNI tile repacking, and the RouteCrack 2.0 L1 Micro-Arbiter accelerate Sparse MoE and Hybrid GDN/Mamba-2 inference on CPU.')">🧠 Systems Architecture Summary</button>
  <button class="preset-btn" onclick="clearChat()" style="margin-left:auto;">🗑️ Clear Context</button>
</div>

<main id="chat">
  <div class="msg assistant">👋 <b>Welcome to the llama-modchip v3.1.0 Live Console!</b>
Send any prompt below or click one of the <b>1-Click Benchmarks</b> above to watch real-time token streaming and measure your CPU's <b>Prefill tok/s</b>, <b>Decode tok/s</b>, and <b>RouteCrack 2.0 speculative acceptance</b>.</div>
</main>

<footer>
  <div class="input-row">
    <select id="think-mode" title="Reasoning Budget">
      <option value="0" selected>⚡ No Think (Fastest)</option>
      <option value="4096">🧠 Med Think (4K)</option>
      <option value="-1">🔬 Max Think</option>
    </select>
    <textarea id="prompt" placeholder="Ask anything or paste code to refactor... (Enter to send, Shift+Enter for newline)"></textarea>
    <button class="send-btn" id="send-btn" onclick="sendMessage()">Send ↵</button>
  </div>
</footer>

<script>
let historyMsgs = [];
let peakTps = 0.0;
let activeModel = '';

async function pollTelemetry() {
  try {
    const r = await fetch('/api/modchip');
    if (!r.ok) return;
    const d = await r.json();
    if (d.active_model) activeModel = d.active_model;
    const badge = document.getElementById('status-badge');
    if (d.backend_up) {
      badge.textContent = d.slot0_state || 'ONLINE';
      badge.style.color = '#3fb950';
    } else {
      badge.textContent = 'WAITING FOR LLAMA-SERVER';
      badge.style.color = '#d29922';
    }
    const modelTag = d.active_model ? ` | Model: ${d.active_model}` : '';
    document.getElementById('cpu-banner').textContent = `${d.cpu_model}${modelTag} | Upstream: ${d.upstream}`;
    document.getElementById('hud-slot').textContent = `${(d.slot0_tokens || 0).toLocaleString()} tok`;
    document.getElementById('hud-hp').textContent = `2MB THP: ${d.anon_hugepages_gib} GiB`;
    if (d.shm) {
      const moeLabel = (!d.shm.moe_top_k || d.shm.moe_top_k === 0) ? 'Auto' : String(d.shm.moe_top_k);
      document.getElementById('hud-locks').textContent = `MoE Top-K=${moeLabel} | Gates ${d.shm.rc_gates}`;
      const swaMode = d.shm.attn_window === 0 ? 'Full 256K Context' : `SWA ${d.shm.attn_window}t`;
      document.getElementById('hud-l3').textContent = `L3 NTA: ${d.shm.l3_window}t | ${swaMode}`;
    }
  } catch (e) {}
}
setInterval(pollTelemetry, 2500);
pollTelemetry();

function clearChat() {
  historyMsgs = [];
  document.getElementById('chat').innerHTML = '<div class="msg assistant">Context cleared. Ready for new benchmark or prompt.</div>';
}

function runPreset(text) {
  document.getElementById('prompt').value = text;
  sendMessage();
}

document.getElementById('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

async function sendMessage() {
  const input = document.getElementById('prompt');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  const btn = document.getElementById('send-btn');
  btn.disabled = true;

  const chat = document.getElementById('chat');
  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.textContent = text;
  chat.appendChild(uDiv);

  const aDiv = document.createElement('div');
  aDiv.className = 'msg assistant';
  const contentSpan = document.createElement('div');
  const metaDiv = document.createElement('div');
  metaDiv.className = 'msg-meta';
  metaDiv.textContent = '⏳ Evaluating prompt...';
  aDiv.appendChild(contentSpan);
  aDiv.appendChild(metaDiv);
  chat.appendChild(aDiv);
  chat.scrollTop = chat.scrollHeight;

  historyMsgs.push({role: 'user', content: text});
  const budget = parseInt(document.getElementById('think-mode').value, 10);

  const tStart = performance.now();
  let tFirst = null;
  let tokCount = 0;
  let fullText = '';
  let thinkText = '';

  try {
    if (!activeModel) {
      try {
        const mr = await fetch('/api/modchip');
        if (mr.ok) { const md = await mr.json(); if (md.active_model) activeModel = md.active_model; }
      } catch (e) {}
    }
    const lowModel = (activeModel || '').toLowerCase();
    const isQwen38_32GB = (lowModel.includes('qwen3.8') || lowModel.includes('qwen38') || lowModel.includes('qwen4exp')) && lowModel.includes('32gb');
    const reqBody = {
      model: activeModel || 'modchip-local',
      messages: historyMsgs,
      stream: true,
      max_tokens: (budget !== 0 ? 16384 : 4096),
      reasoning_budget: budget,
      chat_template_kwargs: {enable_thinking: budget !== 0},
      timings_per_token: true
    };
    if (isQwen38_32GB) {
      reqBody.temperature = 0.6;
      reqBody.top_p = 0.95;
      reqBody.top_k = 20;
      reqBody.presence_penalty = 0.2;
      reqBody.repeat_penalty = 1.05;
    } else {
      reqBody.temperature = 0.0;
    }
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(reqBody)
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalTimings = null;

    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});
      const lines = buffer.split('\\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (payload === '[DONE]') continue;
        try {
          const ev = JSON.parse(payload);
          if (ev.timings) finalTimings = ev.timings;
          const delta = (ev.choices && ev.choices[0] && ev.choices[0].delta) || {};
          const rc = delta.reasoning_content || '';
          const cc = delta.content || '';
          if (rc || cc) {
            const now = performance.now();
            if (!tFirst) {
              tFirst = now;
              document.getElementById('hud-ttft').textContent = `${Math.round(tFirst - tStart)} ms`;
            }
            tokCount++;
            if (rc) thinkText += rc;
            if (cc) fullText += cc;

            const elapsedSec = Math.max(0.05, (now - tFirst) / 1000.0);
            if (tokCount > 2) {
              const liveTps = tokCount / elapsedSec;
              if (liveTps > peakTps && tokCount >= 5) peakTps = liveTps;
              document.getElementById('hud-tps').textContent = `${liveTps.toFixed(1)} tok/s`;
              document.getElementById('hud-peak').textContent = `Peak burst: ${peakTps.toFixed(1)} tok/s`;
              metaDiv.textContent = `⚡ Streaming: ${liveTps.toFixed(1)} tok/s (${tokCount} chunks)`;
            }

            let html = '';
            if (thinkText) {
              html += `<details class="think-box"><summary>🧠 Reasoning Trace (${thinkText.length} chars)</summary>${thinkText.replace(/</g, '&lt;')}</details>`;
            }
            html += fullText.replace(/</g, '&lt;');
            contentSpan.innerHTML = html;
            chat.scrollTop = chat.scrollHeight;
          }
        } catch (e) {}
      }
    }

    historyMsgs.push({role: 'assistant', content: fullText || thinkText});
    if (finalTimings) {
      const decTps = finalTimings.predicted_per_second || 0;
      const ppTps = finalTimings.prompt_per_second || 0;
      const nPred = finalTimings.predicted_n || tokCount;
      if (decTps > peakTps) peakTps = decTps;
      document.getElementById('hud-tps').textContent = `${decTps.toFixed(2)} tok/s`;
      document.getElementById('hud-peak').textContent = `Peak burst: ${peakTps.toFixed(2)} tok/s`;
      if (ppTps > 0) document.getElementById('hud-pp').textContent = `Prefill: ${ppTps.toFixed(1)} tok/s`;

      let specStr = '';
      if (finalTimings.draft_n && finalTimings.draft_n > 0) {
        const accPct = (100.0 * (finalTimings.draft_n_accepted || 0) / finalTimings.draft_n).toFixed(1);
        document.getElementById('hud-spec').textContent = `${accPct}% Accepted`;
        document.getElementById('hud-spec-sub').textContent = `${finalTimings.draft_n_accepted}/${finalTimings.draft_n} draft tokens`;
        specStr = ` | ♟️ RouteCrack 2.0: ${finalTimings.draft_n_accepted}/${finalTimings.draft_n} (${accPct}% accepted)`;
      }
      metaDiv.textContent = `⚡ Decode: ${decTps.toFixed(2)} tok/s (${nPred} tok) | 🧠 Prefill: ${ppTps.toFixed(1)} tok/s | ⏱️ TTFT: ${Math.round((tFirst || performance.now()) - tStart)} ms${specStr}`;
    } else {
      const totalSec = Math.max(0.05, (performance.now() - (tFirst || tStart)) / 1000.0);
      const avgTps = tokCount / totalSec;
      metaDiv.textContent = `⚡ Decode: ~${avgTps.toFixed(1)} tok/s (${tokCount} chunks) | ⏱️ TTFT: ${Math.round((tFirst || performance.now()) - tStart)} ms`;
    }
    pollTelemetry();
  } catch (err) {
    metaDiv.textContent = `❌ Error contacting backend: ${err}`;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_cors_headers(self):
        self.send_header("X-Modchip-Gateway", GATEWAY_BUILD_HASH)
        if not GATEWAY_CORS_ORIGIN:
            return
        self.send_header("Access-Control-Allow-Origin", GATEWAY_CORS_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")

    def _send_json_error(self, code: int, message: str):
        body = json.dumps({"error": {"message": message, "type": "invalid_request_error", "code": code}}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        norm_path = _normalize_gateway_path(self.path)
        if norm_path is None:
            self._send_json_error(400, "Malformed request path")
            return
        self.path = norm_path
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def _check_auth(self) -> bool:
        if not GATEWAY_API_KEY:
            return True
        import hmac
        auth_hdr = self.headers.get("Authorization", "")
        token = auth_hdr[7:].strip() if auth_hdr.lower().startswith("bearer ") else self.headers.get("x-api-key", "").strip()
        if hmac.compare_digest(token.encode("utf-8"), GATEWAY_API_KEY.encode("utf-8")):
            return True
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(b'{"error":"Unauthorized: set Authorization: Bearer <GATEWAY_API_KEY>"}')
        return False

    def do_GET(self):
        norm_path = _normalize_gateway_path(self.path)
        if norm_path is None:
            self._send_json_error(400, "Malformed request path")
            return
        self.path = norm_path
        clean_path = self.path.split("?", 1)[0]
        if clean_path not in ("/", "/index.html", "/ui", "/health") and not self._check_auth():
            return
        if clean_path in ("/", "/index.html", "/ui"):
            body = WEB_UI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
            return
        if clean_path == "/api/modchip":
            body = json.dumps(_build_modchip_telemetry_json()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward_request("GET")

    def _send_anthropic_text_response(self, text, is_stream=False, model_name=None):
        resolved_model = model_name or _detect_active_model(UPSTREAM_BASE)
        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_cors_headers()
            self.end_headers()

            events = [
                ("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": "msg_modchip_cmd",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": resolved_model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }
                }),
                ("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                }),
                ("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text}
                }),
                ("content_block_stop", {
                    "type": "content_block_stop",
                    "index": 0
                }),
                ("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": max(1, len(text.split()))}
                }),
                ("message_stop", {
                    "type": "message_stop"
                })
            ]
            for ev_type, ev_data in events:
                chunk = f"event: {ev_type}\ndata: {json.dumps(ev_data)}\n\n".encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()
        else:
            resp = {
                "id": "msg_modchip_cmd",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
                "model": resolved_model,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": max(1, len(text.split()))}
            }
            resp_body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(resp_body)

    def do_POST(self):
        global _last_endpoint_was_chat
        norm_path = _normalize_gateway_path(self.path)
        if norm_path is None:
            self._send_json_error(400, "Malformed request path")
            return
        self.path = norm_path
        if not self._check_auth():
            return
        max_body_bytes = int(os.environ.get("GATEWAY_MAX_BODY_BYTES", str(64 * 1024 * 1024)))
        raw_cl = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_cl)
        except ValueError:
            self._send_json_error(400, "Invalid Content-Length header")
            return
        if length < 0 or length > max_body_bytes:
            self._send_json_error(413, f"Payload size {length} outside allowed bounds [0, {max_body_bytes}]")
            return
        body_bytes = self.rfile.read(length) if length else b""

        # Session tracking: bind session header and client IP
        session_id = self.headers.get("X-Session-ID") or self.headers.get("X-Modchip-Session")
        client_ip = self.client_address[0]
        init_mode = (self.headers.get("X-Init-Think") or self.headers.get("X-Modchip-Init-Think") or "").lower()

        if session_id:
            active_mode = get_session_mode(session_id, init_mode)
            set_session_mode(client_ip, active_mode)
        else:
            active_mode = get_session_mode(client_ip, init_mode)

        log_sess = session_id[:8] if session_id else client_ip
        target_upstream = UPSTREAM_BASE
        harp_meta = None
        tool_schemas = {}

        clean_post_path = self.path.split("?", 1)[0]
        if clean_post_path == "/v1/messages":
            try:
                data = json.loads(body_bytes.decode("utf-8"))
                messages = data.get("messages", [])
                is_stream = bool(data.get("stream", False))

                # Check recent user messages for slash command or think switch
                action, new_mode, remainder = "NORMAL", "", ""
                target_user_msg = None
                target_part = None

                for msg in reversed(messages[-4:]):
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", "")
                    cand_text = ""
                    cand_part = None
                    if isinstance(content, str):
                        cand_text = content
                    elif isinstance(content, list):
                        for part in reversed(content):
                            if isinstance(part, dict) and part.get("type") == "text":
                                t = part.get("text", "").strip()
                                if not t.startswith("<system-reminder>"):
                                    if not cand_text:
                                        cand_text = t
                                        cand_part = part
                                    act_chk, _, _ = parse_slash_command(t)
                                    if act_chk in ("SET_MODE", "STATUS"):
                                        cand_text = t
                                        cand_part = part
                                        break
                    act, mode, rem = parse_slash_command(cand_text)
                    if act in ("SET_MODE", "STATUS"):
                        action, new_mode, remainder = act, mode, rem
                        target_user_msg = msg
                        target_part = cand_part
                        break

                if action == "STATUS":
                    status_msg = (
                        f"🧠 **Current Thinking Mode: `{active_mode.upper()}`** (Host: `{client_ip}`)\n\n"
                        "Available commands:\n"
                        "• `/think:think no`  — No thinking (immediate direct response)\n"
                        "• `/think:think med` — Medium thinking (1,024-token budget)\n"
                        "• `/think:think max` — Max thinking (unrestricted)\n"
                    )
                    print(f"[Gateway] [{log_sess}] Responded to status query: {active_mode.upper()}", flush=True)
                    self._send_anthropic_text_response(status_msg, is_stream=is_stream)
                    return

                if action == "SET_MODE":
                    if session_id:
                        set_session_mode(session_id, new_mode)
                    set_session_mode(client_ip, new_mode)
                    active_mode = new_mode
                    desc = {
                        "no": "NO THINKING (Immediate direct response)",
                        "med": "MEDIUM THINKING (1,024-token budget)",
                        "max": "MAX THINKING (Unrestricted)"
                    }.get(new_mode, new_mode)
                    print(f"[Gateway] [{log_sess}] Mode switched to: {active_mode.upper()} ({desc})", flush=True)

                    if not remainder:
                        ack_msg = f"⚡ *Thinking Mode set to **{new_mode.upper()}** ({desc})*"
                        self._send_anthropic_text_response(ack_msg, is_stream=is_stream)
                        return
                    else:
                        content = target_user_msg.get("content", "")
                        is_list = isinstance(content, list)
                        if is_list and target_part:
                            target_part["text"] = remainder
                            target_user_msg["content"] = [
                                p for p in content
                                if not (isinstance(p, dict) and p.get("type") == "text" and "slash command /think" in p.get("text", ""))
                            ]
                        else:
                            target_user_msg["content"] = remainder

                # Sanitize system reminders & mid-conversation system injections:
                # Strip <system-reminder>/<total_tokens> blocks by default (GATEWAY_STRIP_REMINDERS=1) to keep Slot 0 prefixes invariant.
                # Set GATEWAY_STRIP_REMINDERS=0 to retain client system reminders.
                strip_reminders = (
                    os.environ.get("MODCHIP_STRIP_SYSTEM_REMINDERS") == "1"
                    or GATEWAY_STRIP_REMINDERS
                )
                extracted_memory_blocks = []
                filtered_messages = []
                for idx_m, msg in enumerate(messages):
                    role_m = msg.get("role", "")
                    if role_m == "system" and strip_reminders:
                        continue
                    if role_m == "user":
                        u_content = msg.get("content")
                        if isinstance(u_content, list):
                            user_parts = []
                            for part in u_content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    txt = part.get("text", "")
                                    if "<system-reminder>" in txt or "<total_tokens>" in txt:
                                        if idx_m == 0 and "Contents of " in txt and ("MEMORY.md" in txt or "CLAUDE.md" in txt):
                                            clean_mem = re.sub(r"\n# currentDate\n.*?(?=</system-reminder>)", "", txt, flags=re.S)
                                            extracted_memory_blocks.append(clean_mem)
                                        elif not strip_reminders:
                                            user_parts.append(part)
                                    else:
                                        user_parts.append(part)
                                else:
                                    user_parts.append(part)
                            if user_parts:
                                msg["content"] = user_parts
                            else:
                                continue
                        elif isinstance(u_content, str) and ("<system-reminder>" in u_content or "<total_tokens>" in u_content):
                            if strip_reminders:
                                cleaned_str = re.sub(r"<system-reminder>[\s\S]*?</system-reminder>", "", u_content)
                                cleaned_str = re.sub(r"<total_tokens>[\s\S]*?</total_tokens>", "", cleaned_str).strip()
                                if cleaned_str:
                                    msg["content"] = cleaned_str
                                else:
                                    continue
                    filtered_messages.append(msg)
                messages[:] = filtered_messages
                data["messages"] = messages

                # Normalize Anthropic tool calling for llama-server
                normalize_anthropic_messages(messages)

                # Intercept Claude Code background utility probes (topic/title classifiers, file-path extractors,
                # or explicit GATEWAY_INTERCEPT_ALL_PROBES=1) so they never overwrite or block Slot 0,
                # while still forwarding legitimate non-streaming or tool-less /v1/messages requests.
                is_direct = self.headers.get("X-Direct") or self.headers.get("X-Modchip-Direct")
                is_prewarm = self.headers.get("X-Prewarm") or self.headers.get("X-Modchip-Prewarm")
                if (not data.get("tools") or not is_stream) and not is_direct and not is_prewarm:
                    raw_str = json.dumps(data)
                    is_claude_bg_probe = (
                        os.environ.get("GATEWAY_INTERCEPT_ALL_PROBES") == "1"
                        or "isNewTopic" in raw_str
                        or "Please write a 5-10 word title" in raw_str
                        or "Analyze if this message indicates a new conversation topic" in raw_str
                        or "Extract any file paths" in raw_str
                        or "claude_code_bg_probe" in raw_str
                    )
                    if is_claude_bg_probe:
                        if "isNewTopic" in raw_str:
                            self._send_anthropic_text_response('{"isNewTopic": false, "title": "Session"}', is_stream=is_stream)
                        else:
                            self._send_anthropic_text_response("OK", is_stream=is_stream)
                        print(f"[Gateway] [{log_sess}] Intercepted background probe (tools={bool(data.get('tools'))}, stream={is_stream}) locally (0ms, Slot 0 preserved)", flush=True)
                        return

                # Preserve pre-extracted memory blocks if replaying from claude_warmup_payload.json
                if not extracted_memory_blocks and isinstance(data.get("_ preserved_memory"), list):
                    extracted_memory_blocks = data.pop("_ preserved_memory")
                elif "_ preserved_memory" in data:
                    data.pop("_ preserved_memory", None)

                # Resolve active model alias dynamically so Qwen3.8 / DeepSeek-V4 / Nemotron keep their true identity
                resolved_model = _detect_active_model(UPSTREAM_BASE, fallback=str(data.get("model") or "modchip-local"))

                # SINGLE CENTRALIZED SYSTEM PROMPT (gateway/system_prompt.md)
                _sys_prompt_path = os.environ.get(
                    "GATEWAY_SYSTEM_PROMPT",
                    os.environ.get("SYSTEM_PROMPT_PATH", os.path.join(_GATEWAY_DIR, "system_prompt.md")),
                )
                try:
                    with open(_sys_prompt_path, "r", encoding="utf-8") as _spf:
                        master_system_prompt = _spf.read().strip()
                except Exception:
                    master_system_prompt = "You are a senior software engineering and systems architecture assistant running locally on llama-modchip."

                # Keep data["system"] invariant in the top-level Anthropic `system` field (never inject role="system" into Anthropic messages[]).
                # Preserve client-supplied system prompts by default unless MODCHIP_OVERRIDE_SYSTEM_PROMPT=1 is set.
                unified_system = []
                orig_system = data.get("system")
                if orig_system and os.environ.get("MODCHIP_OVERRIDE_SYSTEM_PROMPT") != "1":
                    if isinstance(orig_system, list):
                        unified_system.extend(orig_system)
                    elif isinstance(orig_system, str):
                        unified_system.append({"type": "text", "text": orig_system})
                else:
                    raw_in_sys = json.dumps(data.get("system", ""))
                    if "x-anthropic-billing-header" in raw_in_sys:
                        m_bh = re.search(r"(x-anthropic-billing-header:[^\"\\]+)", raw_in_sys)
                        bh_str = m_bh.group(1) if m_bh else "x-anthropic-billing-header: cc_version=2.1.283.49b; cc_entrypoint=cli;"
                        unified_system.append({"type": "text", "text": bh_str})
                    unified_system.append({"type": "text", "text": master_system_prompt})
                if extracted_memory_blocks:
                    unified_system.append({"type": "text", "text": "\n\n".join(extracted_memory_blocks)})
                data["system"] = unified_system

                # RouteCrack-KV: Opt-in capture of Claude Code base prefix (CAPTURE_WARMUP_PAYLOAD=1, mode 0600)
                _warmup_path = os.environ.get("WARMUP_PAYLOAD_PATH", os.path.join(_resolve_workspace_dir(), "cache", "claude_warmup_payload.json"))
                if (
                    os.environ.get("CAPTURE_WARMUP_PAYLOAD") == "1"
                    and "system" in data
                    and "tools" in data
                    and len(data["tools"]) >= 40
                    and not os.path.exists(_warmup_path)
                ):
                    try:
                        os.makedirs(os.path.dirname(_warmup_path), mode=0o700, exist_ok=True)
                        _wfd = os.open(_warmup_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                        with os.fdopen(_wfd, "w", encoding="utf-8") as _wf:
                            json.dump({
                                "model": resolved_model,
                                "system": data["system"],
                                "_ preserved_memory": extracted_memory_blocks,
                                "tools": data["tools"],
                                "messages": [{"role": "user", "content": "ping"}],
                                "max_tokens": 1
                            }, _wf)
                        print(f"[Gateway] [HARP-Prewarm] Captured raw Claude Code base payload to {_warmup_path} (mode 0600)", flush=True)
                    except Exception as _werr:
                        print(f"[Gateway] [HARP-Prewarm] Notice: {_werr}", flush=True)

                # Ensure tools have both input_schema and parameters
                if "tools" in data and isinstance(data["tools"], list):
                    for tool in data["tools"]:
                        if isinstance(tool, dict) and "input_schema" in tool and "parameters" not in tool:
                            tool["parameters"] = tool["input_schema"]
                        if isinstance(tool, dict) and tool.get("name") and isinstance(tool.get("parameters"), dict):
                            tool_schemas[tool["name"]] = tool["parameters"].get("properties", {})

                    # Optional Compact Tool Table (22,225 -> 2,940 prefix tokens for DeepSeek-V4-Flash)
                    use_compact_tools = (
                        os.environ.get("MODCHIP_COMPACT_TOOLS") == "1"
                        or "dsv4" in SLOT0_SNAPSHOT_NAME.lower()
                    )
                    if use_compact_tools and len(data["tools"]) >= 15:
                        compact_lines = [
                            "## Available Tools (invoke via <｜DSML｜tool_calls><｜DSML｜invoke name=\"TOOL_NAME\"><｜DSML｜parameter name=\"PARAM\" string=\"true|false\">VAL</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>):"
                        ]
                        for tool in data["tools"]:
                            if not isinstance(tool, dict):
                                continue
                            tname = tool.get("name", "")
                            schema = tool.get("input_schema") or tool.get("parameters") or {}
                            props = schema.get("properties") or {}
                            reqs = set(schema.get("required") or [])
                            psigs = [f"{pk}{'*' if pk in reqs else ''}:{pv.get('type','any') if isinstance(pv,dict) else 'any'}" for pk, pv in props.items()]
                            compact_lines.append(f"- {tname}({', '.join(psigs)})")
                        data["system"].append({"type": "text", "text": "\n".join(compact_lines)})
                        data.pop("tools", None)

                # Enforce safe upper bound on max_tokens (allow up to 16384 when thinking is enabled in max/med mode)
                max_tok_cap = 16384 if active_mode == "max" else (8192 if active_mode == "med" else 4096)
                cur_max = data.get("max_tokens")
                if cur_max is None:
                    data["max_tokens"] = 16384 if active_mode == "max" else 4096
                elif cur_max > max_tok_cap:
                    data["max_tokens"] = max_tok_cap
                # repeat_penalty omitted so <|im_end|> stop token is never penalized

                # Configure reasoning budget for llama-server
                data.setdefault("chat_template_kwargs", {})

                if active_mode == "no":
                    data["chat_template_kwargs"]["enable_thinking"] = False
                    data["reasoning_budget"] = 0
                    data["reasoning_budget_tokens"] = 0
                    data["thinking_budget_tokens"] = 0
                elif active_mode == "med":
                    data["chat_template_kwargs"]["enable_thinking"] = True
                    data["reasoning_budget"] = 1024
                    data["reasoning_budget_tokens"] = 1024
                    data["thinking_budget_tokens"] = 1024
                elif active_mode == "max":
                    data["chat_template_kwargs"]["enable_thinking"] = True
                    data["reasoning_budget"] = -1
                    data["reasoning_budget_tokens"] = -1
                    data["thinking_budget_tokens"] = -1

                target_upstream = UPSTREAM_BASE
                data["model"] = resolved_model

                # Auto-restore Slot 0 snapshot in ~18.9-32 ms if Slot 0 was evicted or is cold
                if (len(messages) <= 2 or _last_endpoint_was_chat or (isinstance(data.get("tools"), list) and len(data["tools"]) >= 20)) and not is_prewarm:
                    _ensure_slot0_restored(target_upstream)
                    _last_endpoint_was_chat = False



                # RouteCrack-KV: HARP-Cache Prefix Identification & Zero-Prefill Ingestion
                harp_meta = {"hit": False, "cached_tokens": 0, "saved_sec": 0.0}
                if _harp_engine is not None:
                    try:
                        prefix_elements = []
                        if "system" in data:
                            prefix_elements.append(str(data["system"]))
                        if "tools" in data:
                            prefix_elements.append(json.dumps(data["tools"], sort_keys=True))
                        if len(messages) > 1:
                            for m in messages[:-1]:
                                prefix_elements.append(json.dumps(m, sort_keys=True))
                        prefix_text = "\n".join(prefix_elements)
                        sim_tokens = [
                            int.from_bytes(hashlib.sha256(chunk.encode("utf-8")).digest()[:4], "little") & 0x7fffffff
                            for chunk in [prefix_text[i:i+4] for i in range(0, len(prefix_text), 4)]
                        ]
                        if len(sim_tokens) >= 32:
                            cached_tokens, matched_pages, suffix = _harp_engine.resolve_prefix(sim_tokens)
                            if len(sim_tokens) - cached_tokens >= 64:
                                new_pages = _harp_engine.store_prefix(sim_tokens)
                                print(f"[Gateway] [HARP-Cache] Registered new prefix: {len(sim_tokens)} tokens ({new_pages} pages in /dev/shm)", flush=True)
                            else:
                                harp_meta["hit"] = True
                                harp_meta["cached_tokens"] = cached_tokens
                                harp_meta["saved_sec"] = cached_tokens / 250.0
                                print(f"[Gateway] [HARP-Cache] ⚡ ZERO-PREFILL HIT: {cached_tokens} tokens ({len(matched_pages)} pages). Saved CPU prefill: {harp_meta['saved_sec']:.2f}s", flush=True)
                    except Exception as _harp_err:
                        print(f"[Gateway] [HARP-Cache] Error in prefix tracking: {_harp_err}", flush=True)

                print(f"[Gateway] [{log_sess}] Forwarding to {target_upstream}: model={data['model']} mode={active_mode.upper()} budget={data.get('reasoning_budget')} stream={is_stream}", flush=True)
                body_bytes = json.dumps(data).encode("utf-8")
            except Exception as e:
                print(f"[Gateway] Error: {e}", flush=True)
                target_upstream = UPSTREAM_BASE
                harp_meta = None

        if clean_post_path not in ("/v1/messages", "/v1/messages/count_tokens"):
            _last_endpoint_was_chat = True
            if clean_post_path == "/v1/chat/completions" and body_bytes:
                try:
                    cdata = json.loads(body_bytes.decode("utf-8"))
                    if isinstance(cdata, dict):
                        cmsgs = cdata.get("messages")
                        if isinstance(cmsgs, list) and cmsgs:
                            def _extract_msg_text(c):
                                if isinstance(c, str):
                                    return c
                                if isinstance(c, list):
                                    return "\n".join(
                                        p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text"
                                    )
                                return "" if c is None else str(c)

                            # Merge leading system/developer messages into a single index-0 system message
                            # so strict Jinja chat templates never fail on multiple or developer-role headers.
                            lead_sys, rest_msgs, seen_non_sys = [], [], False
                            for m in cmsgs:
                                if isinstance(m, dict) and m.get("role") in ("system", "developer") and not seen_non_sys:
                                    lead_sys.append(_extract_msg_text(m.get("content")))
                                    continue
                                if isinstance(m, dict) and m.get("role") == "developer":
                                    m = {**m, "role": "system"}
                                seen_non_sys = True
                                rest_msgs.append(m)
                            sys_text = "\n\n".join(t for t in lead_sys if t) or "You are a helpful, accurate, and concise AI assistant."
                            cdata["messages"] = [{"role": "system", "content": sys_text}] + rest_msgs
                        mct = cdata.pop("max_completion_tokens", None)
                        if mct is not None:
                            cdata.setdefault("max_tokens", mct)
                        cdata.pop("store", None)
                        # Omit repeat_penalty when non-empty tools list is present (empty tools=[] still gets 1.05)
                        if not cdata.get("tools"):
                            cdata.setdefault("repeat_penalty", 1.05)
                            cdata.setdefault("repeat_last_n", 64)
                        body_bytes = json.dumps(cdata).encode("utf-8")
                except Exception:
                    pass

        self._forward_request("POST", body_bytes, active_mode=active_mode, target_upstream=target_upstream, harp_meta=harp_meta, tool_schemas=tool_schemas)

    def _forward_request(self, method, body_bytes=None, active_mode="med", target_upstream=UPSTREAM_BASE, harp_meta=None, tool_schemas=None):
        clean_path = self.path.split("?", 1)[0]
        allowed_set = ALLOWED_POST_PATHS if method == "POST" else ALLOWED_GET_PATHS
        if clean_path not in allowed_set:
            self._send_json_error(403, f"Path '{clean_path}' is not in the gateway allow-list")
            return
        url = target_upstream + self.path
        req_headers = {}
        drop_headers = {
            "host",
            "content-length",
            "connection",
            "keep-alive",
            "transfer-encoding",
            "te",
            "trailer",
            "upgrade",
            "proxy-authorization",
            "proxy-authenticate",
            "accept-encoding",
        }
        if GATEWAY_API_KEY:
            drop_headers.update({"authorization", "x-api-key"})
        for k, v in self.headers.items():
            if k.lower() not in drop_headers:
                req_headers[k] = v
        req_headers["Accept-Encoding"] = "identity"

        req = urllib.request.Request(url, data=body_bytes if method == "POST" else None, headers=req_headers, method=method)
        headers_sent = False

        try:
            with urllib.request.urlopen(req, timeout=1800) as upstream:
                self.send_response(upstream.status)
                is_event_stream = "event-stream" in upstream.headers.get("Content-Type", "")
                for k, v in upstream.getheaders():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                if harp_meta and harp_meta.get("hit"):
                    self.send_header("X-HARP-Cache", "HIT")
                    self.send_header("X-HARP-Cached-Tokens", str(harp_meta.get("cached_tokens", 0)))
                    self.send_header("X-HARP-Prefill-Saved-Sec", f"{harp_meta.get('saved_sec', 0.0):.2f}")
                self.send_cors_headers()
                self.end_headers()
                headers_sent = True

                if is_event_stream and clean_path == "/v1/messages":
                    stream_filter = ThinkBlockFilter(mode=active_mode, schemas=tool_schemas)
                    utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                    last_event_time = time.time()
                    while True:
                        chunk = upstream.read1(4096) if hasattr(upstream, "read1") else upstream.read(256)
                        if not chunk:
                            break
                        decoded = utf8_decoder.decode(chunk, final=False)
                        filtered = stream_filter.feed(decoded)
                        if filtered:
                            try:
                                self.wfile.write(filtered.encode("utf-8"))
                                self.wfile.flush()
                                last_event_time = time.time()
                            except BrokenPipeError:
                                return
                        else:
                            now = time.time()
                            if now - last_event_time >= 10.0:
                                try:
                                    self.wfile.write(b"event: ping\ndata: {\"type\": \"ping\"}\n\n")
                                    self.wfile.flush()
                                    last_event_time = now
                                except BrokenPipeError:
                                    return
                    rem_utf8 = utf8_decoder.decode(b"", final=True)
                    tail = (stream_filter.feed(rem_utf8) if rem_utf8 else "") + stream_filter.flush()
                    if tail:
                        try:
                            self.wfile.write(tail.encode("utf-8"))
                            self.wfile.flush()
                        except BrokenPipeError:
                            pass
                elif is_event_stream:
                    while True:
                        chunk = upstream.read1(4096) if hasattr(upstream, "read1") else upstream.read(256)
                        if not chunk:
                            break
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except BrokenPipeError:
                            return
                else:
                    full_body = upstream.read()
                    if clean_path == "/v1/messages":
                        try:
                            resp_json = json.loads(full_body.decode("utf-8"))
                            resp_json = sanitize_non_streaming_response(resp_json, tool_schemas)
                            full_body = json.dumps(resp_json).encode("utf-8")
                        except Exception:
                            pass
                    try:
                        self.wfile.write(full_body)
                        self.wfile.flush()
                    except BrokenPipeError:
                        pass

        except urllib.error.HTTPError as e:
            err_body = e.read()
            if not headers_sent:
                self.send_response(e.code)
                for k, v in e.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                        self.send_header(k, v)
                self.send_cors_headers()
                self.end_headers()
                try:
                    self.wfile.write(err_body)
                except BrokenPipeError:
                    pass
        except Exception as e:
            print(f"[Gateway] Upstream error: {e}", flush=True)
            if not headers_sent:
                try:
                    self.send_error(502, f"Upstream error: {e}")
                except Exception:
                    pass
            else:
                err_ev = f'event: error\ndata: {json.dumps({"type": "error", "error": {"type": "api_error", "message": str(e)}})}\n\n'
                try:
                    self.wfile.write(err_ev.encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("GATEWAY_PORT", 8787))
    server = ThreadingHTTPServer((GATEWAY_HOST, port), GatewayHandler)
    print(f"[Gateway] Multi-threaded Listening on {GATEWAY_HOST}:{port} (Session-isolated, default: {_default_think_mode})", flush=True)
    server.serve_forever()
