#!/usr/bin/env python3
"""
Mandatory Release Preflight Gate for synchopate/llama-modchip.
Blocks tagging/publishing if docs, paper/, ABI enums, rc_*.h headers, SHA-256s,
benchmark tables, platform launchers, tests, or OPSEC checks drift.

Usage:
  python3 scripts/preflight_release_check.py --strict [--run-cpp-tests] [--verify-remote] [--allow-dirty]
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "release_manifest.json"

# Files that legitimately document historical v2.x releases (exempt from active-doc legacy-ABI/deleted-file checks)
HISTORICAL_ALLOWLIST = {
    "RELEASE_NOTES.md",
    "benchmarks/HYBRID_SSM_VS_DENSE_GQA_BENCHMARK.md",
    "benchmarks/KV_SCALING_BENCHMARK_REPORT.md",
}

DELETED_FILES_BLACKLIST = [
    "scripts/transcode_dsv4.c",
    "scripts/transcode_dsv4_shard.py",
    "scripts/build_fractal_moe_gguf.sh",
    "scripts/build_nemotron_vq3.py",
    "patches/zen5-nemotron-256k-optimizations.patch",
    "patches/deepseek-v4-flash-fractal-256k.patch",
    "patches/lut-vq3-64-avx512-vbmi.patch",
    "patches/vq3_codebook.h",
    "patches/apple-silicon/",
    "/tmp/dsv4_moe_cfg.bin",
    "%TEMP%\\dsv4_moe_cfg.bin",
]

# Public-safe structural OPSEC checks (catches private IPs, personal home dirs, storage/mnt paths, and API tokens)
OPSEC_FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\b192\.168\.\d{1,3}\.\d{1,3}\b", "Private LAN IPv4 address (192.168.x.x)"),
    (r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "Private RFC1918 IPv4 address (10.x.x.x)"),
    (r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b", "Private RFC1918 IPv4 address (172.16-31.x.x)"),
    (r"/(?:home|Users)/(?!user\b|youruser\b|ubuntu\b|runner\b|llama\b|dev\b|alice\b|bob\b)[a-zA-Z0-9._-]+/", "Personal user home directory path"),
    (r"/storage\d+tb\b", "Private local storage mount path"),
    (r"/mnt/[a-zA-Z0-9._-]+/", "Private /mnt filesystem path"),
    (r"\bhf_[A-Za-z0-9]{20,}\b", "HuggingFace API token"),
    (r"\bghp_[A-Za-z0-9]{20,}\b", "GitHub personal access token"),
    (r"\bgithub_pat_[A-Za-z0-9_]{20,}\b", "GitHub fine-grained token"),
    (r"\bsk-[A-Za-z0-9]{20,}\b", "API secret key"),
    (r"(?i)\bproxmox\b", "Hypervisor vendor name (Proxmox)"),
    (r"/usr/share/ollama\b", "Host-specific Ollama storage path"),
]


class GateResult:
    def __init__(self, name: str) -> None:
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def run_cmd(cmd: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)


def git_tracked_files() -> list[Path]:
    res = run_cmd(["git", "ls-files", "-z"])
    if res.returncode != 0:
        raise RuntimeError(f"git ls-files failed: {res.stderr}")
    return [REPO_ROOT / p for p in res.stdout.split("\0") if p and (REPO_ROOT / p).exists()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Create release_manifest.json (see docs/RELEASE_PROCEDURE.md)."
        )
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Gate 1: Version & Upstream Base Commit/Build Number Parity
# ---------------------------------------------------------------------------
def check_version_and_build_parity(manifest: dict[str, Any], g: GateResult) -> None:
    tag = manifest["release_tag"]
    ver = manifest["release_version"]
    commit = manifest["upstream_llama_cpp"]["commit"]
    build_num = str(manifest["upstream_llama_cpp"]["build_number"])

    version_targets = [
        ("README.md", tag),
        ("RELEASE_NOTES.md", tag),
        ("ARCHITECTURE.md", tag),
        ("install.sh", tag),
        ("modchip", tag),
    ]
    for rel, needle in version_targets:
        p = REPO_ROOT / rel
        if not p.exists():
            g.fail(f"Missing required file: {rel}")
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        if needle not in txt and ver not in txt:
            g.fail(f"{rel}: does not reference current release {tag}/{ver}")

    build_scripts = [
        "scripts/build_zen5_llama.sh",
        "fedora/nemotron.sh",
        "apple-silicon/build-dsv4-engine.sh",
        "windows/build_patched_engine.ps1",
    ]
    for rel in build_scripts:
        p = REPO_ROOT / rel
        if not p.exists():
            g.fail(f"Missing required build script: {rel}")
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        if commit[:9] not in txt:
            g.fail(f"{rel}: pinned llama.cpp commit {commit} not found")
        has_build_num = (
            f"LLAMA_BUILD_NUMBER={build_num}" in txt
            or (f'LLAMA_BUILD_NUM="{build_num}"' in txt and 'LLAMA_BUILD_NUMBER="${LLAMA_BUILD_NUM}"' in txt)
        )
        if not has_build_num:
            g.fail(f"{rel}: missing -DLLAMA_BUILD_NUMBER={build_num}")

    # Check top-level patches/ ambiguity
    top_patches = sorted((REPO_ROOT / "patches").glob("llama-modchip-v*.patch"))
    expected_patch = REPO_ROOT / manifest["patch_file"]
    if len(top_patches) > 1:
        g.warn(
            f"Multiple top-level patches in patches/: {[p.name for p in top_patches]} "
            "(move older patches to patches/archive/)"
        )
    if not expected_patch.exists():
        g.fail(f"Active patch missing: {manifest['patch_file']}")
    else:
        actual_sha = sha256_file(expected_patch)
        if actual_sha != manifest["patch_sha256"]:
            g.fail(
                f"{manifest['patch_file']} SHA-256 mismatch: "
                f"actual={actual_sha}, manifest={manifest['patch_sha256']}"
            )
        for doc in ("README.md", "RELEASE_NOTES.md", "modchip"):
            txt = (REPO_ROOT / doc).read_text(encoding="utf-8")
            if actual_sha not in txt:
                g.fail(f"{doc}: missing active patch SHA-256 {actual_sha[:12]}...")


# ---------------------------------------------------------------------------
# Gate 2: ABI Enum Extraction from Active Patch vs gguf_tensor_type.py & Docs
# ---------------------------------------------------------------------------
def check_abi_enums(manifest: dict[str, Any], g: GateResult) -> None:
    patch_path = REPO_ROOT / manifest["patch_file"]
    if not patch_path.exists():
        g.fail(f"Cannot verify ABI enums; missing {patch_path}")
        return
    patch_txt = patch_path.read_text(encoding="utf-8", errors="replace")

    for num_str, spec in manifest["abi_enums"].items():
        c_name = spec["c_name"]
        pat = rf"\b{re.escape(c_name)}\s*=\s*{re.escape(num_str)}\b"
        if not re.search(pat, patch_txt):
            g.fail(f"{manifest['patch_file']}: missing expected enum '{c_name} = {num_str}'")

    py_tool = REPO_ROOT / "scripts/gguf_tensor_type.py"
    if not py_tool.exists():
        g.fail("Missing scripts/gguf_tensor_type.py")
        return
    py_txt = py_tool.read_text(encoding="utf-8")
    for num_str, spec in manifest["abi_enums"].items():
        py_name = spec["py_name"]
        m = re.search(rf"^\s*{num_str}\s*:\s*['\"]([^'\"]+)['\"]", py_txt, re.MULTILINE)
        if not m:
            g.fail(f"scripts/gguf_tensor_type.py: enum ID {num_str} missing")
        elif m.group(1) != py_name:
            g.fail(
                f"scripts/gguf_tensor_type.py: enum ID {num_str} mapped to "
                f"'{m.group(1)}', expected '{py_name}' ({spec['c_name']})"
            )

    banned_abi_patterns = [
        (r"GGML_TYPE_VQ3_0\s*=\s*43", "Retired enum GGML_TYPE_VQ3_0 = 43 (now GGML_TYPE_VQ3_64 = 45)"),
        (r"hijacks?\s+`?GGML_TYPE_Q2_0", "Outdated claim that Fractal-64 hijacks GGML_TYPE_Q2_0"),
        (r"Custom\s+Q2_0\s*\(type\s*42\)", "Outdated claim of custom Q2_0 type 42"),
    ]
    for p in git_tracked_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel in HISTORICAL_ALLOWLIST or not rel.endswith((".md", ".tex")):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        for pat, desc in banned_abi_patterns:
            for m in re.finditer(pat, txt, re.IGNORECASE):
                line_no = txt.count("\n", 0, m.start()) + 1
                g.fail(f"{rel}:{line_no}: {desc}")


# ---------------------------------------------------------------------------
# Gate 3: speculative/cpp/rc_*.h Byte-Identity Inside Active Patch
# ---------------------------------------------------------------------------
def extract_new_file_from_patch(patch_txt: str, target_rel: str) -> str | None:
    header = f"+++ b/{target_rel}\n"
    idx = patch_txt.find(header)
    if idx == -1:
        return None
    rest = patch_txt[idx + len(header) :]
    out: list[str] = []
    in_hunk = False
    for line in rest.splitlines(keepends=True):
        if line.startswith("diff --git "):
            break
        if line.startswith("@@ "):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+"):
            out.append(line[1:])
        elif line.startswith(" "):
            out.append(line[1:])
    return "".join(out) if out else None


def check_rc_header_sync(manifest: dict[str, Any], g: GateResult) -> None:
    patch_path = REPO_ROOT / manifest["patch_file"]
    if not patch_path.exists():
        return
    patch_txt = patch_path.read_text(encoding="utf-8", errors="replace")
    for hdr in manifest.get("rc_synced_headers", []):
        if "/" in hdr:
            patch_rel = hdr
            local_name = Path(hdr).name
        else:
            patch_rel = f"common/{hdr}"
            local_name = hdr
        local_p = REPO_ROOT / "speculative/cpp" / local_name
        if not local_p.exists():
            g.fail(f"Missing header speculative/cpp/{local_name}")
            continue
        embedded = extract_new_file_from_patch(patch_txt, patch_rel)
        if embedded is None:
            g.fail(f"{manifest['patch_file']}: {patch_rel} not found as added file in patch")
            continue
        local_txt = local_p.read_text(encoding="utf-8")
        if local_txt != embedded:
            g.fail(f"Header drift: speculative/cpp/{local_name} != {patch_rel} inside {manifest['patch_file']}")


# ---------------------------------------------------------------------------
# Gate 4: SHA-256 Parity (install.sh, modchip download-bin & checksums, docs) & Pinned HF MODEL_REV
# ---------------------------------------------------------------------------
def _extract_bash_fn(script_txt: str, fn_name: str) -> str:
    m = re.search(rf"^{re.escape(fn_name)}\(\)\s*\{{", script_txt, re.MULTILINE)
    if not m:
        return ""
    rest = script_txt[m.start() :]
    end_m = re.search(r"^\}\s*$", rest, re.MULTILINE)
    return rest[: end_m.end()] if end_m else rest


def check_hashes_and_pinned_revs(manifest: dict[str, Any], g: GateResult) -> None:
    install_sh = (REPO_ROOT / "install.sh").read_text(encoding="utf-8")
    modchip_sh = (REPO_ROOT / "modchip").read_text(encoding="utf-8")
    rel_notes = (REPO_ROOT / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    readme_md = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    # 1. Every MODEL_REV= in install.sh must be a pinned 40-hex commit SHA
    for m in re.finditer(r'MODEL_REV="([^"]+)"', install_sh):
        rev = m.group(1)
        line_no = install_sh.count("\n", 0, m.start()) + 1
        if not re.fullmatch(r"[0-9a-f]{40}", rev):
            g.fail(f'install.sh:{line_no}: floating/unpinned MODEL_REV="{rev}" (must be 40-hex HF SHA)')

    # 2. install.sh must NOT gate existing-file SHA-256 verification only on MODEL_FAMILY == nemotron
    if re.search(r'if\s+\[\[\s*"\$MODEL_FAMILY"\s*==\s*"nemotron"\s*\]\]', install_sh):
        g.fail("install.sh: SHA-256 verification for existing files is gated only on MODEL_FAMILY == nemotron")

    artifacts = manifest.get("artifacts", {})
    gh_only = manifest.get("github_only_artifacts", {})

    # 3. Explicitly verify install.sh EXPECTED_BIN_SHA256 matches avx2 & avx512 server binaries
    install_bin_shas = set(re.findall(r'EXPECTED_BIN_SHA256="([0-9a-f]{64})"', install_sh))
    for srv_name in ("llama-server-modchip-linux-x86_64-avx2", "llama-server-modchip-linux-x86_64-avx512"):
        exp = artifacts.get(srv_name)
        if exp and exp not in install_bin_shas:
            g.fail(f"install.sh: EXPECTED_BIN_SHA256 missing/mismatched for {srv_name} (expected {exp[:12]}...)")

    # 4. Explicitly verify modchip cmd_download_bin() hashes (NOT just cmd_checksums!)
    dl_fn = _extract_bash_fn(modchip_sh, "cmd_download_bin")
    cs_fn = _extract_bash_fn(modchip_sh, "cmd_checksums")
    if not dl_fn:
        g.fail("modchip: could not locate cmd_download_bin() function")
    else:
        dl_shas = set(re.findall(r'(?:srv|cli|bnc|ppl|qnt)_sha256="([0-9a-f]{64})"', dl_fn))
        for tool in ("llama-server", "llama-cli", "llama-bench", "llama-perplexity", "llama-quantize"):
            for isa in ("avx2", "avx512"):
                art = f"{tool}-modchip-linux-x86_64-{isa}"
                exp = artifacts.get(art)
                if exp and exp not in dl_shas:
                    g.fail(f"modchip cmd_download_bin(): missing/mismatched SHA-256 for {art} (expected {exp[:12]}...)")

    if not cs_fn:
        g.fail("modchip: could not locate cmd_checksums() function")

    # 5. Manifest artifact hashes must appear consistently in RELEASE_NOTES.md, README.md, and modchip cmd_checksums()
    all_artifacts = {}
    all_artifacts.update(artifacts)
    all_artifacts.update(gh_only)
    for artifact, expected_sha in all_artifacts.items():
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            g.fail(f"release_manifest.json: invalid SHA-256 for {artifact}: '{expected_sha}'")
            continue
        if expected_sha not in rel_notes:
            g.fail(f"RELEASE_NOTES.md: missing SHA-256 {expected_sha[:12]}... for {artifact}")
        if expected_sha not in readme_md:
            g.fail(f"README.md: missing SHA-256 {expected_sha[:12]}... for {artifact}")
        if cs_fn and expected_sha not in cs_fn:
            g.fail(f"modchip cmd_checksums(): missing SHA-256 {expected_sha[:12]}... for {artifact}")


# ---------------------------------------------------------------------------
# Gate 5: Benchmark JSON <-> Markdown Reconciliation (Aggregate + W1..W4 + Accept % + Mean Len)
# ---------------------------------------------------------------------------
def nested_get(data: dict[str, Any], dotted_key: str) -> Any:
    cur: Any = data
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def check_benchmarks_vs_docs(manifest: dict[str, Any], g: GateResult) -> None:
    rc2_wkeys = (
        "w1_claude_refactor",
        "w2_claude_multitool_json",
        "w3_greenfield_c99",
        "w4_tech_prose_it_en",
    )
    for entry in manifest.get("benchmarks", []):
        jp = REPO_ROOT / entry["json"]
        if not jp.exists():
            g.fail(f"Missing benchmark JSON: {entry['json']}")
            continue
        raw = json.loads(jp.read_text(encoding="utf-8"))
        if not raw.get("model_label"):
            g.fail(f"{entry['json']}: 'model_label' is null/empty")
        mpath = str(raw.get("model_path") or "")
        if mpath.startswith("/") or "\\" in mpath:
            g.fail(f"{entry['json']}: 'model_path' contains raw filesystem path '{mpath}' (sanitize to filename)")

        doc_texts = {doc_rel: (REPO_ROOT / doc_rel).read_text(encoding="utf-8") for doc_rel in entry.get("docs", [])}

        # 1. Check explicit manifest key_metrics
        for key, expected_val in entry.get("key_metrics", {}).items():
            actual = nested_get(raw, key)
            if actual is None:
                g.fail(f"{entry['json']}: missing key '{key}'")
                continue
            if abs(float(actual) - float(expected_val)) > 0.015:
                g.fail(
                    f"{entry['json']}: key '{key}'={actual} differs from manifest {expected_val}"
                )
            needle = f"{float(expected_val):.2f}"
            for doc_rel, doc_txt in doc_texts.items():
                if needle not in doc_txt:
                    g.fail(f"{doc_rel}: does not report {key}={needle} from {entry['json']}")

        # 2. Check RouteCrack 2.0 per-workload W1..W4 tok/s, exact accept %, and mean_verified_len
        rc2_cfg = nested_get(raw, "configs.config3_routecrack2")
        if isinstance(rc2_cfg, dict):
            derived: dict[str, str] = {}
            wls = rc2_cfg.get("workloads") or {}
            for wk in rc2_wkeys:
                if wk in wls and "tok_per_s" in wls[wk]:
                    derived[wk] = f"{float(wls[wk]['tok_per_s']):.2f}"
            tot_n = int(rc2_cfg.get("total_draft_n") or 0)
            tot_acc = int(rc2_cfg.get("total_draft_accepted") or 0)
            if tot_n > 0:
                derived["accept_pct"] = f"{(100.0 * tot_acc / tot_n):.2f}"
            elif "aggregate_accept_rate" in rc2_cfg:
                derived["accept_pct"] = f"{float(rc2_cfg['aggregate_accept_rate']) * 100.0:.2f}"
            if "mean_verified_len" in rc2_cfg:
                derived["mean_verified_len"] = f"{float(rc2_cfg['mean_verified_len']):.2f}"

            for metric_name, needle in derived.items():
                for doc_rel, doc_txt in doc_texts.items():
                    if needle not in doc_txt:
                        g.fail(
                            f"{doc_rel}: missing RouteCrack 2.0 metric {metric_name}={needle} from {entry['json']}"
                        )


# ---------------------------------------------------------------------------
# Gate 6: Technical Paper (`paper/`) Freshness, Tarball Parity (tex + figures) & Layout Check
# ---------------------------------------------------------------------------
def check_paper(manifest: dict[str, Any], g: GateResult) -> None:
    pcfg = manifest.get("paper", {})
    tex_p = REPO_ROOT / pcfg.get("tex", "paper/main.tex")
    pdf_p = REPO_ROOT / pcfg.get("pdf", "paper/llama-modchip-paper.pdf")
    tar_p = REPO_ROOT / pcfg.get("tarball", "paper/arxiv-submission.tar.gz")
    req_ver = pcfg.get("required_version_string", manifest["release_tag"])

    for p in (tex_p, pdf_p, tar_p):
        if not p.exists():
            g.fail(f"Paper artifact missing: {p.relative_to(REPO_ROOT)}")
    if not tex_p.exists():
        return

    tex_txt = tex_p.read_text(encoding="utf-8")
    tex_norm = tex_txt.replace(r"\_", "_")
    if req_ver not in tex_norm:
        g.fail(f"{tex_p.name}: missing required version '{req_ver}' (paper is stale)")
    for sec in pcfg.get("required_sections", []):
        if sec not in tex_norm:
            g.fail(f"{tex_p.name}: missing required topic '{sec}'")

    # Ensure every \bibitem is cited at least once in the paper body
    bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", tex_txt)
    cited_keys: set[str] = set()
    for m in re.finditer(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", tex_txt):
        for k in m.group(1).split(","):
            cited_keys.add(k.strip())
    for bk in bib_keys:
        if bk not in cited_keys:
            g.fail(f"{tex_p.name}: uncited bibliography entry '\\bibitem{{{bk}}}'")

    fig_matches = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex_txt)
    if len(fig_matches) < int(pcfg.get("min_figures", 1)):
        g.fail(f"{tex_p.name}: has {len(fig_matches)} figures (minimum required: {pcfg.get('min_figures', 1)})")
    for fig_rel in fig_matches:
        fig_p = tex_p.parent / fig_rel
        if not fig_p.exists():
            g.fail(f"{tex_p.name}: referenced figure missing on disk: {fig_rel}")

    # Verify arxiv-submission.tar.gz contains byte-identical main.tex AND byte-identical figures/*.pdf
    if tar_p.exists():
        with tarfile.open(tar_p, "r:gz") as tf:
            names = tf.getnames()
            tex_member = next((n for n in names if n.endswith("main.tex")), None)
            if not tex_member:
                g.fail(f"{tar_p.name}: does not contain main.tex")
            else:
                ef = tf.extractfile(tex_member)
                tar_tex = ef.read().decode("utf-8", errors="replace") if ef else ""
                if tar_tex != tex_txt:
                    g.fail(f"{tar_p.name}: {tex_member} differs from paper/main.tex (rebuild tarball)")
            for fig_rel in fig_matches:
                fig_member = next((n for n in names if n.endswith(fig_rel)), None)
                if not fig_member:
                    g.fail(f"{tar_p.name}: missing referenced figure '{fig_rel}'")
                else:
                    fig_p = tex_p.parent / fig_rel
                    if fig_p.exists():
                        ef = tf.extractfile(fig_member)
                        tar_sha = hashlib.sha256(ef.read()).hexdigest() if ef else ""
                        disk_sha = sha256_file(fig_p)
                        if tar_sha != disk_sha:
                            g.fail(
                                f"{tar_p.name}: figure '{fig_rel}' SHA-256 mismatch vs paper/{fig_rel} (rebuild tarball)"
                            )

    # Verify compiled PDF freshness, version, unresolved citations, CLI hyphen rendering, and float placement
    if pdf_p.exists() and shutil.which("pdftotext"):
        res = run_cmd(["pdftotext", str(pdf_p), "-"])
        if res.returncode != 0:
            g.fail(f"pdftotext failed on {pdf_p.name}: {res.stderr}")
        else:
            pdf_txt = res.stdout
            if req_ver not in pdf_txt:
                g.fail(f"{pdf_p.name}: does not contain '{req_ver}' (recompile PDF from main.tex)")
            if "[?]" in pdf_txt or "??" in pdf_txt:
                g.fail(f"{pdf_p.name}: contains unresolved citation/reference marker ('[?]' or '??')")
            if "\u2013repack" in pdf_txt or "\u2013spec-type" in pdf_txt or "\u2013poll" in pdf_txt:
                g.fail(f"{pdf_p.name}: CLI double-hyphen flags collapsed into en-dash (escape double hyphens inside \\texttt)")
            ref_matches = list(re.finditer(r"(?:^|\n|\f)References\s*\n", pdf_txt))
            if not ref_matches:
                g.fail(f"{pdf_p.name}: could not locate References section in extracted text")
            else:
                ref_idx = ref_matches[-1].start()
                if re.search(r"\bTable\s+\d+:", pdf_txt[ref_idx:]):
                    g.fail(f"{pdf_p.name}: a Table floats AFTER References (add \\clearpage before bibliography)")
    elif pdf_p.exists():
        g.warn("pdftotext not installed; skipping PDF text & float placement inspection")


# ---------------------------------------------------------------------------
# Gate 7: Broken Relative Links, Dead File References & Issue Template Hygiene
# ---------------------------------------------------------------------------
def check_links_and_dead_refs(_manifest: dict[str, Any], g: GateResult) -> None:
    link_re = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for p in git_tracked_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if not rel.endswith((".md", ".tex")):
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")

        if rel not in HISTORICAL_ALLOWLIST and rel != "docs/RELEASE_PROCEDURE.md":
            for dead in DELETED_FILES_BLACKLIST:
                if dead in txt:
                    g.fail(f"{rel}: references deleted/removed artifact '{dead}'")

        if rel.endswith(".md"):
            stripped = re.sub(r"```[\s\S]*?```", lambda m: "\n" * m.group(0).count("\n"), txt)
            for m in link_re.finditer(stripped):
                target = m.group(1).strip().split()[0]
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_part = target.split("#", 1)[0]
                if not path_part:
                    continue
                resolved = (p.parent / path_part).resolve()
                if not resolved.exists():
                    line_no = stripped.count("\n", 0, m.start()) + 1
                    g.fail(f"{rel}:{line_no}: broken relative link '{target}'")

    tpl_dir = REPO_ROOT / ".github/ISSUE_TEMPLATE"
    if tpl_dir.exists():
        bench_tpls = [p.name for p in tpl_dir.iterdir() if "bench" in p.name.lower()]
        if len(bench_tpls) > 1:
            g.fail(f".github/ISSUE_TEMPLATE/: duplicate benchmark templates found: {bench_tpls}")


# ---------------------------------------------------------------------------
# Gate 8: ISA Safety, Threading, THP & IPC Security Invariants
# ---------------------------------------------------------------------------
def check_runtime_invariants(_manifest: dict[str, Any], g: GateResult) -> None:
    rc_py = (REPO_ROOT / "scripts/check_cpu_readiness.py").read_text(encoding="utf-8")
    if re.search(r"has_avx512f\s+and\s+has_avx512_vbmi(?!\s+and\s+has_avx512_vnni)", rc_py):
        g.fail("scripts/check_cpu_readiness.py: avx512 binary check missing 'and has_avx512_vnni' (SIGILL risk)")

    for rel in ("scripts/check_cpu_readiness.py", "fedora/nemotron.sh", "install.sh"):
        p = REPO_ROOT / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            if re.search(r"echo\s+always\s*(?:\|\s*sudo\s+tee|>)\s*/sys/kernel/mm/transparent_hugepage/enabled", txt):
                g.fail(f"{rel}: sets THP=always instead of madvise")

    for rel in ("gateway/sse_gateway.py", "kv_cache/harp_cache.py"):
        p = REPO_ROOT / rel
        if p.exists():
            txt = p.read_text(encoding="utf-8")
            if re.search(r'(?<!f)"/dev/shm/[^"{}]+\.bin"', txt):
                g.fail(f"{rel}: uses hardcoded global /dev/shm/*.bin path (must include per-UID suffix + 0600 O_NOFOLLOW)")


# ---------------------------------------------------------------------------
# Gate 9: Syntax, Unit Tests & Multi-ISA C++ ASan/UBSan RouteCrack 2.0 Suite
# ---------------------------------------------------------------------------
def check_syntax_and_tests(run_cpp: bool, g: GateResult) -> None:
    shell_files = [
        p for p in git_tracked_files()
        if p.suffix == ".sh" or p.name in {"install.sh", "modchip"}
    ]
    for sh in shell_files:
        res = run_cmd(["bash", "-n", str(sh)])
        if res.returncode != 0:
            g.fail(f"bash -n failed for {sh.relative_to(REPO_ROOT)}: {res.stderr.strip()}")

    for py in git_tracked_files():
        if py.suffix != ".py":
            continue
        try:
            ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except SyntaxError as exc:
            g.fail(f"Python syntax error in {py.relative_to(REPO_ROOT)}: {exc}")

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for suite in ("gateway", "kv_cache"):
        res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", suite, "-p", "test_*.py", "-v"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            env=env,
        )
        if res.returncode != 0:
            g.fail(f"unittest suite '{suite}' failed (rc={res.returncode}):\n{res.stderr.strip()}")
        m_ran = re.search(r"Ran\s+(\d+)\s+tests?", res.stderr)
        if not m_ran or int(m_ran.group(1)) == 0:
            g.fail(f"unittest suite '{suite}' discovered 0 tests")

    if run_cpp:
        cpp_test = REPO_ROOT / "speculative/cpp/test_routecrack2.cpp"
        if not cpp_test.exists():
            g.fail("Missing speculative/cpp/test_routecrack2.cpp")
        elif not shutil.which("g++"):
            g.fail("g++ not available for --run-cpp-tests")
        else:
            isa_profiles: list[tuple[str, list[str]]] = [
                ("scalar", ["-mno-avx512f", "-mno-avx2"]),
                ("avx2-fma", ["-mavx2", "-mfma", "-mno-avx512f"]),
                ("native", ["-march=native"]),
            ]
            if os.uname().machine not in ("x86_64", "AMD64"):
                isa_profiles = [("native", ["-march=native"])]
            with tempfile.TemporaryDirectory() as td:
                for label, extra_flags in isa_profiles:
                    out_bin = Path(td) / f"test_routecrack2_{label}"
                    comp = run_cmd([
                        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        *extra_flags,
                        "-fsanitize=address,undefined",
                        "-I", str(REPO_ROOT / "speculative/cpp"),
                        str(cpp_test), "-o", str(out_bin),
                    ])
                    if comp.returncode != 0:
                        g.fail(f"C++ compile failed ({label}):\n{comp.stderr.strip()}")
                        continue
                    run = run_cmd([str(out_bin)])
                    if run.returncode != 0:
                        g.fail(f"test_routecrack2 ({label}) failed (rc={run.returncode}):\n{run.stdout}\n{run.stderr}")
                harp_test = REPO_ROOT / "speculative/cpp/test_harp_radix.cpp"
                if harp_test.exists():
                    harp_bin = Path(td) / "test_harp_radix"
                    comp_h = run_cmd([
                        "g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fsanitize=address,undefined",
                        "-I", str(REPO_ROOT / "speculative/cpp"),
                        str(harp_test), "-o", str(harp_bin),
                    ])
                    if comp_h.returncode != 0:
                        g.fail(f"C++ compile failed (test_harp_radix):\n{comp_h.stderr.strip()}")
                    else:
                        run_h = run_cmd([str(harp_bin)])
                        if run_h.returncode != 0:
                            g.fail(f"test_harp_radix failed (rc={run_h.returncode}):\n{run_h.stdout}\n{run_h.stderr}")


# ---------------------------------------------------------------------------
# Gate 10: OPSEC & Proprietary Separation Scan (Tracked Files + Paper PDF + Figure PDFs)
# ---------------------------------------------------------------------------
def check_opsec(manifest: dict[str, Any], g: GateResult) -> None:
    self_rel = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    patterns = list(OPSEC_FORBIDDEN_PATTERNS)
    extra_env = os.environ.get("MODCHIP_OPSEC_EXTRA_PATTERNS", "").strip()
    if extra_env:
        for tok in extra_env.split(","):
            tok = tok.strip()
            if tok:
                patterns.append((re.escape(tok), f"Custom forbidden token '{tok}'"))
    compiled = [(re.compile(pat, re.IGNORECASE), desc) for pat, desc in patterns]

    for p in git_tracked_files():
        rel = p.relative_to(REPO_ROOT).as_posix()
        if rel == self_rel or p.suffix.lower() in {".png", ".gif", ".pdf", ".gz", ".zip", ".bin"}:
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for cre, desc in compiled:
            for m in cre.finditer(txt):
                line_no = txt.count("\n", 0, m.start()) + 1
                g.fail(f"OPSEC leak in {rel}:{line_no}: {desc} ('{m.group(0)}')")

    if shutil.which("pdftotext"):
        pdf_targets: list[Path] = []
        main_pdf = REPO_ROOT / manifest.get("paper", {}).get("pdf", "paper/llama-modchip-paper.pdf")
        if main_pdf.exists():
            pdf_targets.append(main_pdf)
        fig_dir = REPO_ROOT / "paper/figures"
        if fig_dir.exists():
            pdf_targets.extend(sorted(fig_dir.glob("*.pdf")))
        for pdf_p in pdf_targets:
            res = run_cmd(["pdftotext", str(pdf_p), "-"])
            if res.returncode == 0:
                for cre, desc in compiled:
                    m = cre.search(res.stdout)
                    if m:
                        g.fail(f"OPSEC leak in PDF {pdf_p.relative_to(REPO_ROOT)}: {desc} ('{m.group(0)}')")


# ---------------------------------------------------------------------------
# Gate 11: Remote Dual-Mirror Verification (GitHub Release + HF Mirror + SHA256SUMS + Remote Patch)
# ---------------------------------------------------------------------------
def _extract_hex64(val: str | None) -> str | None:
    if not val:
        return None
    m = re.search(r"\b([0-9a-f]{64})\b", val.lower())
    return m.group(1) if m else None


class _CaptureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Captures pre-redirect headers (e.g. Hugging Face X-Linked-ETag LFS SHA-256) before following to CDN."""

    def __init__(self) -> None:
        super().__init__()
        self.redirect_headers: list[dict[str, str]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.redirect_headers.append({k.lower(): v for k, v in headers.items()})
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_remote_mirrors(manifest: dict[str, Any], g: GateResult) -> None:
    gh_base = manifest["mirrors"]["github_release_base"].rstrip("/")
    hf_base = manifest["mirrors"]["hf_binary_fallback_base"].rstrip("/")
    headers = {"User-Agent": "llama-modchip-preflight/1.1"}

    # 1. Verify all dual-mirror binaries (HTTP 200, matching Content-Length, and HF LFS X-Linked-ETag SHA-256)
    for artifact, expected_sha in manifest.get("artifacts", {}).items():
        sizes: dict[str, int | None] = {}
        for label, url in (("GitHub", f"{gh_base}/{artifact}"), ("HF", f"{hf_base}/{artifact}")):
            redir = _CaptureRedirectHandler()
            opener = urllib.request.build_opener(redir)
            req = urllib.request.Request(url, method="HEAD", headers=headers)
            try:
                with opener.open(req, timeout=20) as resp:
                    if resp.status != 200:
                        g.fail(f"{label} mirror HTTP {resp.status} for {artifact}: {url}")
                    cl = resp.headers.get("Content-Length")
                    sizes[label] = int(cl) if cl and cl.isdigit() else None
                    if label == "HF":
                        linked_etag = resp.headers.get("X-Linked-Etag")
                        for rh in redir.redirect_headers:
                            if "x-linked-etag" in rh:
                                linked_etag = rh["x-linked-etag"]
                                break
                        hf_sha = _extract_hex64(linked_etag)
                        if not hf_sha:
                            g.fail(f"HF mirror missing X-Linked-ETag LFS SHA-256 header for {artifact}")
                        elif hf_sha != expected_sha:
                            g.fail(
                                f"HF LFS SHA-256 mismatch for {artifact}: "
                                f"remote={hf_sha[:12]}... vs manifest={expected_sha[:12]}..."
                            )
            except Exception as exc:
                g.fail(f"{label} mirror unreachable for {artifact}: {url} ({exc})")
        if sizes.get("GitHub") and sizes.get("HF") and sizes["GitHub"] != sizes["HF"]:
            g.fail(
                f"Mirror size mismatch for {artifact}: "
                f"GitHub={sizes['GitHub']} B vs HF={sizes['HF']} B"
            )

    # 2. Verify GitHub-only artifacts (release tarballs)
    for artifact in manifest.get("github_only_artifacts", {}):
        url = f"{gh_base}/{artifact}"
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                if resp.status != 200:
                    g.fail(f"GitHub release asset HTTP {resp.status} for {artifact}: {url}")
        except Exception as exc:
            g.fail(f"GitHub release asset unreachable for {artifact}: {url} ({exc})")

    # 3. Verify remote GitHub Release SHA256SUMS covers every binary, tarball, and active patch with exact SHA-256
    sums_url = f"{gh_base}/SHA256SUMS"
    patch_name = Path(manifest["patch_file"]).name
    patch_sha = manifest["patch_sha256"]
    try:
        req = urllib.request.Request(sums_url, method="GET", headers=headers)
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status != 200:
                g.fail(f"GitHub SHA256SUMS HTTP {resp.status}: {sums_url}")
            sums_txt = resp.read().decode("utf-8", errors="replace")
        remote_sums: dict[str, str] = {}
        for line in sums_txt.splitlines():
            m = re.match(r"^\s*([0-9a-fA-F]{64})\s+\*?(\S+)", line)
            if m:
                remote_sums[Path(m.group(2)).name] = m.group(1).lower()
        expected_remote: dict[str, str] = {}
        expected_remote.update(manifest.get("artifacts", {}))
        expected_remote.update(manifest.get("github_only_artifacts", {}))
        expected_remote[patch_name] = patch_sha
        for name, exp_sha in expected_remote.items():
            got = remote_sums.get(name)
            if got is None:
                g.fail(f"GitHub SHA256SUMS missing entry for {name}")
            elif got != exp_sha:
                g.fail(f"GitHub SHA256SUMS mismatch for {name}: remote={got[:12]}... vs manifest={exp_sha[:12]}...")
    except Exception as exc:
        g.fail(f"GitHub SHA256SUMS unreachable ({sums_url}): {exc}")

    # 4. Download & verify live remote patch SHA-256 on BOTH GitHub Release and HF mirror
    for label, url in (("GitHub", f"{gh_base}/{patch_name}"), ("HF", f"{hf_base}/{patch_name}")):
        try:
            req = urllib.request.Request(url, method="GET", headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status != 200:
                    g.fail(f"{label} remote patch HTTP {resp.status}: {url}")
                    continue
                got_sha = hashlib.sha256(resp.read()).hexdigest()
                if got_sha != patch_sha:
                    g.fail(
                        f"{label} remote {patch_name} SHA-256 drift: "
                        f"remote={got_sha[:12]}... vs manifest={patch_sha[:12]}..."
                    )
        except Exception as exc:
            g.fail(f"{label} remote patch unreachable ({url}): {exc}")


# ---------------------------------------------------------------------------
# Gate 12: Clean Working Tree Check
# ---------------------------------------------------------------------------
def check_clean_git(g: GateResult) -> None:
    res = run_cmd(["git", "status", "--porcelain"])
    if res.stdout.strip():
        g.fail(f"Working tree is dirty:\n{res.stdout.strip()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="llama-modchip Mandatory Release Preflight Gate")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    ap.add_argument("--run-cpp-tests", action="store_true", help="Compile & run RouteCrack 2.0 C++ ASan/UBSan tests")
    ap.add_argument("--verify-remote", action="store_true", help="Verify GitHub Release + HF binary fallback mirrors")
    ap.add_argument("--allow-dirty", action="store_true", help="Allow uncommitted local edits during development")
    args = ap.parse_args()

    try:
        manifest = load_manifest()
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    gates: list[tuple[str, Callable[[GateResult], None]]] = [
        ("1. Version, Commit, Build Number & Patch SHA-256", lambda g: check_version_and_build_parity(manifest, g)),
        ("2. ABI Enum Parity (ggml.h <-> gguf_tensor_type.py <-> Active Docs)", lambda g: check_abi_enums(manifest, g)),
        ("3. RouteCrack 2.0 Header Sync (speculative/cpp/rc_*.h <-> Patch)", lambda g: check_rc_header_sync(manifest, g)),
        ("4. Artifact SHA-256 Parity & Pinned HF MODEL_REV", lambda g: check_hashes_and_pinned_revs(manifest, g)),
        ("5. Benchmark JSON <-> Markdown Reconciliation", lambda g: check_benchmarks_vs_docs(manifest, g)),
        ("6. Technical Paper (main.tex <-> PDF <-> arxiv-submission.tar.gz)", lambda g: check_paper(manifest, g)),
        ("7. Links, Deleted File Refs & Issue Template Hygiene", lambda g: check_links_and_dead_refs(manifest, g)),
        ("8. Runtime Safety Invariants (AVX-512 VNNI, THP=madvise, /dev/shm UID)", lambda g: check_runtime_invariants(manifest, g)),
        ("9. Shell/Python Syntax, Unittests & Multi-ISA C++ Sanitizers", lambda g: check_syntax_and_tests(args.run_cpp_tests, g)),
        ("10. OPSEC & Proprietary Separation (Files + PDF + Figures)", lambda g: check_opsec(manifest, g)),
    ]
    if args.verify_remote:
        gates.append(("11. Remote Dual-Mirror Sync (GitHub + HF + SHA256SUMS + Patch)", lambda g: check_remote_mirrors(manifest, g)))
    if not args.allow_dirty:
        gates.append(("12. Clean Git Working Tree", check_clean_git))

    total_errors = 0
    total_warnings = 0
    print(f"=== llama-modchip Release Preflight Gate ({manifest['release_tag']}) ===")
    for name, fn in gates:
        res = GateResult(name)
        fn(res)
        status = "PASS" if res.ok and (not args.strict or not res.warnings) else "FAIL"
        print(f"[{status}] {name}")
        for w in res.warnings:
            print(f"   WARN: {w}")
        for e in res.errors:
            print(f"   FAIL: {e}")
        total_errors += len(res.errors)
        total_warnings += len(res.warnings)

    print(f"\nSummary: {total_errors} error(s), {total_warnings} warning(s)")
    if total_errors > 0 or (args.strict and total_warnings > 0):
        print("RELEASE BLOCKED: Fix all failures above before tagging or publishing.")
        return 1
    print("RELEASE GATE PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
