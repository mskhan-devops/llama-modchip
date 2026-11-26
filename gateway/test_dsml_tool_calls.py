"""Regression tests for DeepSeek DSML tool calls and SSE stream hardening.

The compact-tools prompt in sse_gateway.py (`MODCHIP_COMPACT_TOOLS` / dsv4 tools injection) and official DeepSeek V3.2/V4
Jinja templates emit `<｜DSML｜tool_calls><｜DSML｜invoke name="TOOL_NAME"><｜DSML｜parameter name="PARAM"
string="true|false">VAL</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>` (with U+FF5C DSML markers in both opening and
closing tags), while stripped variants emit plain `</parameter></invoke></tool_calls>` closers. Both formats are tested
across single-shot parsing, arbitrary two-way SSE delta splits with trailing prose, O(N) truncated-tag ReDoS bounds, and
empty-schema (`schemas={}`) whitelist enforcement.

Run with: python -m pytest -q gateway/test_dsml_tool_calls.py
"""

import importlib.util
import json
import os
import time
import unittest
from pathlib import Path

GATEWAY_PATH = Path(__file__).resolve().parent / "sse_gateway.py"


def _load_gateway():
    # Keep the import side-effect free: the module opens /dev/shm when HARP_CACHE is enabled, which a unit test
    # must not do. Restore the caller's environment.
    previous = os.environ.get("HARP_CACHE")
    os.environ["HARP_CACHE"] = "0"
    try:
        spec = importlib.util.spec_from_file_location("sse_gateway_dsml_under_test", GATEWAY_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("HARP_CACHE", None)
        else:
            os.environ["HARP_CACHE"] = previous
    return module


gw = _load_gateway()

DSML = "｜DSML｜"
DSML_OPEN = f"<{DSML}tool_calls>"
DSML_CLOSE_FULL = f"</{DSML}tool_calls>"
DSML_CLOSE_PLAIN = "</tool_calls>"
SCHEMAS = {
    "Read": {"file_path": {"type": "string"}, "offset": {"type": "integer"}},
    "Bash": {"command": {"type": "string"}, "run_in_background": {"type": "boolean"}},
}


def _invoke(name, params, full_close=True):
    p_close = f"</{DSML}parameter>" if full_close else "</parameter>"
    i_close = f"</{DSML}invoke>" if full_close else "</invoke>"
    parts = []
    for item in params:
        if len(item) == 3:
            key, value, string_attr = item
        else:
            key, value = item
            string_attr = "true" if isinstance(value, str) else "false"
        parts.append(f'<{DSML}parameter name="{key}" string="{string_attr}">{value}{p_close}')
    return f'<{DSML}invoke name="{name}">' + "".join(parts) + i_close


def _block(*invokes, full_close=True):
    closer = DSML_CLOSE_FULL if full_close else DSML_CLOSE_PLAIN
    return DSML_OPEN + "".join(invokes) + closer


def _event(type_, **payload):
    payload["type"] = type_
    return f"event: {type_}\ndata: {json.dumps(payload)}\n\n"


def _stream(parts):
    out = [
        _event(
            "message_start",
            message={
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        ),
        _event("content_block_start", index=0, content_block={"type": "text", "text": ""}),
    ]
    for part in parts:
        out.append(_event("content_block_delta", index=0, delta={"type": "text_delta", "text": part}))
    out.append(_event("content_block_stop", index=0))
    out.append(
        _event(
            "message_delta",
            delta={"stop_reason": "end_turn", "stop_sequence": None},
            usage={"output_tokens": 1},
        )
    )
    out.append(_event("message_stop"))
    return "".join(out)


def _events(rendered):
    events = []
    for block in rendered.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class ParserTest(unittest.TestCase):
    def test_the_compact_tools_prompt_advertises_the_syntax_the_parser_reads(self):
        source = GATEWAY_PATH.read_text(encoding="utf-8")
        self.assertIn(f"invoke via {DSML_OPEN}", source)
        self.assertIn(f'<{DSML}invoke name=', source)
        self.assertIn(f'<{DSML}parameter name=', source)
        self.assertIn(f"</{DSML}parameter></{DSML}invoke>{DSML_CLOSE_FULL}", source)

    def test_reads_both_full_and_plain_closing_tags(self):
        for full_close in (True, False):
            with self.subTest(full_close=full_close):
                text = _block(
                    _invoke("Read", [("file_path", "/etc/hosts"), ("offset", 3)], full_close=full_close),
                    full_close=full_close,
                )
                self.assertEqual(
                    gw.try_parse_tool_call(text, SCHEMAS),
                    {"name": "Read", "input": {"file_path": "/etc/hosts", "offset": 3}},
                )

    def test_parameter_types_follow_schema_and_string_attribute(self):
        # string="true" stays a string on string properties ("true"), while explicit integer/boolean schema
        # properties coerce cleanly even if the model mistakenly emits string="true".
        text = _block(
            _invoke("Bash", [("command", "true", "true"), ("run_in_background", False, "false")])
        )
        self.assertEqual(
            gw.try_parse_tool_call(text, SCHEMAS),
            {"name": "Bash", "input": {"command": "true", "run_in_background": False}},
        )
        coerced = _block(_invoke("Read", [("file_path", "/etc/hosts", "true"), ("offset", "42", "true")]))
        self.assertEqual(
            gw.try_parse_tool_call(coerced, SCHEMAS),
            {"name": "Read", "input": {"file_path": "/etc/hosts", "offset": 42}},
        )

    def test_unknown_tool_is_rejected_when_schemas_are_given(self):
        text = _block(_invoke("Nope", [("x", "1")]))
        self.assertIsNone(gw.try_parse_tool_call(text, SCHEMAS))

    def test_empty_schemas_rejects_all_tool_calls(self):
        dsml_text = _block(_invoke("Bash", [("command", "id")]))
        json_text = '<tool_call>{"name": "Bash", "arguments": {"command": "id"}}</tool_call>'
        xml_text = "<tool_call><function=Bash><parameter=command>id</parameter></function></tool_call>"
        for payload in (dsml_text, json_text, xml_text):
            with self.subTest(payload=payload[:24]):
                self.assertIsNone(gw.try_parse_tool_call(payload, {}))
                self.assertEqual(gw.parse_tool_calls(payload, {}), [])
                resp = gw.sanitize_non_streaming_response(
                    {"content": [{"type": "text", "text": payload}]}, {}
                )
                self.assertEqual([b for b in resp["content"] if b.get("type") == "tool_use"], [])

    def test_plain_text_is_not_a_tool_call(self):
        self.assertIsNone(gw.try_parse_tool_call("just an answer", SCHEMAS))

    def test_non_streaming_text_becomes_deduped_capped_tool_use_blocks(self):
        block = _block(
            _invoke("Read", [("file_path", "/a")]),
            _invoke("Read", [("file_path", "/a")]),  # duplicate
            _invoke("Bash", [("command", "ls")]),
            _invoke("Read", [("file_path", "/b")]),  # exceeds default max_tool_calls=2
        )
        resp = gw.sanitize_non_streaming_response({"content": [{"type": "text", "text": block}]}, SCHEMAS)
        uses = [b for b in resp["content"] if b.get("type") == "tool_use"]
        self.assertEqual(
            [(u["name"], u["input"]) for u in uses],
            [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls"})],
        )
        self.assertEqual(resp["stop_reason"], "tool_use")

    def test_truncated_or_unclosed_dsml_parses_in_linear_time(self):
        unclosed_tag = f"{DSML_OPEN}<{DSML}invoke " + (" " * 20000)
        repeated_openers = DSML_OPEN + (f'<{DSML}invoke name="Read">' * 2000)
        for label, payload in (("20kb_ws_no_gt", unclosed_tag), ("2000_unclosed_invokes", repeated_openers)):
            with self.subTest(case=label):
                t0 = time.perf_counter()
                parsed = gw.parse_dsml_tool_calls(payload, SCHEMAS)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                self.assertEqual(parsed, [])
                self.assertLess(elapsed_ms, 20.0, f"{label} took {elapsed_ms:.2f} ms")


class DsmlStreamingTest(unittest.TestCase):
    def _run(self, parts, schemas=SCHEMAS):
        stream_filter = gw.ThinkBlockFilter(mode="med", schemas=schemas)
        rendered = stream_filter.feed(_stream(parts)) + stream_filter.flush()
        return _events(rendered)

    def _tool_uses(self, events):
        starts = [
            e
            for e in events
            if e["type"] == "content_block_start" and e.get("content_block", {}).get("type") == "tool_use"
        ]
        deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta" and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        self.assertEqual(len(starts), len(deltas), events)
        return [
            (start["content_block"]["name"], json.loads(delta["delta"]["partial_json"]))
            for start, delta in zip(starts, deltas)
        ]

    def _assert_single_read(self, events):
        self.assertEqual(
            self._tool_uses(events),
            [("Read", {"file_path": "/etc/hosts"})],
            events,
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_dsml_block_in_one_delta(self):
        for full_close in (True, False):
            with self.subTest(full_close=full_close):
                block = _block(_invoke("Read", [("file_path", "/etc/hosts")], full_close=full_close), full_close=full_close)
                self._assert_single_read(self._run([block]))

    def test_dsml_block_split_at_every_position(self):
        for full_close in (True, False):
            text = _block(_invoke("Read", [("file_path", "/etc/hosts")], full_close=full_close), full_close=full_close)
            for split in range(1, len(text)):
                with self.subTest(full_close=full_close, split=split):
                    self._assert_single_read(self._run([text[:split], text[split:]]))

    def test_text_before_and_after_the_block_is_kept_across_all_splits(self):
        # Covers all 234 split positions of official </｜DSML｜tool_calls> and all 151 split positions of plain </tool_calls>
        for full_close in (True, False):
            text = (
                "Checking.\n"
                + _block(_invoke("Read", [("file_path", "/etc/hosts")], full_close=full_close), full_close=full_close)
                + "\nDone."
            )
            for split in range(1, len(text)):
                with self.subTest(full_close=full_close, split=split):
                    events = self._run([text[:split], text[split:]])
                    self._assert_single_read(events)
                    visible = "".join(
                        e["delta"]["text"]
                        for e in events
                        if e["type"] == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta"
                    )
                    self.assertIn("Checking.", visible)
                    self.assertIn("Done.", visible)
                    self.assertNotIn(f"<{DSML}", visible)
                    self.assertNotIn(DSML_CLOSE_FULL, visible)
                    self.assertNotIn(DSML_CLOSE_PLAIN, visible)

    def test_two_invokes_emit_two_tool_uses(self):
        text = _block(
            _invoke("Read", [("file_path", "/a")]),
            _invoke("Bash", [("command", "ls"), ("run_in_background", False)]),
        )
        events = self._run([text])
        self.assertEqual(
            self._tool_uses(events),
            [("Read", {"file_path": "/a"}), ("Bash", {"command": "ls", "run_in_background": False})],
            events,
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_streaming_empty_schemas_never_emits_tool_use(self):
        for payload in (
            _block(_invoke("Bash", [("command", "id")])),
            '<tool_call>{"name": "Bash", "arguments": {"command": "id"}}</tool_call>',
        ):
            with self.subTest(payload=payload[:20]):
                events = self._run([payload], schemas={})
                self.assertEqual(self._tool_uses(events), [])

    def test_unclosed_tool_buf_stays_bounded(self):
        stream_filter = gw.ThinkBlockFilter(mode="med", schemas=SCHEMAS)
        big_chunk = "x" * 65536
        stream_filter.feed(_event("content_block_delta", index=0, delta={"type": "text_delta", "text": DSML_OPEN}))
        for _ in range(6):
            stream_filter.feed(_event("content_block_delta", index=0, delta={"type": "text_delta", "text": big_chunk}))
            self.assertLessEqual(len(stream_filter.tool_buf), gw._MAX_TOOL_BUF_BYTES)


if __name__ == "__main__":
    unittest.main()
