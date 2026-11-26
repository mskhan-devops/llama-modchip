"""Regression tests for the Anthropic SSE normalizer in sse_gateway.py."""

import importlib.util
import json
import os
import unittest
from pathlib import Path


def _load_gateway():
    # Keep the import side-effect free: the module opens /dev/shm when HARP_CACHE
    # is enabled, which a unit test must not do. Restore the caller's environment.
    previous = os.environ.get("HARP_CACHE")
    os.environ["HARP_CACHE"] = "0"
    try:
        path = Path(__file__).resolve().parent / "sse_gateway.py"
        spec = importlib.util.spec_from_file_location("sse_gateway_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            os.environ.pop("HARP_CACHE", None)
        else:
            os.environ["HARP_CACHE"] = previous
    return module


gw = _load_gateway()

TOOL_OPEN = "<tool_call>"
TOOL_CLOSE = "</tool_call>"


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
        _event(
            "content_block_start",
            index=0,
            content_block={"type": "text", "text": ""},
        ),
    ]
    for part in parts:
        out.append(
            _event(
                "content_block_delta",
                index=0,
                delta={"type": "text_delta", "text": part},
            )
        )
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


class ToolCallStreamingTest(unittest.TestCase):
    def _run(self, parts):
        stream_filter = gw.ThinkBlockFilter(
            mode="med", schemas={"Bash": {"command": {"type": "string"}}}
        )
        rendered = stream_filter.feed(_stream(parts)) + stream_filter.flush()
        return _events(rendered)

    def _assert_single_bash_call(self, events):
        starts = [
            e
            for e in events
            if e["type"] == "content_block_start"
            and e.get("content_block", {}).get("type") == "tool_use"
        ]
        self.assertEqual(len(starts), 1, events)
        self.assertEqual(starts[0]["content_block"]["name"], "Bash")
        deltas = [
            e
            for e in events
            if e["type"] == "content_block_delta"
            and e.get("delta", {}).get("type") == "input_json_delta"
        ]
        self.assertEqual(len(deltas), 1, events)
        self.assertEqual(
            json.loads(deltas[0]["delta"]["partial_json"]), {"command": "ls"}
        )
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

    def test_tool_call_in_one_delta(self):
        text = (
            TOOL_OPEN
            + '{"name": "Bash", "arguments": {"command": "ls"}}'
            + TOOL_CLOSE
        )
        self._assert_single_bash_call(self._run([text]))

    def test_close_tag_split_across_deltas(self):
        text = (
            TOOL_OPEN
            + '{"name": "Bash", "arguments": {"command": "ls"}}'
            + TOOL_CLOSE
        )
        for split in range(1, len(text)):
            with self.subTest(split=split):
                self._assert_single_bash_call(self._run([text[:split], text[split:]]))

    def test_text_before_and_after_tool_call(self):
        text = (
            "Running it now.\n"
            + TOOL_OPEN
            + '{"name": "Bash", "arguments": {"command": "ls"}}'
            + TOOL_CLOSE
            + "\nDone."
        )
        for split in range(1, len(text)):
            with self.subTest(split=split):
                events = self._run([text[:split], text[split:]])
                self._assert_single_bash_call(events)
                visible = "".join(
                    e["delta"]["text"]
                    for e in events
                    if e["type"] == "content_block_delta"
                    and e.get("delta", {}).get("type") == "text_delta"
                )
                self.assertIn("Running it now.", visible)
                self.assertIn("Done.", visible)
                self.assertNotIn("<tool_call>", visible)
                self.assertNotIn("</tool_call>", visible)


class NativeBlockIndexTest(unittest.TestCase):
    """Native thinking-after-text streams must never reuse or orphan block indices."""

    def _native_stream(self):
        def cb_start(index, cb):
            return _event("content_block_start", index=index, content_block=cb)

        def cb_delta(index, delta):
            return _event("content_block_delta", index=index, delta=delta)

        return (
            _event(
                "message_start",
                message={
                    "id": "msg", "type": "message", "role": "assistant",
                    "content": [], "model": "test", "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
            + cb_start(0, {"type": "text", "text": ""})
            + cb_delta(0, {"type": "text_delta", "text": "Here"})
            + _event("content_block_stop", index=0)
            + cb_start(1, {"type": "thinking", "thinking": ""})
            + cb_delta(1, {"type": "thinking_delta", "thinking": "double check"})
            + _event("content_block_stop", index=1)
            + cb_start(2, {"type": "text", "text": ""})
            + cb_delta(2, {"type": "text_delta", "text": "Final: 42."})
            + _event("content_block_stop", index=2)
            + _event(
                "message_delta",
                delta={"stop_reason": "end_turn", "stop_sequence": None},
                usage={"output_tokens": 3},
            )
            + _event("message_stop")
        )

    def test_native_thinking_after_text_uses_distinct_indices(self):
        stream = self._native_stream()
        for feed_size in (1, 2, 3, 5, 7, 11, 13, 4096):
            with self.subTest(feed=feed_size):
                stream_filter = gw.ThinkBlockFilter(mode="med", schemas={})
                rendered = ""
                for i in range(0, len(stream), feed_size):
                    rendered += stream_filter.feed(stream[i:i + feed_size])
                rendered += stream_filter.flush()
                events = _events(rendered)

                starts = [
                    (e["index"], e["content_block"]["type"])
                    for e in events
                    if e["type"] == "content_block_start"
                ]
                stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
                deltas = [
                    (e["index"], e["delta"]["type"])
                    for e in events
                    if e["type"] == "content_block_delta"
                ]
                opened = [i for i, _ in starts]
                self.assertEqual(
                    len(opened), len(set(opened)),
                    f"duplicate block index in {starts}",
                )
                self.assertEqual(
                    sorted(opened), sorted(set(stops)),
                    "every opened block must be stopped exactly once",
                )
                self.assertTrue(
                    all(i in opened for i, _ in deltas),
                    f"delta on a block that never opened: {deltas}",
                )
                self.assertEqual(
                    [t for _, t in starts], ["text", "thinking", "text"]
                )

    def test_native_thinking_after_text_in_no_think_mode(self):
        """mode=no still emits the text block; its stop must be closed when thinking follows."""
        stream = self._native_stream()
        for feed_size in (1, 3, 7, 64, 4096):
            with self.subTest(feed=feed_size):
                stream_filter = gw.ThinkBlockFilter(mode="no", schemas={})
                rendered = ""
                for i in range(0, len(stream), feed_size):
                    rendered += stream_filter.feed(stream[i:i + feed_size])
                rendered += stream_filter.flush()
                events = _events(rendered)

                opened = [
                    e["index"] for e in events
                    if e["type"] == "content_block_start"
                ]
                stopped = [
                    e["index"] for e in events
                    if e["type"] == "content_block_stop"
                ]
                self.assertEqual(len(opened), len(set(opened)),
                                 f"duplicate block index in {opened}")
                self.assertEqual(sorted(opened), sorted(stopped),
                                 f"block lifecycle mismatch: {opened} vs {stopped}")
                # thinking content must be suppressed in mode=no
                self.assertFalse(any(
                    e.get("delta", {}).get("type") == "thinking_delta"
                    for e in events
                ))


if __name__ == "__main__":
    unittest.main()
