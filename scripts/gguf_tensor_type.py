#!/usr/bin/env python3
"""Print the GGML type of the routed-expert tensors in a GGUF file.

The trunk-quantization profiles keep the routed experts at whatever type the
input already uses, and only touch the always-active trunk. llama-quantize
copies a tensor unchanged when the requested type equals its current type, so
the builder needs that current type as its base type. Reading it from the GGUF
header avoids assuming Q4_0 (or requantizing the experts by accident).

Usage:
    python3 scripts/gguf_tensor_type.py MODEL.gguf --match ffn_up_exps --match ffn_down_exps

Prints the most common type among the matching tensors, lowercase (q4_0,
q5_0, ...), or exits 1 with a message. A sharded model is read across all of
its shards. --verbose writes the per-type counts to stderr.
"""

from __future__ import annotations

import argparse
import collections
import re
import struct
import sys
from pathlib import Path

# ggml/include/ggml.h, GGML_TYPE_*.
GGML_TYPES = {
    0: "f32",
    1: "f16",
    2: "q4_0",
    3: "q4_1",
    6: "q5_0",
    7: "q5_1",
    8: "q8_0",
    9: "q8_1",
    10: "q2_k",
    11: "q3_k",
    12: "q4_k",
    13: "q5_k",
    14: "q6_k",
    15: "q8_k",
    16: "iq2_xxs",
    17: "iq2_xs",
    18: "iq3_xxs",
    19: "iq1_s",
    20: "iq4_nl",
    21: "iq3_s",
    22: "iq2_s",
    23: "iq4_xs",
    29: "iq1_m",
    30: "bf16",
    34: "tq1_0",
    35: "tq2_0",
    39: "mxfp4",
    40: "nvfp4",
    41: "q1_0",
    42: "q2_0",
    43: "f8_e4m3",
    44: "fractal_64",
    45: "vq3_64",
}

# GGUF value types (gguf.h).
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL = range(8)
_STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = 8, 9, 10, 11, 12
_FIXED_VALUE_SIZE = {
    _UINT8: 1, _INT8: 1, _UINT16: 2, _INT16: 2,
    _UINT32: 4, _INT32: 4, _FLOAT32: 4, _BOOL: 1,
    _UINT64: 8, _INT64: 8, _FLOAT64: 8,
}


def _read(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError("unexpected end of GGUF header")
    return data


def _skip_value(f, vtype: int) -> None:
    size = _FIXED_VALUE_SIZE.get(vtype)
    if size is not None:
        _read(f, size)
        return
    if vtype == _STRING:
        _read(f, struct.unpack("<Q", _read(f, 8))[0])
        return
    if vtype == _ARRAY:
        atype, alen = struct.unpack("<IQ", _read(f, 12))
        size = _FIXED_VALUE_SIZE.get(atype)
        if size is not None:
            _read(f, alen * size)
        elif atype == _STRING:
            for _ in range(alen):
                _read(f, struct.unpack("<Q", _read(f, 8))[0])
        elif atype == _ARRAY:
            for _ in range(alen):
                _skip_value(f, _ARRAY)
        else:
            raise ValueError(f"unknown GGUF array element type {atype}")
        return
    raise ValueError(f"unknown GGUF value type {vtype}")


def tensor_types(path: Path):
    """Yield (name, type id) for every tensor in one GGUF file."""
    with path.open("rb") as f:
        if _read(f, 4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        _version, n_tensors, n_kv = struct.unpack("<IQQ", _read(f, 20))
        for _ in range(n_kv):
            key_len = struct.unpack("<Q", _read(f, 8))[0]
            _read(f, key_len)
            _skip_value(f, struct.unpack("<I", _read(f, 4))[0])
        for _ in range(n_tensors):
            name_len = struct.unpack("<Q", _read(f, 8))[0]
            name = _read(f, name_len).decode("utf-8", errors="replace")
            n_dims = struct.unpack("<I", _read(f, 4))[0]
            _read(f, 8 * n_dims)
            type_id = struct.unpack("<I", _read(f, 4))[0]
            _read(f, 8)  # offset
            yield name, type_id


def shard_paths(path: Path) -> list[Path]:
    """A split GGUF's shards, or the single file."""
    match = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name)
    if not match:
        return [path]
    prefix = path.name[: match.start()]
    total = int(match.group(2))
    shards = [path.with_name(f"{prefix}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]
    missing = [p for p in shards if not p.is_file()]
    if missing:
        raise ValueError(f"missing GGUF shard: {missing[0]}")
    return shards


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--match", action="append", default=[], metavar="SUBSTR",
                    help="tensor-name substring to include (repeatable; default: routed experts)")
    ap.add_argument("--verbose", action="store_true", help="print every type count to stderr")
    args = ap.parse_args()

    matches = args.match or ["ffn_up_exps", "ffn_down_exps", "ffn_gate_exps"]
    counts: collections.Counter[int] = collections.Counter()
    for shard in shard_paths(args.model):
        for name, type_id in tensor_types(shard):
            if any(m in name for m in matches):
                counts[type_id] += 1

    if not counts:
        print(f"error: no tensor matching {matches} in {args.model}", file=sys.stderr)
        return 1
    unknown = sorted(set(counts) - set(GGML_TYPES))
    if unknown:
        print(f"error: unmapped GGML type id(s) {unknown}; add them to GGML_TYPES", file=sys.stderr)
        return 1

    if args.verbose or len(counts) > 1:
        detail = ", ".join(f"{GGML_TYPES[t]}={n}" for t, n in counts.most_common())
        print(f"expert tensor types: {detail}", file=sys.stderr)
    print(GGML_TYPES[counts.most_common(1)[0][0]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
