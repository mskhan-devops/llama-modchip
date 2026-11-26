#!/usr/bin/env python3
"""Native-accelerated 64-bit FNV-1a with a pure-Python fallback.

HARP-Cache hashes every 32-token block on both the store and resolve paths
(fnv1a_64 accounts for ~93% of resolve_prefix cumulative time).  The byte loop
is strictly sequential, so SIMD cannot help, but a single C pass is ~16x
faster than the Python loop.  This module compiles a tiny shared object once
per process and falls back to Python whenever a compiler is unavailable.

The hash is bit-identical to the reference implementation in harp_cache.py:
    h = seed; for b in data: h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
"""

import ctypes
import hashlib
import os
import shutil
import stat
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

FNV_OFFSET_BASIS = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK_64 = 0xFFFFFFFFFFFFFFFF

# Fixed-width 64-bit type.  ctypes.c_ulong is only 32-bit on Windows and
# would silently truncate the 64-bit seed there; ctypes.c_uint64 is always 8 bytes.
_U64 = ctypes.c_uint64

_C_SOURCE = """
#include <stdint.h>
#include <stddef.h>

uint64_t modchip_fnv1a_64(const uint8_t *data, size_t n, uint64_t seed) {
    uint64_t h = seed;
    for (size_t i = 0; i < n; i++) h = (h ^ data[i]) * UINT64_C(1099511628211);
    return h;
}

/* Chained FNV-1a over fixed-size blocks of little-endian int32 tokens packed
 * into a byte buffer.  out[k] = FNV64(seed || block_0 || ... || block_k);
 * returns the number of hashes written.  Equivalent to calling
 * modchip_fnv1a_64 on each block's 4*block bytes with the previous block's
 * hash as seed.
 *
 * `out_cap` is the number of uint64 slots in `out`; the function never writes
 * more than that.  Passing a capacity smaller than the number of blocks
 * truncates the chain rather than overflowing the buffer. */
size_t modchip_fnv_chain(const uint8_t *data, size_t nbytes,
                          size_t block, uint64_t seed,
                          uint64_t *out, size_t out_cap) {
    if (block == 0 || out == NULL) return 0;
    size_t nblocks = nbytes / (block * 4), k = 0;
    uint64_t h = seed;
    for (size_t b = 0; b < nblocks; b++) {
        if (k >= out_cap) break;
        const uint8_t *p = data + b * block * 4;
        for (size_t i = 0; i < block * 4; i++) {
            h = (h ^ p[i]) * UINT64_C(1099511628211);
        }
        out[k++] = h;
    }
    return k;
}

/* Build tag.  The loader rejects a cached artifact whose tag does not match
 * the current source, so editing the C source always forces a rebuild. */
uint32_t modchip_fnv_version(void) { return MODCHIP_FNV_VERSION; }
"""

# Source-derived build tag: changing _C_SOURCE changes this digest, which
# changes the artifact filename, so a stale .so can never be picked up.
_BUILD_TAG = hashlib.sha256(_C_SOURCE.encode("ascii")).hexdigest()[:12]

# Cached build artifact, next to this file.  *.so is gitignored.
_LIB_NAME = f".fnv_accel_{_BUILD_TAG}.so"
_LIB_PATH = Path(__file__).resolve().parent / _LIB_NAME


def _fallback_dir() -> Path:
    """Private per-user cache directory used when the source dir is read-only.

    The leaf directory is created with mode 0700 and validated on load, so a
    different local user cannot plant a symlink or a poisoned .so here.
    """
    try:
        uid = os.getuid()  # not available on Windows
    except AttributeError:
        uid = "default"
    return Path(tempfile.gettempdir()) / f"modchip-fnv-{uid}" / "cache"


def _trusted_dir(path: Path) -> bool:
    """Reject a cache directory that is a symlink or writable by others.

    Both the primary (next to the source) and the fallback location are
    checked.  A world/group-WRITABLE directory would let another local user
    replace the artifact with a library that passes the version check but
    returns wrong hashes, silently poisoning HARP-Cache lookups.  Read and
    execute bits are fine — the source directory is usually 0755.
    """
    try:
        st = path.lstat()  # do not follow a symlink
        if not stat.S_ISDIR(st.st_mode):
            return False
        if st.st_mode & 0o022:  # group- or world-writable
            return False
        try:
            if st.st_uid != os.getuid():
                return False
        except AttributeError:  # Windows
            pass
        return True
    except OSError:
        return False


def _candidate_paths() -> list:
    """Every location the cached artifact may live in, most preferred first."""
    return [_LIB_PATH, _fallback_dir() / _LIB_NAME]


def _fnv1a_python(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    h = seed
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK_64
    return h


def _bind(lib) -> tuple:
    fn = lib.modchip_fnv1a_64
    fn.argtypes = [ctypes.c_char_p, ctypes.c_size_t, _U64]
    fn.restype = _U64
    chain = lib.modchip_fnv_chain
    chain.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                      ctypes.c_size_t, _U64,
                      ctypes.POINTER(_U64), ctypes.c_size_t]
    chain.restype = ctypes.c_size_t
    ver = lib.modchip_fnv_version
    ver.argtypes = []
    ver.restype = ctypes.c_uint32
    return fn, chain, ver


def _probe(fn_native) -> bool:
    """Verify bit-identity of BOTH native functions against the reference.

    Checking only the scalar hash is not enough: a poisoned artifact could
    implement `modchip_fnv1a_64` correctly and return garbage from
    `modchip_fnv_chain`, which is what the HARP hot path actually uses.
    """
    probe = bytes(range(256))
    tokens = [0, 1, -1, 2**31 - 1, -(2**31)] * 8  # 40 tokens -> 1 full block
    try:
        for seed in (FNV_OFFSET_BASIS, 0, 0xFFFFFFFFFFFFFFFF):
            if fn_native[0](probe, len(probe), seed) != _fnv1a_python(probe, seed):
                return False
        raw = struct.Struct(f"<{len(tokens)}i").pack(*tokens)
        buf = bytearray(8)
        out = (_U64 * 1).from_buffer(buf)
        written = fn_native[1](raw, len(raw), 8, FNV_OFFSET_BASIS, out, 1)
        expected = FNV_OFFSET_BASIS
        for i in range(0, 8 * 4):
            expected = ((expected ^ raw[i]) * FNV_PRIME) & MASK_64
        return written == 1 and out[0] == expected
    except (OSError, ValueError, ctypes.ArgumentError, struct.error):
        return False


def _load_cached() -> Optional[tuple]:
    """Load a previously compiled shared object, if present and valid.

    A cached artifact that fails the version check or the bit-identity probe
    is removed so that the caller rebuilds and replaces it instead of
    silently falling back to the slow Python path.
    """
    for path in _candidate_paths():
        if not _trusted_dir(path.parent):
            continue
        try:
            if not path.exists():
                continue
            if path.is_symlink():
                _discard_cached(path)
                continue
            lib = ctypes.CDLL(str(path))
            fn = _bind(lib)
        except (OSError, ValueError, AttributeError, ctypes.ArgumentError):
            _discard_cached(path)
            continue
        try:
            if fn[2]() != _version_number():
                raise ValueError("cached artifact has a stale build tag")
        except (OSError, ValueError):
            _discard_cached(path)
            continue
        if not _probe(fn):
            _discard_cached(path)
            continue
        return fn
    return None


def _discard_cached(path: Optional[Path] = None) -> None:
    """Best-effort removal of a bad cached artifact."""
    try:
        (path or _LIB_PATH).unlink()
    except OSError:
        pass


def _version_number() -> int:
    """Numeric build tag embedded in the shared object."""
    return int(_BUILD_TAG[:8], 16)


def _install(so: Path) -> Optional[Path]:
    """Install the compiled artifact at a persistent path, atomically.

    The file is written to a unique temporary name in the target directory and
    then moved into place with os.replace().  That is atomic within a
    filesystem, so a concurrent importer sees either the old or the new
    library but never a half-written one, and a library that is already
    mmap'd by another process is never overwritten in place.

    The preferred location is next to this source file.  If that directory is
    read-only (e.g. a system-wide install), a per-user directory under the
    system temp dir is used instead, so the artifact is still cached across
    processes.

    Returns the persistent path to dlopen, or None if no location is
    writable.
    """
    for target_dir in (_LIB_PATH.parent, _fallback_dir()):
        try:
            target_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not _trusted_dir(target_dir):
                # A symlinked or world/group-writable cache directory could
                # redirect the artifact somewhere another local user
                # controls.  Refuse rather than write through it.
                continue
            fd, tmp_name = tempfile.mkstemp(prefix=".fnv_accel-",
                                            suffix=".so",
                                            dir=str(target_dir))
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                shutil.copyfile(str(so), str(tmp))
                os.chmod(str(tmp), 0o755)
                final = target_dir / _LIB_NAME
                os.replace(str(tmp), str(final))
                return final
            finally:
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
        except OSError:
            continue
    return None


def _compile_and_load() -> Optional[tuple]:
    """Compile the C hash and load it.

    The artifact is installed at a persistent path *before* it is dlopen'd:
    on Windows a loaded DLL cannot be deleted, so loading it from inside the
    temporary build directory would make the cleanup fail and silently
    disable the native path.
    """
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not compiler:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="modchip-fnv-") as d:
            src = Path(d) / "fnv.c"
            so = Path(d) / "fnv.so"
            src.write_text(
                _C_SOURCE.replace("MODCHIP_FNV_VERSION",
                                  f"UINT32_C({_version_number()})"),
                encoding="ascii")
            res = subprocess.run(
                [compiler, "-O3", "-shared", "-fPIC", "-o", str(so), str(src)],
                capture_output=True, text=True, timeout=30,
            )
            if res.returncode != 0 or not so.exists():
                return None
            persistent = _install(so)
            if persistent is None:
                return None
            try:
                fn = _bind(ctypes.CDLL(str(persistent)))
            except (OSError, ValueError, AttributeError, ctypes.ArgumentError):
                return None
            if not _probe(fn):
                return None
            return fn
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _load_native() -> Optional[tuple]:
    """Load the C hash (cached build preferred), or return None if unavailable."""
    fn = _load_cached()
    if fn is None:
        fn = _compile_and_load()
    if fn is None:
        return None
    # Final bit-identity check before the fast path is trusted.
    if not _probe(fn):
        return None
    return fn


_native = _load_native()


def fnv1a_64(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    """Bit-identical 64-bit FNV-1a, native when available, Python otherwise."""
    if _native is not None:
        if not isinstance(data, bytes):
            data = bytes(data)
        return _native[0](data, len(data), seed)
    return _fnv1a_python(data, seed)


def _pack_int32(tokens) -> bytes:
    """Pack token IDs into little-endian int32 bytes.

    Packs the WHOLE list rather than only the complete blocks: copying a
    32k-token slice costs ~0.08 ms, and the C side already floors the byte
    count to whole blocks, so the trailing bytes are ignored.

    Real token IDs are always small non-negative integers.  If a value is
    outside int32 range, let the caller fall back to the per-block path
    rather than guessing how to encode it — a silently zero-padded buffer
    would produce a hash the native llama.cpp side cannot reproduce.
    """
    n = len(tokens)
    return struct.Struct(f"<{n}i").pack(*tokens)


def fnv_chain_blocks_with_bytes(tokens, block: int, seed: int = FNV_OFFSET_BASIS):
    """Chained FNV-1a over `block`-sized chunks of `tokens`, plus the packed bytes.

    Returns ``(hashes, packed)`` where ``hashes`` is the list of chained hashes
    (one per complete block) and ``packed`` is the little-endian int32 buffer
    the hashes were computed over.  ``packed`` lets the caller slice out each
    block's bytes directly instead of slicing the token list, which is ~1.9x
    cheaper.

    The hashes are identical to calling `fnv1a_64` on each block's packed
    little-endian bytes with the previous block's hash as seed.  Returns
    ``(None, None)`` when the native implementation is unavailable, so callers
    can fall back to the per-block path.
    """
    if _native is None:
        return None, None
    n_blocks = len(tokens) // block
    if n_blocks == 0:
        return [], b""
    try:
        raw = _pack_int32(tokens)
        buf = bytearray(8 * n_blocks)
        out = (_U64 * n_blocks).from_buffer(buf)
        _native[1](raw, len(raw), block, seed, out, n_blocks)
        return memoryview(buf).cast("Q").tolist(), raw
    except (ValueError, TypeError, OverflowError, MemoryError, struct.error):
        return None, None


def fnv_chain_blocks(tokens, block: int, seed: int = FNV_OFFSET_BASIS):
    """Chained FNV-1a over `block`-sized chunks of `tokens`.

    Returns the list of chained hashes, one per complete block.  The result is
    identical to calling `fnv1a_64` on each block's packed little-endian bytes
    with the previous block's hash as seed.  Returns None when the native
    implementation is unavailable, so callers can fall back to the per-block
    path.
    """
    if _native is None:
        return None
    n_blocks = len(tokens) // block
    if n_blocks == 0:
        return []
    try:
        raw = _pack_int32(tokens)
        # The chain writes uint64 results into a byte buffer; reading them back
        # through memoryview().cast('Q').tolist() is ~2.9x faster than
        # indexing the ctypes array element by element.
        buf = bytearray(8 * n_blocks)
        out = (_U64 * n_blocks).from_buffer(buf)
        _native[1](raw, len(raw), block, seed, out, n_blocks)
        return memoryview(buf).cast("Q").tolist()
    except (ValueError, TypeError, OverflowError, MemoryError, struct.error):
        return None
