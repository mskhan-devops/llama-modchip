"""Tests for the native-accelerated FNV-1a hash used by HARP-Cache.

The hash is on the hot path of every store/resolve (93% of resolve_prefix
cumulative time).  The native implementation must be bit-identical to the
reference Python loop for every input, and the fallback must work when no
compiler is available.
"""

import random
import struct
import unittest

from kv_cache import fnv_accel
from kv_cache.harp_cache import FNV_OFFSET_BASIS, fnv1a_64, hash_token_block


class NativeFNVTest(unittest.TestCase):
    def test_native_matches_python_reference_on_random_buffers(self):
        """The C implementation must be bit-identical for arbitrary bytes+seeds."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        rng = random.Random(0xC0FFEE)
        for _ in range(500):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 512)))
            seed = rng.getrandbits(64)
            self.assertEqual(
                fnv_accel._native[0](data, len(data), seed),
                fnv_accel._fnv1a_python(data, seed),
            )

    def test_harp_fnv_is_bit_identical_to_reference(self):
        """harp_cache.fnv1a_64 delegates without changing hash semantics."""
        rng = random.Random(7)
        for _ in range(200):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(0, 128)))
            seed = rng.getrandbits(64)
            self.assertEqual(fnv1a_64(data, seed), fnv_accel._fnv1a_python(data, seed))

    def test_empty_buffer_and_known_vector(self):
        self.assertEqual(fnv1a_64(b""), FNV_OFFSET_BASIS)
        # FNV-1a 64-bit of ASCII "a" is a published test vector.
        self.assertEqual(fnv1a_64(b"a"), 0xAF63DC4C8601EC8C)

    def test_hash_token_block_non_32_token_branch(self):
        """The fallback branch for partial blocks must still hash correctly."""
        import struct as _struct

        seed = 0x123456789ABCDEF
        self.assertEqual(
            hash_token_block([1, 2, 3], prev_hash=seed),
            fnv1a_64(_struct.pack("<3i", 1, 2, 3), seed),
        )

    def test_hash_token_block_chain_unchanged(self):
        """Causal chaining via prev_hash must produce identical results."""
        block = [1000 + i for i in range(32)]
        h0 = hash_token_block(block)
        h1 = hash_token_block(block, prev_hash=h0)
        self.assertNotEqual(h0, h1)
        self.assertEqual(h0, fnv1a_64(__import__("struct").pack("<32i", *block)))

    def test_batched_chain_bit_identical_to_per_block(self):
        """The batched native chain must equal sequential hash_token_block calls."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        rng = random.Random(0x5EED)
        for _ in range(100):
            nblocks = rng.randint(1, 40)
            tokens = [rng.randint(-(2**31), 2**31 - 1) for _ in range(32 * nblocks)]
            seed = rng.getrandbits(64)
            expected = []
            h = seed
            for b in range(nblocks):
                h = fnv1a_64(struct.pack("<32i", *tokens[b*32:(b+1)*32]), h)
                expected.append(h)
            got = fnv_accel.fnv_chain_blocks(tokens, 32, seed)
            self.assertEqual(got, expected)

    def test_batched_chain_handles_empty_and_partial_tail(self):
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        self.assertEqual(fnv_accel.fnv_chain_blocks([], 32), [])
        self.assertEqual(len(fnv_accel.fnv_chain_blocks([1, 2, 3] * 20, 32)), 1)

    def test_batched_chain_ignores_trailing_partial_block(self):
        """A trailing partial block must not change any completed block's hash.

        The whole token list is packed in one go and the C side floors the byte
        count to whole blocks, so the trailing tokens must not leak into the
        previous blocks' hashes.
        """
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        rng = random.Random(0x7A11)
        for tail in range(1, 32):
            tokens = [rng.randint(0, 32000) for _ in range(32 * 3)]
            seed = rng.getrandbits(64)
            expected = []
            h = seed
            for b in range(3):
                h = fnv1a_64(struct.pack("<32i", *tokens[b*32:(b+1)*32]), h)
                expected.append(h)
            with_tail = tokens + [rng.randint(0, 32000) for _ in range(tail)]
            self.assertEqual(fnv_accel.fnv_chain_blocks(with_tail, 32, seed),
                             expected)

    def test_batched_chain_rejects_out_of_range_token(self):
        """A token outside int32 range must fall back, never silently encode.

        Real token IDs are always small, but if an invalid value slips through
        the batched path must return None so the caller uses hash_token_block,
        which raises rather than producing a hash llama.cpp cannot reproduce.
        """
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        for tokens in ([1] * 32 + [2**40], [1] * 32 + [2**40] + [2] * 31):
            self.assertIsNone(fnv_accel.fnv_chain_blocks(tokens, 32))
            self.assertIsNone(
                fnv_accel.fnv_chain_blocks_with_bytes(tokens, 32)[0])

    def test_batched_chain_is_actually_used(self):
        """The batched path must be live, not silently falling back.

        A bug in the batched helper (e.g. a wrong argument) made it raise and
        return (None, None); the engine then quietly used the per-block
        fallback, so every correctness test still passed while the speedup
        vanished.  Assert the fast path is really taken.
        """
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        import kv_cache.harp_cache as harp_mod

        tokens = [12345] * 64
        hashes, packed = fnv_accel.fnv_chain_blocks_with_bytes(
            tokens, 32, harp_mod.FNV_OFFSET_BASIS)
        self.assertIsNotNone(hashes, "batched chain fell back silently")
        self.assertEqual(len(hashes), 2)
        self.assertIsNotNone(packed)
        self.assertEqual(len(packed), 64 * 4)

        # The engine must use the same path.
        engine = harp_mod.HARPCacheEngine(max_pages=8)
        engine_chain, engine_packed = engine._chain_hashes_and_bytes(
            tokens, engine.fingerprint_seed)
        self.assertIsNotNone(engine_chain,
                             "engine fell back to the per-block hash path")
        self.assertIsNotNone(engine_packed)

    def test_batched_chain_disabled_when_hash_token_block_is_patched(self):
        """HARP collision tests monkeypatch hash_token_block; the batched path
        must fall back to per-block hashing in that case, or the patch is ignored."""
        import kv_cache.harp_cache as harp_mod

        if harp_mod._ORIGINAL_HASH_TOKEN_BLOCK is not harp_mod.hash_token_block:
            self.skipTest("hash_token_block already patched by another test")
        original = harp_mod.hash_token_block
        try:
            self.assertIsNotNone(harp_mod._hash_chain([1] * 64, 32, FNV_OFFSET_BASIS))
            harp_mod.hash_token_block = lambda tokens, prev_hash=FNV_OFFSET_BASIS: 999
            self.assertIsNone(harp_mod._hash_chain([1] * 64, 32, FNV_OFFSET_BASIS))
        finally:
            harp_mod.hash_token_block = original

    def test_batched_chain_collision_rejects_wrong_tokens(self):
        """A same-hash/different-token block must NOT be a cache hit.

        The existing collision tests monkeypatch `hash_token_block`, which
        deliberately disables the batched path — so they never exercise the
        bytes-compare collision check.  Here we patch the batched chain
        itself, leaving `hash_token_block` intact so the native path stays
        live.  Without the bytes comparison this would silently return
        another block's KV data, which is a causal/RoPE soundness violation.
        """
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        import kv_cache.harp_cache as harp_mod

        import kv_cache.fnv_accel as accel_mod

        def constant_chain(tokens, block, seed):
            hashes, packed = real_chain(tokens, block, seed)
            return [999] * len(hashes), packed

        real_chain = accel_mod.fnv_chain_blocks_with_bytes
        original = harp_mod.fnv_chain_blocks_with_bytes
        original_sentinel = harp_mod._ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES
        # Patch the function AND the guard sentinel together: this simulates a
        # genuine 64-bit collision in the native chain, which the guard cannot
        # distinguish from an unchanged primitive.
        harp_mod.fnv_chain_blocks_with_bytes = constant_chain
        harp_mod._ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES = constant_chain
        try:
            engine = harp_mod.HARPCacheEngine(max_pages=64)
            block_a = [1] * 32
            block_b = [2] * 32
            engine.store_prefix(block_a)
            cached, pages, suffix = engine.resolve_prefix(block_b)
            self.assertEqual(cached, 0, "colliding block must not be a hit")
            self.assertEqual(pages, [])
            self.assertEqual(suffix, block_b)
        finally:
            harp_mod.fnv_chain_blocks_with_bytes = original
            harp_mod._ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES = original_sentinel

    def test_batched_chain_collision_eviction_recycles_correct_bytes(self):
        """When a collision evicts a page, the new page must carry its OWN
        token bytes, not the evicted page's."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        import kv_cache.harp_cache as harp_mod

        import kv_cache.fnv_accel as accel_mod

        def constant_chain(tokens, block, seed):
            hashes, packed = real_chain(tokens, block, seed)
            return [999] * len(hashes), packed

        real_chain = accel_mod.fnv_chain_blocks_with_bytes
        original = harp_mod.fnv_chain_blocks_with_bytes
        original_sentinel = harp_mod._ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES
        harp_mod.fnv_chain_blocks_with_bytes = constant_chain
        harp_mod._ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES = constant_chain
        try:
            engine = harp_mod.HARPCacheEngine(max_pages=4)
            block_a = [1] * 32
            block_b = [2] * 32
            engine.store_prefix(block_a + block_a + block_b)
            page = engine.hash_to_page[999]
            self.assertEqual(page.tokens, block_b)
            self.assertEqual(page.token_bytes, harp_mod._pack_tokens(block_b))
            cached, pages, _ = engine.resolve_prefix(block_b)
            self.assertEqual(cached, 32)
            self.assertEqual(pages[0].tokens, block_b)
        finally:
            harp_mod.fnv_chain_blocks_with_bytes = original
            harp_mod._ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES = original_sentinel

    def test_batched_chain_disabled_when_fnv_primitive_is_patched(self):
        """Patching the hash PRIMITIVE must disable the batched path too.

        The original guard only checked `hash_token_block`, so a test or
        caller that patched `fnv1a_64` to force collisions still got real
        hashes from the native chain — the injected collision was silently
        ignored.  All hash primitives are now guarded.
        """
        import kv_cache.harp_cache as harp_mod

        if not harp_mod._hash_primitives_intact():
            self.skipTest("hash primitives already patched by another test")
        original = harp_mod.fnv1a_64
        try:
            harp_mod.fnv1a_64 = lambda data, seed=harp_mod.FNV_OFFSET_BASIS: 999
            tokens = [1] * 64
            self.assertIsNone(harp_mod._hash_chain(tokens, 32,
                                                   harp_mod.FNV_OFFSET_BASIS))
            self.assertEqual(
                harp_mod._hash_chain_with_bytes(tokens, 32,
                                                harp_mod.FNV_OFFSET_BASIS),
                (None, None),
            )
            # And an actual engine must honour the patch.
            engine = harp_mod.HARPCacheEngine(max_pages=8)
            engine.store_prefix(tokens)
            self.assertEqual(list(engine.hash_to_page), [999])
        finally:
            harp_mod.fnv1a_64 = original

    def test_bytes_compare_matches_list_compare(self):
        """The radix walk's bytes-compare path must be semantically identical
        to the original token-list compare path, including after collisions.
        """
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        import kv_cache.harp_cache as harp_mod

        rng = random.Random(0xD1FF)
        for _ in range(30):
            nblocks = rng.randint(1, 20)
            tokens = [rng.randint(0, 32000) for _ in range(32 * nblocks)]

            fast = harp_mod.HARPCacheEngine(max_pages=4096)
            fast.store_prefix(tokens)
            r_fast = fast.resolve_prefix(tokens)

            original = harp_mod.hash_token_block
            original_sentinel = harp_mod._ORIGINAL_HASH_TOKEN_BLOCK
            # Force the per-block fallback path.
            harp_mod._ORIGINAL_HASH_TOKEN_BLOCK = object()
            try:
                slow = harp_mod.HARPCacheEngine(max_pages=4096)
                slow.store_prefix(tokens)
                r_slow = slow.resolve_prefix(tokens)
            finally:
                harp_mod._ORIGINAL_HASH_TOKEN_BLOCK = original_sentinel
                harp_mod.hash_token_block = original

            self.assertEqual(r_fast[0], r_slow[0])
            self.assertEqual(len(r_fast[1]), len(r_slow[1]))
            for h, page in fast.hash_to_page.items():
                self.assertEqual(page.token_bytes,
                                 harp_mod._pack_tokens(page.tokens))

    def test_native_faster_than_python(self):
        """Guard against accidentally falling back to the slow Python loop."""
        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        import time

        data = bytes(range(256)) * 4
        iterations = 2000
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            fnv_accel._fnv1a_python(data)
        t_py = time.perf_counter_ns() - t0
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            fnv1a_64(data)
        t_native = time.perf_counter_ns() - t0
        self.assertLess(t_native, t_py / 3,
                        f"native too slow: {t_native} vs Python {t_py}")

    def test_embedded_null_bytes_hash_full_buffer(self):
        """c_char_p must not truncate the buffer at an embedded NUL."""
        rng = random.Random(0xFEED)
        for _ in range(50):
            data = bytes(rng.getrandbits(8) for _ in range(rng.randint(1, 256)))
            data = data[:len(data)//2] + b"\x00" + data[len(data)//2:]
            seed = rng.getrandbits(64)
            self.assertEqual(fnv1a_64(data, seed), fnv_accel._fnv1a_python(data, seed))

    def test_bytes_like_input(self):
        """bytearray/memoryview should work just like bytes."""
        data = b"a" * 100
        seed = 0xDEADBEEF
        expected = fnv1a_64(data, seed)
        self.assertEqual(fnv1a_64(bytearray(data), seed), expected)
        self.assertEqual(fnv1a_64(memoryview(data), seed), expected)

    def test_no_compiler_falls_back_to_python(self):
        """When no compiler is found, fnv1a_64 must still be bit-identical."""
        from unittest import mock

        with mock.patch.object(fnv_accel.shutil, "which", return_value=None):
            with mock.patch.object(fnv_accel, "_native", None):
                self.assertEqual(fnv1a_64(b"fallback", 12345),
                                 fnv_accel._fnv1a_python(b"fallback", 12345))
                raw = FNV_OFFSET_BASIS.to_bytes(8, "little")
                self.assertEqual(fnv1a_64(raw), fnv_accel._fnv1a_python(raw))

    def test_corrupt_cached_artifact_is_discarded(self):
        """A cached .so that fails the probe must be removed, not trusted."""
        import tempfile
        from pathlib import Path
        from unittest import mock

        with tempfile.TemporaryDirectory() as d:
            bogus = Path(d) / "bogus.so"
            bogus.write_bytes(b"not a shared object")
            with mock.patch.object(fnv_accel, "_candidate_paths", return_value=[bogus]):
                self.assertIsNone(fnv_accel._load_cached())
            # _load_cached() removes the unusable artifact so that the next
            # import rebuilds and replaces it.
            self.assertFalse(bogus.exists())

    def test_stale_version_artifact_is_discarded(self):
        """A cached .so built from older source must not be used."""
        from unittest import mock

        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        if not fnv_accel._LIB_PATH.exists():
            self.skipTest("no cached artifact on this host")

        with mock.patch.object(fnv_accel, "_version_number", return_value=0), \
             mock.patch.object(fnv_accel, "_discard_cached") as discard:
            self.assertIsNone(fnv_accel._load_cached())
            discard.assert_called_once()

    def test_cached_artifact_is_installed_atomically(self):
        """The installed artifact must be a complete, loadable shared object."""
        import ctypes
        from unittest import mock

        if fnv_accel._native is None:
            self.skipTest("no C compiler available on this host")
        if not fnv_accel._LIB_PATH.exists():
            self.skipTest("no cached artifact on this host")

        # No stray temporary files must be left behind by the install.
        leftovers = list(fnv_accel._LIB_PATH.parent.glob(".fnv_accel-*.so"))
        self.assertEqual(leftovers, [],
                         f"stray temp artifacts: {leftovers}")

        # And the artifact on disk must load and hash correctly.
        lib = ctypes.CDLL(str(fnv_accel._LIB_PATH))
        bound = fnv_accel._bind(lib)
        self.assertEqual(bound[2](), fnv_accel._version_number())
        self.assertTrue(fnv_accel._probe(bound))


if __name__ == "__main__":
    unittest.main()
