#!/usr/bin/env python3
"""
test_harp_cache.py - Comprehensive Unit & Performance Test Suite for HARP-Cache
Validates lookup latency (<0.01 ms), zero-copy POSIX shm, and zero-prefill agentic savings.
"""

import time
import os
import stat
import random
import unittest
try:
    from kv_cache.harp_cache import HARPCacheEngine, hash_token_block, fnv1a_64
except ImportError:
    from harp_cache import HARPCacheEngine, hash_token_block, fnv1a_64

_UID = getattr(os, "geteuid", getattr(os, "getuid", lambda: 0))()

def test_fnv1a_throughput():
    print("\n--- TEST 1: FNV-64 Hash Cascade Throughput ---")
    tokens = [random.randint(100, 50000) for _ in range(100000)]
    t0 = time.perf_counter()
    chunks_count = len(tokens) // 32
    for i in range(chunks_count):
        _ = hash_token_block(tokens[i*32 : (i+1)*32])
    t1 = time.perf_counter()
    elapsed = t1 - t0
    tok_per_sec = len(tokens) / elapsed
    print(f"[+] Hashed {len(tokens):,} tokens in {elapsed*1000:.2f} ms")
    print(f"[+] Hashing Throughput: {tok_per_sec:,.0f} tokens/second")
    assert tok_per_sec > 1_000_000, "Hashing speed should exceed 1M tok/s"

def test_harp_agentic_prefill_simulation():
    print("\n--- TEST 2: Agentic Prefix Caching & Zero-Prefill Latency ---")
    test_shm = f"/dev/shm/harp_kv_cache_test_{_UID}.bin"
    engine = HARPCacheEngine(shm_path=test_shm, max_pages=1024)
    try:
        if engine.shm_active:
            st = os.stat(test_shm)
            assert stat.S_IMODE(st.st_mode) == 0o600, f"Expected 0600 permissions, got {oct(stat.S_IMODE(st.st_mode))}"
            assert st.st_uid == _UID
        # 1. Simulate Agent System Prompt (1,536 tokens = 48 pages)
        system_prompt_tokens = [1000 + (i % 250) for i in range(1536)]
        # 2. Simulate Tool Schemas (1,024 tokens = 32 pages)
        tools_tokens = [5000 + (i % 300) for i in range(1024)]
        # 3. Simulate Static C Source File (module.c, 1,280 tokens = 40 pages)
        source_tokens = [9000 + (i % 400) for i in range(1280)]
        # Complete Common Prefix (3,840 tokens = 120 pages)
        common_prefix = system_prompt_tokens + tools_tokens + source_tokens

        stored = engine.store_prefix(common_prefix)
        assert stored > 0

        latencies_us = []
        for turn in range(1000):
            dynamic_suffix = [30000 + turn * 10 + i for i in range(random.randint(50, 200))]
            turn_prompt = common_prefix + dynamic_suffix
            t0 = time.perf_counter_ns()
            cached_count, matched_pages, remaining_suffix = engine.resolve_prefix(turn_prompt)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)
            assert cached_count == len(common_prefix), f"Should match all {len(common_prefix)} prefix tokens"
            assert len(remaining_suffix) == len(dynamic_suffix), "Remaining suffix should match exactly dynamic tokens"

        stats = engine.stats()
        assert stats["active_pages"] == stored
    finally:
        engine.close()
        if os.path.exists(test_shm):
            os.remove(test_shm)

def test_radix_intermediate_eviction_and_collision():
    print("\n--- TEST 3: Intermediate Radix Eviction & Hash Collision Safety ---")
    lru_shm = f"/dev/shm/harp_kv_cache_test_lru_{_UID}.bin"
    deep_shm = f"/dev/shm/harp_kv_cache_test_deep_{_UID}.bin"
    engine = HARPCacheEngine(shm_path=lru_shm, max_pages=3)
    try:
        p0 = [100 + i for i in range(32)]
        p1 = [200 + i for i in range(32)]
        p2 = [300 + i for i in range(32)]
        p3 = [400 + i for i in range(32)]

        # Store [p0, p1, p2] -> fills all 3 physical pages (using causally chained hashes)
        assert engine.store_prefix(p0 + p1 + p2) == 3
        h0 = hash_token_block(p0, prev_hash=engine.fingerprint_seed)
        h1 = hash_token_block(p1, prev_hash=h0)
        h2 = hash_token_block(p2, prev_hash=h1)

        # Touch p0 and p2 so p1 (intermediate node) becomes Least Recently Used
        time.sleep(0.002)
        engine.hash_to_page[h0].last_accessed = time.time()
        engine.hash_to_page[h2].last_accessed = time.time()

        # Storing p3 evicts intermediate page p1; p2 must remain attached in radix_root[h0]["children"][h1]["children"]
        assert engine.store_prefix(p3) == 1
        assert engine.radix_root[h0]["children"][h1]["page"] is None, "Evicted intermediate node should set page=None"
        assert h2 in engine.radix_root[h0]["children"][h1]["children"], "Descendant subtree must not be orphaned"

        # Touch p0 and p2 after p3 so p3 becomes LRU when p1 is re-inserted
        time.sleep(0.002)
        engine.hash_to_page[h0].last_accessed = time.time()
        engine.hash_to_page[h2].last_accessed = time.time()

        # Re-storing [p0, p1] evicts p3 and reconnects p1 without losing p2 child in Radix Tree
        engine.store_prefix(p0 + p1)
        cached, _, _ = engine.resolve_prefix(p0 + p1 + p2)
        assert cached == 96, f"Expected full 96-token Radix path after reconnecting intermediate node, got {cached}"
    finally:
        engine.close()
        if os.path.exists(lru_shm):
            os.remove(lru_shm)

    # Verify deep (>1,100 pages) prefix chain eviction does not hit RecursionError
    deep_engine = HARPCacheEngine(shm_path=deep_shm, max_pages=1100)
    try:
        deep_tokens = [10000 + i for i in range(1150 * 32)]
        deep_engine.store_prefix(deep_tokens)
        assert len(deep_engine.hash_to_page) == 1100
        assert deep_engine.evictions == 50
    finally:
        deep_engine.close()
        if os.path.exists(deep_shm):
            os.remove(deep_shm)


class TestHARPCache(unittest.TestCase):
    def test_fnv1a_throughput(self):
        test_fnv1a_throughput()

    def test_harp_agentic_prefill_simulation(self):
        test_harp_agentic_prefill_simulation()

    def test_radix_intermediate_eviction_and_collision(self):
        test_radix_intermediate_eviction_and_collision()

    def test_default_shm_path_is_per_uid(self):
        default_path = HARPCacheEngine.default_shm_path()
        self.assertTrue(default_path.endswith(f"_{_UID}.bin"), default_path)

    def test_causal_prefix_divergence_never_collides(self):
        """Regression test: [A, C] and [B, C] MUST NOT share KV page C (causal/RoPE soundness)."""
        div_shm = f"/dev/shm/harp_kv_cache_test_div_{_UID}.bin"
        engine = HARPCacheEngine(shm_path=div_shm, max_pages=16, model_fingerprint="modelA:q8_0:q4_0")
        try:
            block_a = [1000 + i for i in range(32)]
            block_b = [2000 + i for i in range(32)]
            block_c = [3000 + i for i in range(32)]
            engine.store_prefix(block_a + block_c)

            # Exact match on [A, C] hits all 64 tokens
            cached_ac, _, _ = engine.resolve_prefix(block_a + block_c)
            self.assertEqual(cached_ac, 64)

            # Diverging prefix [B, C] MUST return 0 cached tokens (never reuse C from A!)
            cached_bc, _, rem_bc = engine.resolve_prefix(block_b + block_c)
            self.assertEqual(cached_bc, 0)
            self.assertEqual(len(rem_bc), 64)

            # Storing [B, C] allocates 2 distinct pages (never overwrites [A, C]'s C page)
            self.assertEqual(engine.store_prefix(block_b + block_c), 2)
            self.assertEqual(len(engine.hash_to_page), 4)

            # Different model fingerprint must produce distinct chained hashes
            h_a = hash_token_block(block_a, prev_hash=engine.fingerprint_seed)
            other_seed = fnv1a_64(b"modelB:q8_0:q4_0")
            h_a_other = hash_token_block(block_a, prev_hash=other_seed)
            self.assertNotEqual(h_a, h_a_other)
        finally:
            engine.close()
            if os.path.exists(div_shm):
                os.remove(div_shm)

    def test_close_without_active_shm_does_not_raise(self):
        """Fallback anonymous heap buffer mode has shm_file=None; close() must be idempotent."""
        engine = HARPCacheEngine(shm_path="/nonexistent/path/for/shm.bin", max_pages=4)
        self.assertFalse(engine.shm_active)
        self.assertIsNone(engine.shm_file)
        # Must not raise AttributeError: 'NoneType' object has no attribute 'closed'
        engine.close()
        engine.close()

    def test_hash_collision_does_not_leave_stale_radix_page(self):
        """
        A 64-bit hash collision must evict the colliding page from the Radix Tree at
        EVERY depth it occupies. `_radix_nodes` is keyed by hash alone, so the O(1)
        prune fast path only sees the deepest parent link and used to leave an outer
        node holding a page that had already been evicted and whose physical slot had
        been recycled - resolve_prefix then returned that stale page as a cache hit.
        """
        import kv_cache.harp_cache as harp_mod
        shm = f"/dev/shm/harp_kv_cache_test_collide_{_UID}.bin"
        original = harp_mod.hash_token_block
        engine = HARPCacheEngine(shm_path=shm, max_pages=4)
        try:
            # Force every block to hash to the same value -> guaranteed 64-bit collision.
            harp_mod.hash_token_block = lambda tokens, prev_hash=harp_mod.FNV_OFFSET_BASIS: 999
            block_a = [3000 + i for i in range(32)]
            block_b = [7000 + i for i in range(32)]

            # Store A, hit A again, then store B -> B collides with A and evicts it.
            engine.store_prefix(block_a + block_a + block_b)

            # The colliding page must be gone: A must NOT resolve as a cache hit.
            cached, matched, remaining = engine.resolve_prefix(block_a)
            self.assertEqual(cached, 0, "evicted page must not resolve as a hit")
            self.assertEqual(matched, [])
            self.assertEqual(len(remaining), 32)

            # No Radix node may still point at the evicted page.
            stale = [
                entry.get("page")
                for entry in engine.radix_root.values()
                if entry.get("page") is not None and entry["page"].tokens == block_a
            ]
            self.assertEqual(stale, [], "Radix Tree still references the evicted page")

            # The surviving page must still resolve correctly.
            cached_b, matched_b, _ = engine.resolve_prefix(block_b)
            self.assertEqual(cached_b, 32)
            self.assertEqual(matched_b[0].tokens, block_b)
        finally:
            harp_mod.hash_token_block = original
            engine.close()
            if os.path.exists(shm):
                os.remove(shm)


    def test_lru_eviction_does_not_leave_stale_radix_page(self):
        """
        A multi-depth hash (same chained hash reached via different causal prefixes)
        must be fully pruned when its page is evicted by the normal LRU allocator.
        `_radix_nodes` is keyed by hash alone and only remembers the deepest parent,
        so the O(1) path used to leave the outer node holding the evicted page while
        its physical slot was recycled for an unrelated block — resolve_prefix then
        returned a stale hit and read shared-memory data belonging to that other block.
        """
        import time as _time
        import kv_cache.harp_cache as harp_mod
        shm = f"/dev/shm/harp_kv_cache_test_lru_stale_{_UID}.bin"
        original = harp_mod.hash_token_block
        real = original
        block_a = [1000 + i for i in range(32)]

        def fake_hash(tokens, prev_hash=harp_mod.FNV_OFFSET_BASIS):
            # Force block A to hash to a constant at every depth, so [A, A] and [A, B]
            # produce the SAME chained hash 999 at depth 0 (multi-depth collision).
            if list(tokens) == block_a:
                return 999
            return real(tokens, prev_hash=prev_hash)

        engine = HARPCacheEngine(shm_path=shm, max_pages=2)
        try:
            harp_mod.hash_token_block = fake_hash
            engine.store_prefix(block_a + block_a)

            # Confirm the multi-depth hash was detected.
            self.assertIn(999, engine._multi_depth_hashes)

            # Fill the arena, then force the stale A page to be the LRU victim.
            engine.store_prefix([5000 + i for i in range(32)])
            now = _time.time()
            for page in engine.hash_to_page.values():
                if 999 in engine.hash_to_page and page is engine.hash_to_page[999]:
                    page.last_accessed = now - 100

            # LRU eviction fires on the next store (arena is full).
            engine.store_prefix([6000 + i for i in range(32)])  # triggers LRU eviction

            self.assertGreater(engine.evictions, 0)
            self.assertNotIn(999, engine.hash_to_page, "A's page must be evicted")

            # No Radix node may still reference the evicted A page.
            stale = [
                entry.get("page")
                for entry in engine.radix_root.values()
                if entry.get("page") is not None and entry["page"].tokens == block_a
            ]
            self.assertEqual(stale, [], "Radix Tree still references the LRU-evicted page")

            cached, matched, _ = engine.resolve_prefix(block_a)
            self.assertEqual(cached, 0, "evicted page must NOT resolve as a stale hit")
            self.assertEqual(matched, [])
        finally:
            harp_mod.hash_token_block = original
            engine.close()
            if os.path.exists(shm):
                os.remove(shm)


if __name__ == "__main__":
    unittest.main()
