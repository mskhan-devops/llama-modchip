#!/usr/bin/env python3
"""
harp_cache.py - Hash-Addressed Radix Page Sharing (HARP-Cache) Engine
Part of the RouteCrack-KV Zero-Prefill Agentic Optimization Suite.

Architectural Role & Footprint Clarification:
---------------------------------------------
1. Gateway-Level Prefix Tracker & Algorithmic Reference:
   This module implements the content-addressable 32-token FNV-1a Radix Tree and POSIX
   shared-memory (`/dev/shm`) LRU page table used by `gateway/sse_gateway.py` to track
   immutable multi-turn agent prefixes (system prompt, 40 tool schemas, conversation history)
   in < 1 ms and compute zero-prefill hit telemetry.
2. Native `llama-server` Tensor Persistence:
   Physical Mamba-2 recurrent states (`conv1d` + SSM state tensors across 23 layers) and
   quantized GQA KV cache tensors (`Q8_0/Q4_0` across 6 layers) are persisted and restored
   natively in C++ by `llama-server --slot-save-path /dev/shm --cache-reuse 256`
   (`POST /slots/0?action=restore`, restoring 22,245 tokens / 85 MB in ~18.9 ms).
   When used standalone (or in `benchmarks/`), `HARPCacheEngine` allocates a bounded
   per-UID `/dev/shm` arena (`max_pages * PAGE_KV_BYTES`) with `O_NOFOLLOW` + `0o600`
   permissions and strict LRU page-ID reuse.
"""

import os
import stat
import sys
import time
import mmap
import struct
import threading
from typing import List, Tuple, Dict, Optional, Any, Set

# FNV-1a 64-bit constants
FNV_OFFSET_BASIS = 0xcbf29ce484222325
FNV_PRIME = 0x100000001b3
MASK_64 = 0xffffffffffffffff


_STRUCT_32I = struct.Struct("<32i")

try:
    from kv_cache.fnv_accel import (
        fnv1a_64 as _FNV_IMPL,
        fnv_chain_blocks,
        fnv_chain_blocks_with_bytes,
    )
except ImportError:  # pragma: no cover - standalone `from harp_cache import ...`
    from fnv_accel import (
        fnv1a_64 as _FNV_IMPL,
        fnv_chain_blocks,
        fnv_chain_blocks_with_bytes,
    )


def _pack_tokens(tokens: List[int]) -> bytes:
    """Little-endian int32 packing used for HARPPage.token_bytes."""
    return struct.Struct(f"<{len(tokens)}i").pack(*tokens)


def _page_matches(page: "HARPPage", chunk: Optional[List[int]],
                  chunk_bytes: Optional[bytes]) -> bool:
    """Compare a cached page against a candidate block.

    When the batched native path supplied the packed buffer, comparing bytes
    avoids materialising a list slice of 32 Python ints for every block.
    """
    if chunk_bytes is not None:
        return page.token_bytes == chunk_bytes
    return page.tokens == chunk


def fnv1a_64(data: bytes, seed: int = FNV_OFFSET_BASIS) -> int:
    """Computes high-speed 64-bit FNV-1a hash over raw byte buffer."""
    # Native C implementation when a compiler is available (~16x faster);
    # bit-identical pure-Python fallback otherwise. See kv_cache/fnv_accel.py.
    return _FNV_IMPL(data, seed)


def hash_token_block(tokens: List[int], prev_hash: int = FNV_OFFSET_BASIS) -> int:
    """
    Hashes a contiguous sequence of token IDs using causally chained 64-bit FNV-1a:
      H_k = FNV64(H_{k-1} || token_block_k).
    When called with `prev_hash`, guarantees diverging prefixes [A, C] and [B, C]
    produce distinct hashes for block C (preserving causal attention & SSM soundness).
    """
    if len(tokens) == 32:
        raw = _STRUCT_32I.pack(*tokens)
    else:
        raw = struct.pack(f"<{len(tokens)}i", *tokens)
    return fnv1a_64(raw, seed=prev_hash)


# The batched native chain is only equivalent to `hash_token_block` while the
# hash primitives are the original implementations.  Tests monkeypatch them to
# force hash collisions, so the batched path must be disabled as soon as any
# of them is replaced: patching only `hash_token_block` would leave the
# batched path live and the injected collision would be silently ignored.
_ORIGINAL_HASH_TOKEN_BLOCK = hash_token_block
_ORIGINAL_FNV1A_64 = fnv1a_64
_ORIGINAL_FNV_CHAIN_BLOCKS = fnv_chain_blocks
_ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES = fnv_chain_blocks_with_bytes


def _hash_primitives_intact() -> bool:
    """True while every hash primitive is the original implementation."""
    return (
        hash_token_block is _ORIGINAL_HASH_TOKEN_BLOCK
        and fnv1a_64 is _ORIGINAL_FNV1A_64
        and fnv_chain_blocks is _ORIGINAL_FNV_CHAIN_BLOCKS
        and fnv_chain_blocks_with_bytes is _ORIGINAL_FNV_CHAIN_BLOCKS_WITH_BYTES
    )


def _hash_chain(tokens: List[int], block: int, seed: int) -> Optional[List[int]]:
    """Chained hashes for every full `block`-sized chunk of `tokens`.

    Returns None when the batched native path is unavailable or would not match
    `hash_token_block`, so callers fall back to the per-block path.
    """
    if not _hash_primitives_intact():
        return None
    return fnv_chain_blocks(tokens, block, seed)


def _hash_chain_with_bytes(
    tokens: List[int], block: int, seed: int
) -> Tuple[Optional[List[int]], Optional[bytes]]:
    """Chained hashes plus the packed little-endian token buffer.

    Returns (None, None) when the batched native path is unavailable or would
    not match `hash_token_block`, so callers fall back to the per-block path.
    """
    if not _hash_primitives_intact():
        return None, None
    return fnv_chain_blocks_with_bytes(tokens, block, seed)


class HARPPage:
    """Metadata descriptor for a single 32-token KV cache memory page."""
    __slots__ = (
        "page_id",
        "page_hash",
        "token_count",
        "tokens",
        "token_bytes",
        "entropy_score",
        "shm_offset",
        "access_count",
        "created_at",
        "last_accessed",
    )

    def __init__(
        self,
        page_id: int,
        page_hash: int,
        tokens: List[int],
        entropy_score: float,
        shm_offset: int,
        token_bytes: Optional[bytes] = None,
    ):
        now = time.time()
        self.page_id = page_id
        self.page_hash = page_hash
        self.token_count = len(tokens)
        self.tokens = tokens
        # Little-endian int32 packing of `tokens`.  Comparing two byte
        # strings is cheaper than comparing two lists of ints, and slicing
        # bytes out of the packed prefix buffer is ~1.9x cheaper than slicing
        # the token list.  The radix walk does this comparison once per block.
        self.token_bytes = token_bytes if token_bytes is not None else _pack_tokens(tokens)
        self.entropy_score = entropy_score
        self.shm_offset = shm_offset
        self.access_count = 1
        self.created_at = now
        self.last_accessed = now


class HARPCacheEngine:
    """
    High-Performance Zero-Copy Radix Page Sharing Engine.
    Manages content-addressable 32-token pages in POSIX shared memory with
    true LRU eviction, physical page-ID free-list reuse, and Radix Tree prefix indexing.
    """
    PAGE_TOKENS = 32                    # 32 tokens per physical memory page
    PAGE_KV_BYTES = 32 * 196608 // 48   # Reference KV footprint per 32-token page (~131 KB)
    MAX_PAGES = 4096                    # 4096 pages = 131,072 tokens capacity (~512 MB shm)

    @staticmethod
    def default_shm_path() -> str:
        euid = getattr(os, "geteuid", getattr(os, "getuid", lambda: 0))()
        return f"/dev/shm/harp_kv_cache_{euid}.bin"

    def __init__(
        self,
        shm_path: Optional[str] = None,
        max_pages: int = MAX_PAGES,
        model_fingerprint: str = "default",
    ):
        self.shm_path = shm_path or self.default_shm_path()
        self.max_pages = max(1, int(max_pages))
        self.total_shm_bytes = self.max_pages * self.PAGE_KV_BYTES
        self.fingerprint_seed = fnv1a_64(model_fingerprint.encode("utf-8"), seed=FNV_OFFSET_BASIS)
        self.lock = threading.Lock()

        # Causally chained content-addressable index: chained_prefix_hash (uint64) -> HARPPage
        self.hash_to_page: Dict[int, HARPPage] = {}

        # Prefix Radix Tree: each node is Dict[chained_hash, {"page": HARPPage, "children": dict}]
        self.radix_root: Dict[int, Dict[str, Any]] = {}
        # Fast parent index: chained_hash -> (parent_hash, parent_dict, entry_dict)
        self._radix_nodes: Dict[int, Tuple[Optional[int], Dict[int, Dict[str, Any]], Dict[str, Any]]] = {}
        # Hashes that occupy MORE THAN ONE depth in the Radix Tree. `_radix_nodes` is
        # keyed by hash alone, so a hash stored at several depths only remembers its
        # deepest parent link; pruning it via the O(1) fast path would leave the outer
        # node still holding a page that was already evicted and whose physical slot has
        # been recycled. Multi-depth hashes are routed through the bottom-up full scan
        # instead, on BOTH the collision path and the LRU eviction path.
        self._multi_depth_hashes: Set[int] = set()

        # Physical page allocator (monotonic watermark + recycled free-list)
        self.allocated_pages = 0
        self.free_page_ids: List[int] = []
        self.evictions = 0
        self._init_shm()

    def _init_shm(self):
        """Initializes per-UID POSIX shared memory backing file safely (O_NOFOLLOW + 0600) with zero-copy mmap."""
        fd = -1
        self.shm_file = None
        try:
            euid = getattr(os, "geteuid", getattr(os, "getuid", lambda: 0))()
            open_flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(self.shm_path, open_flags, 0o600)
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode) or st.st_uid != euid:
                raise PermissionError(f"Unsafe SHM backing file: {self.shm_path}")
            os.fchmod(fd, 0o600)
            if st.st_size != self.total_shm_bytes:
                os.ftruncate(fd, self.total_shm_bytes)

            self.shm_file = os.fdopen(fd, "r+b")
            fd = -1
            self.shm_mmap = mmap.mmap(
                self.shm_file.fileno(),
                self.total_shm_bytes,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            self.shm_active = True
        except Exception:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if self.shm_file is not None and not self.shm_file.closed:
                try:
                    self.shm_file.close()
                except OSError:
                    pass
            # Fallback to bounded anonymous heap buffer if /dev/shm is restricted
            self.shm_mmap = bytearray(self.total_shm_bytes)
            self.shm_active = False

    def close(self):
        """Cleans up memory mappings and file descriptors."""
        with self.lock:
            if hasattr(self, "shm_mmap") and hasattr(self.shm_mmap, "close"):
                self.shm_mmap.close()
            if getattr(self, "shm_file", None) is not None and not self.shm_file.closed:
                self.shm_file.close()

    def calculate_entropy(self, tokens: List[int]) -> float:
        """Estimates token block entropy using RouteCrack zero-FLOP uniqueness ratio."""
        if not tokens:
            return 0.0
        unique_tokens = len(set(tokens))
        return float(unique_tokens) / float(len(tokens))

    def _prune_radix_hash(self, root_node: Dict[int, Dict[str, Any]], target_hash: int, force_full_scan: bool = False) -> None:
        """
        Removes an evicted page_hash from the Radix Tree in O(1) / O(depth) time without
        destroying active descendant subtrees or hitting Python's recursion limit:
        - If `target_hash` is recorded in `_radix_nodes`, directly retrieves (parent_hash, parent_dict, entry).
        - If `entry["children"]` is non-empty, sets `entry["page"] = None` so descendant
          suffix pages remain attached to the Radix Tree.
        - Only deletes from `parent_dict` when `"children"` is empty, propagating pruning upward.
        - Falls back to stack-based bottom-up traversal if target_hash is not in `_radix_nodes`.

        `force_full_scan=True` skips the O(1) branch and falls straight through to the
        existing bottom-up pass, which removes EVERY node keyed by `target_hash` at any
        depth and cleans up ancestors that become empty as a result. This is required for
        the 64-bit hash-collision path: `_radix_nodes` is keyed by hash alone, so when the
        same hash occurs at several depths only the deepest parent link is recorded.
        The O(1) fast path would prune just that one node and leave an outer node still
        holding a page that was already evicted and whose physical slot has been recycled.
        """
        if not force_full_scan and root_node is self.radix_root and target_hash in self._radix_nodes:
            parent_h, parent_dict, entry = self._radix_nodes[target_hash]
            if entry.get("children"):
                entry["page"] = None
                return

            self._radix_nodes.pop(target_hash, None)
            if target_hash in parent_dict:
                del parent_dict[target_hash]

            curr_h = parent_h
            while curr_h is not None and curr_h in self._radix_nodes:
                gp_h, p_dict, p_entry = self._radix_nodes[curr_h]
                if p_entry.get("page") is None and not p_entry.get("children"):
                    self._radix_nodes.pop(curr_h, None)
                    if curr_h in p_dict:
                        del p_dict[curr_h]
                    curr_h = gp_h
                else:
                    break
            return

        # Stack-based bottom-up traversal: removes the target at EVERY depth and
        # cleans up any ancestor that becomes empty. Forced for multi-depth hashes
        # (collision or LRU eviction) so no stale page reference survives.
        stack: List[Tuple[Dict[int, Dict[str, Any]], bool]] = [(root_node, False)]
        while stack:
            node, visited_children = stack.pop()
            if not visited_children:
                stack.append((node, True))
                for child_entry in node.values():
                    children = child_entry.get("children")
                    if children:
                        stack.append((children, False))
            else:
                for child_hash, child_entry in list(node.items()):
                    children = child_entry.get("children")
                    if child_hash != target_hash and child_entry.get("page") is None and not children:
                        del node[child_hash]
                        self._radix_nodes.pop(child_hash, None)
                        self._multi_depth_hashes.discard(child_hash)
                if target_hash in node:
                    if node[target_hash].get("children"):
                        node[target_hash]["page"] = None
                    else:
                        del node[target_hash]
                        self._radix_nodes.pop(target_hash, None)
                        self._multi_depth_hashes.discard(target_hash)

        self._multi_depth_hashes.discard(target_hash)

    def _acquire_physical_page_id(self) -> int:
        """
        Returns a valid physical page_id in [0, self.max_pages - 1].
        Uses free_page_ids first, then monotonic allocation up to max_pages,
        and finally true LRU eviction (reusing the evicted page's physical page_id).
        """
        if self.free_page_ids:
            return self.free_page_ids.pop()

        if self.allocated_pages < self.max_pages:
            pid = self.allocated_pages
            self.allocated_pages += 1
            return pid

        # Arena is full: evict Least Recently Used (LRU) page, tie-breaking by access_count
        lru_page = min(
            self.hash_to_page.values(),
            key=lambda p: (p.last_accessed, p.access_count),
        )
        lru_hash = lru_page.page_hash
        # Multi-depth hashes need the full bottom-up scan: `_radix_nodes` only tracks
        # the deepest parent, so the O(1) path would leave outer nodes holding the
        # evicted page while its physical slot is recycled for another block.
        self._prune_radix_hash(
            self.radix_root, lru_hash,
            force_full_scan=lru_hash in self._multi_depth_hashes,
        )
        evicted_page = self.hash_to_page.pop(lru_hash)
        self.evictions += 1
        return evicted_page.page_id

    def _chain_hashes(self, token_stream: List[int], seed: int) -> Optional[List[int]]:
        """Chained FNV-1a hashes for every full PAGE_TOKENS block of `token_stream`.

        Returns None when the native batched implementation is unavailable, in
        which case callers fall back to the per-block `hash_token_block` path.
        """
        hashes = _hash_chain(token_stream, self.PAGE_TOKENS, seed)
        if hashes is None or len(hashes) != len(token_stream) // self.PAGE_TOKENS:
            return None
        return hashes

    def _chain_hashes_and_bytes(
        self, token_stream: List[int], seed: int
    ) -> Tuple[Optional[List[int]], Optional[bytes]]:
        """Chained hashes plus the packed little-endian token buffer.

        The packed buffer lets the radix walk slice each block's bytes out of
        one contiguous buffer instead of copying a slice out of the token list,
        which is ~1.9x cheaper per block.
        """
        hashes, packed = _hash_chain_with_bytes(
            token_stream, self.PAGE_TOKENS, seed)
        if hashes is None or len(hashes) != len(token_stream) // self.PAGE_TOKENS:
            return None, None
        return hashes, packed

    def resolve_prefix(self, token_stream: List[int]) -> Tuple[int, List[HARPPage], List[int]]:
        """
        Causally Sound Longest Common Prefix (LCP) Resolver.
        Walks the Prefix Radix Tree using chained rolling hashes H_k = FNV64(H_{k-1} || B_k)
        seeded by `model_fingerprint`. Diverging prefixes [A, C] and [B, C] never collide on C.
        Returns:
            (cached_tokens_count, matched_pages, remaining_suffix_tokens)
        """
        matched_pages: List[HARPPage] = []
        n_tokens = len(token_stream)
        curr_offset = 0
        now = time.time()

        with self.lock:
            radix_cursor: Optional[Dict[int, Dict[str, Any]]] = self.radix_root
            rolling_hash = self.fingerprint_seed
            # Hash the whole chain in one native call when possible; the walk
            # below still stops at the first block that does not match.
            chain, packed = self._chain_hashes_and_bytes(token_stream, rolling_hash)
            block_bytes = self.PAGE_TOKENS * 4
            block_index = 0
            while curr_offset + self.PAGE_TOKENS <= n_tokens:
                chunk: Optional[List[int]] = None
                chunk_bytes: Optional[bytes] = None
                if chain is not None:
                    h = chain[block_index]
                    # Slice the block's packed bytes out of the contiguous
                    # buffer; comparing bytes is cheaper than comparing the
                    # token lists element by element.
                    chunk_bytes = packed[curr_offset * 4:
                                         curr_offset * 4 + block_bytes]
                else:
                    chunk = token_stream[curr_offset : curr_offset + self.PAGE_TOKENS]
                    rolling_hash = hash_token_block(chunk, prev_hash=rolling_hash)
                    h = rolling_hash
                block_index += 1

                page: Optional[HARPPage] = None
                if radix_cursor is not None and h in radix_cursor:
                    cand = radix_cursor[h].get("page")
                    if cand is not None and _page_matches(cand, chunk, chunk_bytes):
                        page = cand
                        radix_cursor = radix_cursor[h]["children"]
                    else:
                        radix_cursor = None

                # O(1) chained-hash fallback is causally sound ONLY because `h` commits to
                # the entire preceding prefix [B_0 .. B_k] + model fingerprint.
                if page is None:
                    radix_cursor = None
                    cand = self.hash_to_page.get(h)
                    if cand is not None and _page_matches(cand, chunk, chunk_bytes):
                        page = cand

                if page is not None:
                    page.access_count += 1
                    page.last_accessed = now
                    matched_pages.append(page)
                    curr_offset += self.PAGE_TOKENS
                else:
                    break

        cached_count = len(matched_pages) * self.PAGE_TOKENS
        remaining_suffix = token_stream[cached_count:]
        return cached_count, matched_pages, remaining_suffix

    def store_prefix(self, token_stream: List[int], simulated_kv_data: Optional[bytes] = None) -> int:
        """
        Chunks and stores an immutable prefix into HARP-Cache in shared memory using
        causally chained hashes H_k = FNV64(H_{k-1} || B_k), updating both the
        chained-hash table and the Prefix Radix Tree.
        Returns number of newly stored 32-token pages.
        """
        new_pages = 0
        n_tokens = len(token_stream)
        curr_offset = 0
        now = time.time()

        with self.lock:
            radix_cursor = self.radix_root
            parent_hash: Optional[int] = None
            rolling_hash = self.fingerprint_seed
            # Hash the whole chain in one native call when possible.
            chain, packed = self._chain_hashes_and_bytes(token_stream, rolling_hash)
            block_bytes = self.PAGE_TOKENS * 4
            block_index = 0
            while curr_offset + self.PAGE_TOKENS <= n_tokens:
                chunk: Optional[List[int]] = None
                chunk_bytes: Optional[bytes] = None
                if chain is not None:
                    h = chain[block_index]
                    # The block's packed bytes come straight out of the
                    # contiguous buffer; the token list is only materialised
                    # when a new page has to be built.
                    chunk_bytes = packed[curr_offset * 4:
                                         curr_offset * 4 + block_bytes]
                else:
                    chunk = token_stream[curr_offset : curr_offset + self.PAGE_TOKENS]
                    rolling_hash = hash_token_block(chunk, prev_hash=rolling_hash)
                    h = rolling_hash
                block_index += 1

                existing = self.hash_to_page.get(h)
                if existing is not None and _page_matches(existing, chunk, chunk_bytes):
                    existing.last_accessed = now
                    existing.access_count += 1
                    page = existing
                else:
                    if existing is not None:
                        # 64-bit hash collision with distinct token block: evict & recycle colliding entry cleanly.
                        # The same hash can already exist at several depths of the Radix Tree, and
                        # `_radix_nodes` only tracks one parent link per hash, so the O(1) fast
                        # path would leave an outer node holding the evicted (and already
                        # recycled) page. Prune every occurrence instead.
                        self._prune_radix_hash(self.radix_root, h, force_full_scan=True)
                        colliding_page = self.hash_to_page.pop(h)
                        self.free_page_ids.append(colliding_page.page_id)
                        self.evictions += 1

                    page_id = self._acquire_physical_page_id()
                    shm_offset = page_id * self.PAGE_KV_BYTES
                    if chunk is None:
                        chunk = token_stream[curr_offset : curr_offset + self.PAGE_TOKENS]
                    entropy = self.calculate_entropy(chunk)

                    page = HARPPage(
                        page_id=page_id,
                        page_hash=h,
                        tokens=chunk,
                        entropy_score=entropy,
                        shm_offset=shm_offset,
                        token_bytes=chunk_bytes,
                    )

                    # Write page payload or header descriptor to bounded /dev/shm slot
                    if simulated_kv_data and len(simulated_kv_data) >= self.PAGE_KV_BYTES:
                        self.shm_mmap[shm_offset : shm_offset + self.PAGE_KV_BYTES] = simulated_kv_data[:self.PAGE_KV_BYTES]
                    else:
                        self.shm_mmap[shm_offset : shm_offset + 32] = struct.pack(
                            "QQQQ", h, len(chunk), int(entropy * 1000), page_id
                        )

                    self.hash_to_page[h] = page
                    new_pages += 1

                if h not in radix_cursor:
                    entry = {"page": page, "children": {}}
                    radix_cursor[h] = entry
                else:
                    entry = radix_cursor[h]
                    entry["page"] = page
                if h in self._radix_nodes and self._radix_nodes[h][1] is not radix_cursor:
                    # Same chained hash reached at a different depth: `_radix_nodes`
                    # can only remember one parent, so future evictions of `h` must
                    # use the full bottom-up scan instead of the O(1) fast path.
                    self._multi_depth_hashes.add(h)
                self._radix_nodes[h] = (parent_hash, radix_cursor, entry)
                parent_hash = h
                radix_cursor = entry["children"]

                curr_offset += self.PAGE_TOKENS

        return new_pages

    def stats(self) -> Dict[str, Any]:
        """Returns engine runtime diagnostics."""
        with self.lock:
            cached_pages = len(self.hash_to_page)
            total_cached_tokens = cached_pages * self.PAGE_TOKENS
            total_shm_mb = (cached_pages * self.PAGE_KV_BYTES) / (1024 * 1024)
            avg_entropy = sum(p.entropy_score for p in self.hash_to_page.values()) / max(1, cached_pages)
            return {
                "active_pages": cached_pages,
                "max_pages": self.max_pages,
                "allocated_pages": self.allocated_pages,
                "evictions": self.evictions,
                "radix_root_branches": len(self.radix_root),
                "cached_tokens": total_cached_tokens,
                "shm_footprint_mb": total_shm_mb,
                "avg_entropy": avg_entropy,
                "shm_active": self.shm_active,
                "shm_path": self.shm_path,
            }


if __name__ == "__main__":
    print("[*] HARP-Cache Engine Module compiled successfully.")
