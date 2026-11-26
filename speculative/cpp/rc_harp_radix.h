#pragma once

// rc_harp_radix.h — Native C++ Chained-Prefix HARP-Cache v2 (llama-modchip v3.2)
// ============================================================================
// Provides:
//  1. Causally-sound chained 64-bit FNV-1a prefix hashing:
//       H_0 = fingerprint_seed
//       H_k = FNV64(H_{k-1} || token_block_k)
//     Guarantees that diverging prefixes [A, C] and [B, C] NEVER collide on block C
//     for both GQA attention (RoPE offset) and Hybrid Mamba-2 / GDN recurrent states.
//  2. O(L / BLOCK) Radix prefix index for multi-branch agentic workloads inside
//     `server_prompt_cache` (true LRU + non-destructive shared-prefix preservation).
//  3. Optional per-UID hardened `/dev/shm` persistent checkpoint mirror
//     (`O_NOFOLLOW`, mode `0600`, atomic temp+rename, model+quant fingerprint check).

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__has_include)
#  if __has_include("llama.h")
#    include "llama.h"
#  else
typedef int32_t llama_token;
#  endif
#else
#  include "llama.h"
#endif

static constexpr uint64_t RC_HARP_FNV_OFFSET = 0xcbf29ce484222325ULL;
static constexpr uint64_t RC_HARP_FNV_PRIME  = 0x100000001b3ULL;
static constexpr size_t   RC_HARP_BLOCK_TOKS = 256; // aligned with --cache-reuse 256 & checkpoints

static inline uint64_t rc_harp_fnv64_bytes(uint64_t seed, const void * data, size_t nbytes) {
    const uint8_t * p = static_cast<const uint8_t *>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < nbytes; ++i) {
        h ^= (uint64_t) p[i];
        h *= RC_HARP_FNV_PRIME;
    }
    return h;
}

static inline uint64_t rc_harp_chain_block(uint64_t prev_hash, const llama_token * tokens, size_t n) {
    uint64_t h = prev_hash;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t t = static_cast<uint32_t>(tokens[i]);
        h ^= (t & 0xFFu);         h *= RC_HARP_FNV_PRIME;
        h ^= ((t >> 8) & 0xFFu);  h *= RC_HARP_FNV_PRIME;
        h ^= ((t >> 16) & 0xFFu); h *= RC_HARP_FNV_PRIME;
        h ^= ((t >> 24) & 0xFFu); h *= RC_HARP_FNV_PRIME;
    }
    return h;
}

struct rc_harp_fingerprint {
    uint64_t hash = RC_HARP_FNV_OFFSET;

    static rc_harp_fingerprint build(
            const std::string & model_id,
            int32_t n_ctx,
            int32_t type_k,
            int32_t type_v) {
        rc_harp_fingerprint fp;
        uint64_t h = RC_HARP_FNV_OFFSET;
        h = rc_harp_fnv64_bytes(h, model_id.data(), model_id.size());
        const int32_t cfg[4] = { n_ctx, type_k, type_v, (int32_t) RC_HARP_BLOCK_TOKS };
        h = rc_harp_fnv64_bytes(h, cfg, sizeof(cfg));
        fp.hash = h;
        return fp;
    }
};

// Computes chained prefix hashes at every RC_HARP_BLOCK_TOKS boundary (256, 512, ...).
// Returns vector of (n_tokens_covered, chained_hash) pairs.
template <typename TokenAccessor>
static inline std::vector<std::pair<size_t, uint64_t>> rc_harp_compute_chain(
        uint64_t seed_hash,
        size_t n_tokens,
        TokenAccessor && get_token) {
    std::vector<std::pair<size_t, uint64_t>> chain;
    const size_t n_blocks = n_tokens / RC_HARP_BLOCK_TOKS;
    if (n_blocks == 0) {
        return chain;
    }
    chain.reserve(n_blocks);
    uint64_t h = seed_hash;
    llama_token buf[RC_HARP_BLOCK_TOKS];
    for (size_t b = 0; b < n_blocks; ++b) {
        const size_t base = b * RC_HARP_BLOCK_TOKS;
        for (size_t j = 0; j < RC_HARP_BLOCK_TOKS; ++j) {
            buf[j] = get_token(base + j);
        }
        h = rc_harp_chain_block(h, buf, RC_HARP_BLOCK_TOKS);
        chain.emplace_back(base + RC_HARP_BLOCK_TOKS, h);
    }
    return chain;
}

// Fast O(L / 256) chained-prefix Radix lookup table mapping chained block hash -> list of entry IDs.
// Always paired with exact token verification by the caller before restoring state.
struct rc_harp_radix_index {
    uint64_t seed_hash = RC_HARP_FNV_OFFSET;
    std::unordered_map<uint64_t, std::vector<uint64_t>> hash_to_ids;
    std::unordered_map<uint64_t, std::vector<uint64_t>> id_to_hashes;

    void set_seed(uint64_t s) {
        if (seed_hash != s) {
            clear();
            seed_hash = s;
        }
    }

    void clear() {
        hash_to_ids.clear();
        id_to_hashes.clear();
    }

    template <typename TokenAccessor>
    void insert(uint64_t entry_id, size_t n_tokens, TokenAccessor && get_token) {
        erase(entry_id);
        auto chain = rc_harp_compute_chain(seed_hash, n_tokens, get_token);
        std::vector<uint64_t> hashes;
        hashes.reserve(chain.size());
        for (const auto & kv : chain) {
            hashes.push_back(kv.second);
            hash_to_ids[kv.second].push_back(entry_id);
        }
        id_to_hashes.emplace(entry_id, std::move(hashes));
    }

    void erase(uint64_t entry_id) {
        auto it = id_to_hashes.find(entry_id);
        if (it == id_to_hashes.end()) {
            return;
        }
        for (uint64_t h : it->second) {
            auto hit = hash_to_ids.find(h);
            if (hit != hash_to_ids.end()) {
                auto & vec = hit->second;
                for (size_t i = 0; i < vec.size(); ) {
                    if (vec[i] == entry_id) {
                        vec[i] = vec.back();
                        vec.pop_back();
                    } else {
                        ++i;
                    }
                }
                if (vec.empty()) {
                    hash_to_ids.erase(hit);
                }
            }
        }
        id_to_hashes.erase(it);
    }

    // Returns candidate entry_ids matching the deepest chained 256-token block prefix.
    template <typename TokenAccessor>
    std::vector<uint64_t> lookup_deepest(size_t n_tokens, TokenAccessor && get_token) const {
        auto chain = rc_harp_compute_chain(seed_hash, n_tokens, get_token);
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            auto hit = hash_to_ids.find(it->second);
            if (hit != hash_to_ids.end() && !hit->second.empty()) {
                return hit->second;
            }
        }
        return {};
    }
};

// Fixed-size 24-byte index record mapping any 256-token chained prefix hash -> persisted leaf snapshot hash.
// Enables post-restart restoration of shared prefixes even when the persisted conversation grew longer than the branch.
struct rc_harp_disk_record {
    uint64_t block_hash;
    uint64_t leaf_hash;
    uint64_t n_tokens;
};

static_assert(sizeof(rc_harp_disk_record) == 24, "rc_harp_disk_record must be 24 bytes");

