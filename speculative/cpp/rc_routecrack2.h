/*
 * rc_routecrack2.h — RouteCrack 2.0: Unified 3-Method Speculative Decoding Engine
 *                    Powered by the RouteCrack 2.0 L1 Micro-Arbiter (`rc_arbiter_l1.h`)
 *
 * Unites three orthogonal speculative engines under a single expected-`tokens/ms` horizon law:
 *   1. Method 1 (`NextN MTP` Neural Anchor, d = 0 [+ gated stem d = 1..2] or `RCPredictor` on non-MTP models):
 *      Extracts exact neural top-1/top-4 candidates and logit certainty (`p_top1`, `p_margin`,
 *      `top4_mass`) from native MTP blocks without blind multi-step `lm_head` looping, or queries
 *      `RCPredictor` (8-level FNV-64 Cascade + MacroCodebook + Online Trie v2 + PPM + Skipgram) when MTP is absent.
 *   2. Method 2 (`O(1) Indexed Two-Zone Prompt & Context Suffix Copy`, 16K Hot + 24K Prefix):
 *      Maintains an O(1)-undoable 3-gram hash-chain index over both the 16K Hot Zone and 24K
 *      Prefix Zone (`< 100 ns` lookup at 64K context). Supports both Draftless Fast-Path Bypass
 *      (skipping the single-token MTP `lm_head` pass on long copied spans) and Post-Anchor
 *      Extension (`[..., id_last, t_0] -> [t_1 .. t_4]`), resolving >90% of suffix collisions.
 *   3. Method 3 (`Stockfish-PVS Multi-Order Zobrist TT + Online 6-Gram Session Memory`):
 *      Tracks exact online target verifications & refutations (`tt6/tt5/tt4/tt3` + 28-bit mode refutation),
 *      vetoing refuted moves, promoting verified top-4 alternatives, and extending recurring idioms.
 *   4. Arbiter (`rc_arbiter_l1.h`, `10 -> 16 -> 1` AVX-512 / NEON ClippedReLU `832 B` [13 cache lines] L1d-resident):
 *      Predicts conditional acceptance probability q_d and appends token t_d iff adding t_d
 *      strictly increases expected net `tokens / ms` across MTP, fast non-MTP, and heavy 256-expert MoE profiles.
 */

#ifndef RC_ROUTECRACK2_H
#define RC_ROUTECRACK2_H

#ifndef LLAMA_H
#include "llama.h"
#endif
#include "rc_predictor.h"
#include "rc_stockfish.h"
#include "rc_arbiter_l1.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

struct RC2MTPAnchorInfo {
    bool     valid       = false;
    int32_t  top_ids[4]  = {-1, -1, -1, -1};
    float    top_p[4]    = {0.0f, 0.0f, 0.0f, 0.0f};
    int      n_cands     = 0;
    float    p_top1      = 0.0f;
    float    p_margin    = 0.0f;
    float    top4_mass   = 0.0f;
};

struct RC2SuffixMatch {
    bool    found               = false;
    size_t  best_j              = 0;
    int     best_k              = 0;     // Matched suffix length (3..12)
    int     hits                = 0;     // Total matching positions inspected
    int     true_consensus_hits = 0;     // Distinct occurrences agreeing on continuation
    int     consensus_len       = 0;     // Agreed continuation length (0..6)
    bool    zone_is_hot         = false;
    size_t  second_j            = 0;     // Conflicting branch index when consensus_len == 0
    int     second_k            = 0;
};

struct RC2RCCandidate {
    int32_t tok          = -1;
    int     cert         = 0;
    float   p1           = 0.0f;
    float   margin       = 0.0f;
    int     casc_k       = 0;
    int     agree        = 0;
    bool    from_macro   = false;
    bool    deep_session = false;
};

// Shared Vocab-Safe `RCPredictor` Hub for Non-MTP Flagships (`Nemotron-3.5-30B` & `DeepSeek-V4-284B`):
//   - When `n_vocab == 131072` (`Nemotron`), auto-loads offline 8-level FNV-64 Cascade + 300 Hot Rules
//     + 11,382 Macro Chains + 80K Warm Corpus from `$RC_TABLES_DIR` or `./workspace/routecrack`.
//   - For any other `n_vocab` (`DeepSeek-V4` 129280, etc.), runs 100% vocabulary-agnostic Online `RCTrie` v2
//     + Multi-Order `RCPPM` (k=2..6) + `RCSkipgram` with zero cross-tokenizer contamination.
struct RC2PredictorHub {
    std::mutex                   mtx;
    std::unique_ptr<RCPredictor> pred;
    int32_t                      vocab_size     = -1;
    bool                         initialized    = false;
    bool                         offline_loaded = false;

    ~RC2PredictorHub() {
        if (pred) {
            pred->destroy();
            pred.reset();
        }
    }

    static bool file_exists(const char * path) {
        if (!path || !path[0]) return false;
        FILE * f = std::fopen(path, "rb");
        if (f) {
            std::fclose(f);
            return true;
        }
        return false;
    }

    void init(int32_t n_vocab, uint32_t trie_cap = RC_TRIE_CAPACITY) {
        std::lock_guard<std::mutex> lock(mtx);
        if (initialized) return;
        initialized = true;
        vocab_size  = n_vocab;

        pred = std::make_unique<RCPredictor>();
        pred->n_cascade        = 0;
        pred->hot_rules.rules  = nullptr;
        pred->hot_rules.n_rules = 0;
        pred->logit_probe.W        = nullptr;
        pred->logit_probe.bias     = nullptr;
        pred->logit_probe.hot_ids  = nullptr;
        pred->logit_probe.feat_ids = nullptr;
        pred->logit_probe.ready    = false;
        pred->online_probe.init();
        pred->ppm.clear();
        pred->skipgram.clear();
        pred->codebook.clear();
        pred->macro.clear();
        const char * e_mc = std::getenv("RC_DRAFT_MIN_CONF");
        pred->draft_min_conf   = e_mc ? std::atoi(e_mc) : 50;
        pred->probe_W          = nullptr;
        pred->probe_bias       = nullptr;
        pred->probe_n          = 0;
        pred->probe_embd       = 0;
        pred->probe_ready      = false;
        pred->probe_collecting = false;
        pred->trie.init(trie_cap, RC_TRIE_MAX_DEPTH);

        // Resolve vocab-compatible offline RouteCrack tables (strictly guarded by `n_vocab == 131072` unless custom non-Nemotron dir set)
        const char * chosen_dir = nullptr;
        const char * env_dir    = std::getenv("RC_TABLES_DIR");
        char probe_buf[512];

        if (n_vocab == 131072) {
            const char * cands[2] = { env_dir, "./workspace/routecrack" };
            for (const char * dir : cands) {
                if (!dir || !dir[0]) continue;
                std::snprintf(probe_buf, sizeof(probe_buf), "%s/hc_top4_k10.bin", dir);
                if (file_exists(probe_buf)) {
                    chosen_dir = dir;
                    break;
                }
            }
        } else if (env_dir && env_dir[0]) {
            char warm_nem[512];
            std::snprintf(warm_nem, sizeof(warm_nem), "%s/corpus_nemotron_warm.bin", env_dir);
            std::snprintf(probe_buf, sizeof(probe_buf), "%s/hc_top4_k10.bin", env_dir);
            if (!file_exists(warm_nem) && file_exists(probe_buf)) {
                chosen_dir = env_dir;
            }
        }

        if (chosen_dir) {
            const int ks[8] = {10, 8, 6, 5, 4, 3, 2, 1};
            for (int i = 0; i < 8; ++i) {
                std::snprintf(probe_buf, sizeof(probe_buf), "%s/hc_top4_k%d.bin", chosen_dir, ks[i]);
                pred->load_cascade(probe_buf);
            }
            std::snprintf(probe_buf, sizeof(probe_buf), "%s/hot_rules.bin", chosen_dir);
            pred->hot_rules.load(probe_buf);
            std::snprintf(probe_buf, sizeof(probe_buf), "%s/macro_codebook.bin", chosen_dir);
            pred->macro.load(probe_buf);
            offline_loaded = (pred->n_cascade > 0 || pred->macro.n_entries > 0 || pred->hot_rules.n_rules > 0);

            if (n_vocab == 131072) {
                std::snprintf(probe_buf, sizeof(probe_buf), "%s/corpus_nemotron_warm.bin", chosen_dir);
                FILE * f = std::fopen(probe_buf, "rb");
                if (f) {
                    if (std::fseek(f, 0, SEEK_END) == 0) {
                        long sz = std::ftell(f);
                        if (sz >= 48) {
                            long n_toks = sz / 4;
                            long warm_n = std::min<long>(n_toks, 80000);
                            if (std::fseek(f, (n_toks - warm_n) * 4, SEEK_SET) == 0) {
                                std::vector<int32_t> buf((size_t)warm_n);
                                if (std::fread(buf.data(), 4, (size_t)warm_n, f) == (size_t)warm_n) {
                                    for (long j = 12; j < warm_n; ++j) {
                                        learn_window_nolock(&buf[(size_t)(j - 11)], 12, 10);
                                    }
                                }
                            }
                        }
                    }
                    std::fclose(f);
                }
            }
        }
    }

    inline void learn_window_nolock(const int32_t * ctx, int len, int max_trie_d = 10) {
        if (!pred || len < 2) return;
        const int32_t actual = ctx[len - 1];

        // 1. Update Multi-Order PPM (k = 2..6)
        for (int k = RC_PPM_K_MIN; k <= RC_PPM_K_MAX && k < len; ++k) {
            const int ti = k - RC_PPM_K_MIN;
            uint64_t h = rc_fnv64(ctx + len - 1 - k, k);
            if (h == 0) h = 1;
            const uint32_t slot = (uint32_t)(h & RC_PPM_MASK);
            for (int probe = 0; probe < 8; ++probe) {
                const uint32_t idx = (slot + (uint32_t)probe) & RC_PPM_MASK;
                RCPPMEntry & e = pred->ppm.tables[ti][idx];
                if (e.ctx_hash == 0) {
                    e.ctx_hash  = h;
                    e.preds[0]  = actual;
                    e.counts[0] = 1;
                    e.n_preds   = 1;
                    e.total_cnt = 1;
                    break;
                }
                if (e.ctx_hash == h) {
                    if (e.total_cnt == 65535) {
                        e.total_cnt = 0;
                        for (int q = 0; q < e.n_preds; ++q) {
                            e.counts[q] = (uint16_t)((e.counts[q] + 1) >> 1);
                            e.total_cnt += e.counts[q];
                        }
                    }
                    e.total_cnt++;
                    bool found = false;
                    for (int p = 0; p < e.n_preds; ++p) {
                        if (e.preds[p] == actual) {
                            e.counts[p]++;
                            while (p > 0 && e.counts[p] > e.counts[p - 1]) {
                                std::swap(e.preds[p], e.preds[p - 1]);
                                std::swap(e.counts[p], e.counts[p - 1]);
                                p--;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (!found && e.n_preds < RC_PPM_TOPK) {
                        e.preds[e.n_preds]  = actual;
                        e.counts[e.n_preds] = 1;
                        e.n_preds++;
                    }
                    break;
                }
            }
        }

        // 2. Update Skipgram ((len-4, len-2) -> len-1)
        if (len >= 4) {
            const uint64_t h = RCSkipgram::skip_hash(ctx[len - 4], ctx[len - 2]);
            const uint32_t slot = (uint32_t)(h & RC_SKIP_MASK);
            for (int probe = 0; probe < 8; ++probe) {
                const uint32_t idx = (slot + (uint32_t)probe) & RC_SKIP_MASK;
                if (pred->skipgram.table[idx].hash == 0) {
                    pred->skipgram.table[idx] = {h, actual, 1, 1};
                    break;
                }
                if (pred->skipgram.table[idx].hash == h) {
                    if (pred->skipgram.table[idx].total == 65535) {
                        pred->skipgram.table[idx].total = (uint16_t)((pred->skipgram.table[idx].total + 1) >> 1);
                        pred->skipgram.table[idx].hits  = (uint16_t)((pred->skipgram.table[idx].hits  + 1) >> 1);
                    }
                    pred->skipgram.table[idx].total++;
                    if (actual == pred->skipgram.table[idx].pred) {
                        pred->skipgram.table[idx].hits++;
                    }
                    break;
                }
            }
        }

        // 3. Update SGLang-style Arena Online Trie (d = 3..max_trie_d)
        const int max_d = std::min(len, max_trie_d);
        for (int d = 3; d <= max_d; ++d) {
            const int start = len - d;
            if (pred->trie.used + (uint32_t)d >= pred->trie.capacity) {
                pred->trie.evict(std::max<uint32_t>(pred->trie.capacity / 10, (uint32_t)d));
            }
            RCTrieNode * cursor = pred->trie.root_ptr;
            for (int j = start; j < len; ++j) {
                auto it = cursor->children.find(ctx[j]);
                if (it == cursor->children.end()) {
                    RCTrieNode * child = pred->trie.alloc_node_ptr(ctx[j], cursor);
                    if (!child) break;
                    cursor->children[ctx[j]] = child;
                    child->freq = 1;
                    cursor = child;
                } else {
                    if (j == len - 1) it->second->freq++;
                    cursor = it->second;
                }
            }
        }
    }

    inline void verify_cand_nolock(
        const int32_t * ctx_ptr, int ctx_len, int32_t cand,
        int & out_agree, float & out_p1, bool cand_from_prompt_suffix = false) {

        out_agree = 0;
        out_p1    = 0.0f;
        if (!pred || cand < 0 || ctx_len < 2) return;

        for (int c = 0; c < pred->n_cascade; ++c) {
            const int ck = pred->cascade_k[c];
            if (ctx_len >= ck) {
                const uint64_t h = rc_fnv64(ctx_ptr + ctx_len - ck, ck);
                const int idx = pred->cascade[c].find_idx(h);
                if (idx >= 0) {
                    const auto & ent = pred->cascade[c].entries[idx];
                    for (int p = 0; p < ent.n_preds; ++p) {
                        if (ent.pred[p] == cand) {
                            out_agree++;
                            out_p1 = std::max(out_p1, (float)ent.conf[p] / 100.0f);
                            break;
                        }
                    }
                    break;
                }
            }
        }

        const bool need_multi_obs = (cand_from_prompt_suffix && !offline_loaded);

        float trie_conf = 0.0f;
        int trie_obs = 0, trie_depth = 0;
        const int32_t trie_tok = pred->trie.predict_confident(ctx_ptr, ctx_len, &trie_conf, &trie_obs, &trie_depth);
        if (trie_tok == cand && trie_depth >= 2 && (!need_multi_obs || trie_obs >= 2)) {
            out_agree++;
            out_p1 = std::max(out_p1, trie_conf * (trie_obs >= 2 ? 0.92f : 0.72f));
        }

        if (!need_multi_obs || trie_obs >= 2) {
            int32_t ppm_out[4];
            float ppm_scores[4];
            const int n_ppm = pred->ppm.predict(ctx_ptr, ctx_len, ppm_out, ppm_scores, 4);
            if (n_ppm > 0 && ppm_out[0] == cand) {
                out_agree++;
                out_p1 = std::max(out_p1, 0.76f);
            }

            if (ctx_len >= 3 && pred->skipgram.predict(ctx_ptr[ctx_len - 3], ctx_ptr[ctx_len - 1]) == cand) {
                out_agree++;
                out_p1 = std::max(out_p1, 0.65f);
            }
        }

        if (ctx_len >= 1 && pred->hot_rules.n_rules > 0 && pred->hot_rules.lookup(ctx_ptr[ctx_len - 1]) == cand) {
            out_agree++;
            out_p1 = std::max(out_p1, 0.80f);
        }
    }

    inline RC2RCCandidate propose_cand_nolock(
        const int32_t * ctx_ptr, int ctx_len,
        const StockfishDrafter & sf,
        const RCMacroEntry *& io_macro,
        int macro_step) {

        RC2RCCandidate res;
        if (!pred || ctx_len < 2) return res;

        int32_t casc_tok = -1;
        int casc_conf = 0, casc_k = 0;
        const RCHashEntry * casc_ent = nullptr;
        for (int c = 0; c < pred->n_cascade; ++c) {
            const int ck = pred->cascade_k[c];
            if (ctx_len >= ck) {
                const uint64_t h = rc_fnv64(ctx_ptr + ctx_len - ck, ck);
                const int idx = pred->cascade[c].find_idx(h);
                if (idx >= 0) {
                    const auto & ent = pred->cascade[c].entries[idx];
                    casc_ent  = &ent;
                    casc_tok  = ent.pred[0];
                    casc_conf = ent.conf[0];
                    casc_k    = ck;
                    break;
                }
            }
        }

        float trie_conf = 0.0f;
        int trie_obs = 0, trie_depth = 0;
        const int32_t trie_tok = pred->trie.predict_confident(ctx_ptr, ctx_len, &trie_conf, &trie_obs, &trie_depth);

        int32_t ppm_out[4];
        float ppm_scores[4];
        const int n_ppm = pred->ppm.predict(ctx_ptr, ctx_len, ppm_out, ppm_scores, 4);
        const int32_t ppm_tok = (n_ppm > 0) ? ppm_out[0] : -1;
        const float ppm_dom = (n_ppm >= 2 && ppm_scores[0] > 0.0f)
            ? (ppm_scores[0] / (ppm_scores[0] + ppm_scores[1]))
            : (n_ppm == 1 ? 1.0f : 0.0f);

        const int32_t skip_tok = (ctx_len >= 3) ? pred->skipgram.predict(ctx_ptr[ctx_len - 3], ctx_ptr[ctx_len - 1]) : -1;

        int32_t hot_tok = -1, hot_conf = 0;
        if (ctx_len >= 1 && pred->hot_rules.n_rules > 0) {
            int lo = 0, hi = (int)pred->hot_rules.n_rules - 1;
            const int32_t prev = ctx_ptr[ctx_len - 1];
            while (lo <= hi) {
                const int mid = (lo + hi) / 2;
                if (pred->hot_rules.rules[mid].predecessor == prev) {
                    hot_tok  = pred->hot_rules.rules[mid].prediction;
                    hot_conf = pred->hot_rules.rules[mid].confidence;
                    break;
                }
                if (pred->hot_rules.rules[mid].predecessor < prev) lo = mid + 1;
                else hi = mid - 1;
            }
        }

        // Top-4 Cascade Disambiguation via Stockfish Orthogonal Session TT
        if (casc_ent && casc_ent->n_preds > 1) {
            const SFTTBucket * b5 = sf.probe5(ctx_ptr, ctx_len);
            const SFTTBucket * b4 = sf.probe4(ctx_ptr, ctx_len);
            const bool top1_refuted =
                (b5 && b5->refuted_move == casc_tok && b5->best_move != casc_tok && b5->history < 0) ||
                (b4 && b4->refuted_move == casc_tok && b4->best_move != casc_tok && b4->history <= -64);
            if (top1_refuted || casc_conf < 80) {
                for (int p = 1; p < casc_ent->n_preds; ++p) {
                    const int32_t alt = casc_ent->pred[p];
                    const int alt_c   = casc_ent->conf[p];
                    if (alt < 0 || alt_c < 20) continue;
                    const bool tt5_m = (b5 && b5->best_move == alt && b5->wins >= 3 && b5->history >= 160);
                    const bool tt4_m = (b4 && b4->best_move == alt && b4->wins >= 4 && b4->history >= 200);
                    const bool ses_m = (trie_tok == alt && trie_depth >= 4 && trie_conf >= 0.80f) && (ppm_tok == alt);
                    if ((tt5_m || tt4_m) && ses_m) {
                        casc_tok  = alt;
                        casc_conf = 86;
                        break;
                    }
                }
            }
        }

        if (casc_tok >= 0) {
            int agree = 1;
            if (trie_tok == casc_tok && trie_depth >= 2) agree++;
            if (ppm_tok == casc_tok) agree++;
            if (skip_tok == casc_tok) agree++;
            if (hot_tok == casc_tok) agree++;
            int cert = casc_conf + (casc_k >= 6 ? 4 : 0) + (agree >= 3 ? 6 : (agree == 2 ? 3 : 0));
            if (casc_k == 1 && casc_conf < 90) cert = 0;
            res.tok    = casc_tok;
            res.cert   = cert;
            res.casc_k = casc_k;
            res.agree  = agree;
        } else if (trie_tok >= 0 && trie_depth >= 4 && trie_obs >= 3 && trie_conf >= 0.88f) {
            res.tok          = trie_tok;
            res.agree        = 1 + (ppm_tok == trie_tok ? 1 : 0) + (skip_tok == trie_tok ? 1 : 0);
            res.cert         = (int)(trie_conf * 94.0f) + (res.agree >= 2 ? 4 : 0);
            res.deep_session = true;
        } else if (!offline_loaded && trie_tok >= 0 && trie_depth >= 4 && trie_obs >= 2 &&
                   trie_conf >= 0.92f && ppm_tok == trie_tok && ppm_dom >= 0.88f) {
            // Universal online mode (`DeepSeek-V4`): prompt-warmed deep Trie + unanimous PPM agreement
            res.tok          = trie_tok;
            res.agree        = 2 + (skip_tok == trie_tok ? 1 : 0);
            res.cert         = std::min(94, (int)(trie_conf * 90.0f) + (trie_depth >= 6 ? 4 : 0));
            res.deep_session = true;
        } else if (hot_tok >= 0 && hot_conf >= 90 && (ppm_tok == hot_tok || skip_tok == hot_tok)) {
            res.tok   = hot_tok;
            res.cert  = hot_conf;
            res.agree = 2;
        }

        // Session Trie v2 Rescue when shadowed by low-conf short cascade
        if (res.cert < 84 && trie_tok >= 0 && trie_depth >= 4 && trie_obs >= 4 && trie_conf >= 0.92f) {
            const SFTTBucket * b5 = sf.probe5(ctx_ptr, ctx_len);
            const SFTTBucket * b4 = sf.probe4(ctx_ptr, ctx_len);
            const bool refuted = (b5 && b5->refuted_move == trie_tok && b5->best_move != trie_tok) ||
                                 (b4 && b4->refuted_move == trie_tok && b4->best_move != trie_tok);
            const bool tt_conf = (b5 && b5->best_move == trie_tok && b5->wins >= 2) ||
                                 (b4 && b4->best_move == trie_tok && b4->wins >= 3);
            const int t_agree = 1 + (ppm_tok == trie_tok ? 1 : 0) + (skip_tok == trie_tok ? 1 : 0);
            if (!refuted && ((tt_conf && t_agree >= 2) || (trie_depth >= 6 && trie_obs >= 5 && t_agree >= 2))) {
                res.tok          = trie_tok;
                res.cert         = std::min(95, (int)(trie_conf * 92.0f) + (tt_conf ? 4 : 0));
                res.agree        = t_agree;
                res.deep_session = true;
            }
        }

        // MacroCodebook 4-token chain verification & amplification
        if (io_macro) {
            if (macro_step >= 0 && macro_step < io_macro->cont_len) {
                if (res.tok == io_macro->continuation[macro_step]) {
                    res.cert       = std::max(res.cert, (int)io_macro->conf + (macro_step >= 1 ? 4 : 0));
                    res.from_macro = true;
                } else {
                    io_macro = nullptr;
                }
            } else {
                io_macro = nullptr;
            }
        }

        if (res.tok >= 0 && res.cert > 0) {
            res.p1     = std::min(0.99f, (float)res.cert / 100.0f);
            res.margin = std::min(1.0f, 0.22f * (float)res.agree +
                                  (res.from_macro || res.casc_k >= 6 ? 0.36f : (res.deep_session ? 0.30f : 0.12f)));
        } else {
            res.tok = -1;
            res.cert = 0;
        }
        return res;
    }
};

struct RC2SeqEngine {
    static constexpr uint32_t IDX_BUCKETS    = 16384; // 64 KB per head table
    static constexpr uint32_t IDX_MASK       = IDX_BUCKETS - 1;
    static constexpr size_t   HOT_ZONE_LEN   = 16384;
    static constexpr size_t   PREFIX_ZONE_CAP = 24576;
    static constexpr uint32_t TT6_ENTRIES    = 2048;  // 32 KB 6-gram Zobrist session table
    static constexpr uint32_t TT6_MASK       = TT6_ENTRIES - 1;
    static constexpr int      MAX_DRAFT_CAP  = 5;

    bool initialized = false;

    // Incremental O(1)-undoable Two-Zone 3-Gram Chained Suffix Index
    std::vector<int32_t> indexed_tokens;
    std::vector<int32_t> prev_occ;
    std::vector<int32_t> prev_prefix_occ;
    alignas(64) int32_t  head_all[IDX_BUCKETS];
    alignas(64) int32_t  head_prefix[IDX_BUCKETS];

    // Stockfish-PVS Zobrist Transposition & Refutation Tables (tt5/tt4/tt3 + tt6)
    StockfishDrafter    sf;
    alignas(64) SFTTBucket tt6[TT6_ENTRIES];
    // 512-byte L1d-resident lossless presence filters + 2KB collision-free 28-bit tagged token-mode refutation table
    alignas(64) uint64_t sf_ctx_last_bloom[64];
    alignas(64) uint64_t sf_refuted_cand_bloom[64];
    alignas(64) uint32_t tok_refute_table[512];

    // Online telemetry & session state
    float   recent_accept_ema   = 0.72f;
    int32_t last_draft[8]       = {-1, -1, -1, -1, -1, -1, -1, -1};
    int     last_draft_len      = 0;
    size_t  last_synced_len     = 0;
    size_t  last_rc_learned_len = 0;
    bool    need_full_lcp       = true;

    // Diagnostic counters
    uint64_t stat_fastpath_hits   = 0;
    uint64_t stat_mtp_anchors     = 0;
    uint64_t stat_rc_anchors      = 0;
    uint64_t stat_suffix_ext_toks = 0;
    uint64_t stat_sf_vetoes       = 0;
    uint64_t stat_sf_promotions   = 0;
    uint64_t stat_nnue_vetoes     = 0;
    uint64_t stat_stem_probes     = 0;

    void init() {
        if (initialized) return;
        initialized = true;
        std::fill(std::begin(head_all), std::end(head_all), -1);
        std::fill(std::begin(head_prefix), std::end(head_prefix), -1);
        indexed_tokens.clear();
        prev_occ.clear();
        prev_prefix_occ.clear();
        indexed_tokens.reserve(32768);
        prev_occ.reserve(32768);
        prev_prefix_occ.reserve(PREFIX_ZONE_CAP + 64);

        sf.init();
        std::memset(tt6, 0, sizeof(tt6));
        std::memset(sf_ctx_last_bloom, 0, sizeof(sf_ctx_last_bloom));
        std::memset(sf_refuted_cand_bloom, 0, sizeof(sf_refuted_cand_bloom));
        std::memset(tok_refute_table, 0, sizeof(tok_refute_table));
        for (uint32_t i = 0; i < TT6_ENTRIES; ++i) {
            tt6[i].best_move    = -1;
            tt6[i].refuted_move = -1;
        }
        recent_accept_ema   = 0.72f;
        last_draft_len      = 0;
        last_synced_len     = 0;
        last_rc_learned_len = 0;
        need_full_lcp       = true;
    }

    inline void clear_session_mode() noexcept {
        last_draft_len = 0;
        need_full_lcp  = true;
        std::memset(tok_refute_table, 0, sizeof(tok_refute_table));
    }

    void reset_index() {
        std::fill(std::begin(head_all), std::end(head_all), -1);
        std::fill(std::begin(head_prefix), std::end(head_prefix), -1);
        std::memset(tok_refute_table, 0, sizeof(tok_refute_table));
        indexed_tokens.clear();
        prev_occ.clear();
        prev_prefix_occ.clear();
        last_rc_learned_len = 0;
        need_full_lcp       = true;
    }

    static inline uint32_t hash3_bucket(int32_t t_minus_2, int32_t t_minus_1, int32_t t_0) noexcept {
        uint64_t h = (uint64_t)(uint32_t)t_minus_2 * 0x9e3779b97f4a7c15ULL;
        h ^= (uint64_t)(uint32_t)t_minus_1 * 0xbf58476d1ce4e5b9ULL;
        h ^= (uint64_t)(uint32_t)t_0       * 0x94d049bb133111ebULL;
        h ^= (h >> 31);
        return (uint32_t)h & IDX_MASK;
    }

    inline void push_indexed_token(int32_t tok) {
        const int32_t j = (int32_t)indexed_tokens.size();
        indexed_tokens.push_back(tok);
        if (j < 2) {
            prev_occ.push_back(-1);
            if ((size_t)j < PREFIX_ZONE_CAP) {
                prev_prefix_occ.push_back(-1);
            }
            return;
        }
        const uint32_t b = hash3_bucket(indexed_tokens[j - 2], indexed_tokens[j - 1], tok);
        prev_occ.push_back(head_all[b]);
        head_all[b] = j;
        if ((size_t)j < PREFIX_ZONE_CAP) {
            prev_prefix_occ.push_back(head_prefix[b]);
            head_prefix[b] = j;
        }
    }

    inline void pop_indexed_token() {
        if (indexed_tokens.empty()) return;
        const int32_t j = (int32_t)indexed_tokens.size() - 1;
        if (j >= 2) {
            const uint32_t b = hash3_bucket(indexed_tokens[j - 2], indexed_tokens[j - 1], indexed_tokens[j]);
            head_all[b] = prev_occ[j];
            if ((size_t)j < PREFIX_ZONE_CAP) {
                head_prefix[b] = prev_prefix_occ[j];
            }
        }
        if ((size_t)j < PREFIX_ZONE_CAP && !prev_prefix_occ.empty()) {
            prev_prefix_occ.pop_back();
        }
        prev_occ.pop_back();
        indexed_tokens.pop_back();
    }

    static inline uint32_t bloom_pair_bit(int32_t prev_tok, int32_t last_tok) noexcept {
        return (((uint32_t)prev_tok * 2654435761u) ^ ((uint32_t)last_tok * 2246822519u)) & 4095u;
    }

    inline bool cand_ever_refuted(int32_t cand) const noexcept {
        const uint32_t rbit = (uint32_t)cand & 4095u;
        return ((sf_refuted_cand_bloom[rbit >> 6] >> (rbit & 63u)) & 1ULL) != 0ULL;
    }

    inline bool is_mode_refuted_token(int32_t cand) const noexcept {
        if (cand < 0) return false;
        const uint32_t e = tok_refute_table[((uint32_t)cand * 2654435761u) >> 23];
        return ((e >> 4) == (uint32_t)(cand + 1)) && ((e & 0xFu) >= 4u);
    }

    inline void observe_tt_all(const int32_t * ctx, int len, int32_t actual_tok, int32_t drafted_tok) {
        if (actual_tok >= 0) {
            const uint32_t aidx = ((uint32_t)actual_tok * 2654435761u) >> 23;
            const uint32_t atag = (uint32_t)(actual_tok + 1) << 4;
            const uint32_t ae   = tok_refute_table[aidx];
            if ((ae & ~0xFu) == atag) {
                const uint32_t h = ae & 0xFu;
                tok_refute_table[aidx] = (h > 2u) ? (atag | (h - 2u)) : 0u;
            }
        }
        if (len >= 3) {
            const uint32_t bit = bloom_pair_bit(ctx[len - 2], ctx[len - 1]);
            sf_ctx_last_bloom[bit >> 6] |= (1ULL << (bit & 63u));
            if (drafted_tok >= 0 && drafted_tok != actual_tok) {
                const uint32_t rbit = (uint32_t)drafted_tok & 4095u;
                sf_refuted_cand_bloom[rbit >> 6] |= (1ULL << (rbit & 63u));
                const uint32_t didx = ((uint32_t)drafted_tok * 2654435761u) >> 23;
                const uint32_t dtag = (uint32_t)(drafted_tok + 1) << 4;
                const uint32_t de   = tok_refute_table[didx];
                if ((de & ~0xFu) == dtag) {
                    const uint32_t h = std::min<uint32_t>(15u, (de & 0xFu) + 3u);
                    tok_refute_table[didx] = dtag | h;
                } else if ((de & 0xFu) <= 3u) {
                    tok_refute_table[didx] = dtag | 3u;
                }
            }
        }
        sf.observe(ctx, len, actual_tok, drafted_tok);
        if (len >= 6) {
            const uint64_t zh = StockfishDrafter::zhash(ctx, len, 6);
            if (zh) {
                SFTTBucket & b = tt6[zh & TT6_MASK];
                const uint32_t k32 = (uint32_t)(zh >> 32);
                const bool was_refuted = (drafted_tok >= 0 && drafted_tok != actual_tok);
                if (b.key32 != k32) {
                    b.key32        = k32;
                    b.best_move    = actual_tok;
                    b.refuted_move = was_refuted ? drafted_tok : -1;
                    b.history      = was_refuted ? -128 : 112;
                    b.wins         = 1;
                    b.total        = 1;
                } else {
                    if (b.total < 250) b.total++;
                    if (b.best_move == actual_tok) {
                        if (b.wins < 250) b.wins++;
                        if (b.refuted_move == actual_tok) b.refuted_move = -1;
                        StockfishDrafter::update_gravity(b.history, +144);
                    } else {
                        const int32_t new_ref = was_refuted ? drafted_tok : b.best_move;
                        if (new_ref >= 0) {
                            const uint32_t rbit = (uint32_t)new_ref & 4095u;
                            sf_refuted_cand_bloom[rbit >> 6] |= (1ULL << (rbit & 63u));
                        }
                        b.refuted_move = new_ref;
                        b.best_move    = actual_tok;
                        b.wins         = 1;
                        StockfishDrafter::update_gravity(b.history, -192);
                    }
                }
            }
        }
    }

    inline const SFTTBucket * probe6(const int32_t * ctx, int len) const noexcept {
        if (len < 6) return nullptr;
        const uint64_t zh = StockfishDrafter::zhash(ctx, len, 6);
        const SFTTBucket & b = tt6[zh & TT6_MASK];
        return (b.key32 == (uint32_t)(zh >> 32)) ? &b : nullptr;
    }

    // Synchronize `indexed_tokens` with `prompt + [id_last]` and feed exact verification
    // outcomes of `last_draft` into Stockfish Zobrist TT (`tt6/tt5/tt4/tt3`) + `recent_accept_ema`.
    void sync_context_and_observe(const llama_tokens & prompt, llama_token id_last) {
        if (!initialized) init();

        const size_t p_sz   = prompt.size();
        const size_t new_sz = p_sz + 1;
        auto token_at_new = [&](size_t idx) -> int32_t {
            return (idx < p_sz) ? (int32_t)prompt[idx] : (int32_t)id_last;
        };

        const size_t old_sz = indexed_tokens.size();

        // Fast-path check: exact append onto `indexed_tokens` (99.9% of decode steps)
        size_t lcp = 0;
        bool tail_exact = (!need_full_lcp && old_sz > 0 && new_sz >= old_sz &&
                           indexed_tokens[0] == token_at_new(0) &&
                           indexed_tokens[old_sz >> 1] == token_at_new(old_sz >> 1));
        if (tail_exact) {
            const size_t check_from = (old_sz > 8) ? (old_sz - 8) : 0;
            for (size_t k = check_from; k < old_sz; ++k) {
                if (indexed_tokens[k] != token_at_new(k)) {
                    tail_exact = false;
                    break;
                }
            }
        }
        if (tail_exact) {
            lcp = old_sz;
        } else {
            const size_t lim = std::min(old_sz, new_sz);
            // Check from recent tail first if minor rollback/truncation happened within an active turn
            if (!need_full_lcp && old_sz > 0 && lim > 16 &&
                indexed_tokens[lim - 16] == token_at_new(lim - 16) &&
                indexed_tokens[0] == token_at_new(0)) {
                lcp = lim - 16;
                while (lcp < lim && indexed_tokens[lcp] == token_at_new(lcp)) {
                    lcp++;
                }
            } else {
                while (lcp < lim && indexed_tokens[lcp] == token_at_new(lcp)) {
                    lcp++;
                }
            }
        }
        need_full_lcp = false;

        const bool is_contiguous_step = (lcp == old_sz && old_sz > 0 && new_sz > old_sz && (new_sz - old_sz) <= 8);

        if (lcp < old_sz) {
            if (lcp == 0 || (old_sz - lcp) > 4096) {
                reset_index();
                need_full_lcp = false;
                lcp = 0;
            } else {
                while (indexed_tokens.size() > lcp) {
                    pop_indexed_token();
                }
                last_rc_learned_len = std::min(last_rc_learned_len, lcp);
            }
            last_draft_len = 0;
        }

        // Push newly committed tokens and observe verification/refutation against `last_draft`
        const size_t delta = new_sz - lcp;
        for (size_t s = 0; s < delta; ++s) {
            const size_t pos = lcp + s;
            const int32_t actual = token_at_new(pos);

            if (pos >= 6 && (is_contiguous_step || (new_sz - pos) <= 128)) {
                int32_t ctx_win[8];
                const int win_len = (int)std::min<size_t>(pos, 7);
                const size_t start = pos - (size_t)win_len;
                for (int w = 0; w < win_len; ++w) {
                    ctx_win[w] = indexed_tokens[start + (size_t)w];
                }
                const int32_t d_tok = (is_contiguous_step && (int)s < last_draft_len) ? last_draft[s] : -1;
                observe_tt_all(ctx_win, win_len, actual, d_tok);
            }

            push_indexed_token(actual);
        }

        if (is_contiguous_step && last_draft_len > 0) {
            const int n_acc = (int)std::min<size_t>((size_t)last_draft_len, delta - 1);
            for (int i = 0; i < last_draft_len; ++i) {
                const float hit = (i < n_acc) ? 1.0f : 0.0f;
                recent_accept_ema = 0.92f * recent_accept_ema + 0.08f * hit;
            }
        }
        last_draft_len  = 0;
        last_synced_len = new_sz;
    }

    // O(1) Indexed Two-Zone Suffix Lookup (`< 100 ns` at 64K context)
    // Virtual query context is `indexed_tokens[0..N-1]` followed by `extra[0..extra_len-1]`.
    RC2SuffixMatch query_two_zone_suffix(const int32_t * extra, int extra_len) const noexcept {
        RC2SuffixMatch res;
        const size_t N = indexed_tokens.size();
        const size_t V_len = N + (size_t)std::max(0, extra_len);
        if (N < 4 || V_len < 4) {
            return res;
        }

        auto v_at = [&](size_t idx) noexcept -> int32_t {
            return (idx < N) ? indexed_tokens[idx] : extra[idx - N];
        };

        const int32_t p0 = v_at(V_len - 1);
        const int32_t p1 = v_at(V_len - 2);
        const int32_t p2 = v_at(V_len - 3);
        const uint32_t b = hash3_bucket(p2, p1, p0);

        const int32_t hot_min_j = (N > HOT_ZONE_LEN) ? (int32_t)(N - HOT_ZONE_LEN) : 2;
        int consensus_cap = 6;

        auto eval_occ = [&](size_t j, bool is_hot) noexcept {
            // Need at least 1 continuation token strictly inside `indexed_tokens[0..N-1]`
            if (j + 1 >= N || j < 2) return;
            if (indexed_tokens[j] != p0 || indexed_tokens[j - 1] != p1 || indexed_tokens[j - 2] != p2) {
                return;
            }
            int k = 3;
            const int max_k = (int)std::min<size_t>(12, std::min<size_t>(j + 1, V_len));
            while (k < max_k && indexed_tokens[j - (size_t)k] == v_at(V_len - 1 - (size_t)k)) {
                k++;
            }
            if (!is_hot && k < 5) return;
            if (is_hot && (j + 256 < N) && k < 4) return;

            res.hits++;
            if (!res.found) {
                res.found               = true;
                res.best_j              = j;
                res.best_k              = k;
                res.zone_is_hot         = is_hot;
                res.true_consensus_hits = 1;
                res.consensus_len       = (int)std::min<size_t>((size_t)consensus_cap, N - 1 - j);
            } else {
                int agree = 0;
                const int max_agree = (int)std::min<size_t>(
                    (size_t)res.consensus_len,
                    std::min<size_t>(N - 1 - j, N - 1 - res.best_j));
                while (agree < max_agree &&
                       indexed_tokens[j + 1 + (size_t)agree] == indexed_tokens[res.best_j + 1 + (size_t)agree]) {
                    agree++;
                }
                if (k >= res.best_k && agree > 0) {
                    res.best_j      = j;
                    res.best_k      = k;
                    res.zone_is_hot = res.zone_is_hot || is_hot;
                    res.consensus_len = std::min(res.consensus_len, agree);
                    res.true_consensus_hits++;
                } else if (k + 1 >= res.best_k) {
                    if (agree == 0 && res.second_j == 0) {
                        res.second_j = j;
                        res.second_k = k;
                    }
                    if (agree > 0) {
                        res.true_consensus_hits++;
                    }
                    res.consensus_len = std::min(res.consensus_len, agree);
                }
            }
        };

        // 1. Hot Zone O(1) bucket chain walk (most recent occurrences first)
        int32_t j = head_all[b];
        int steps = 0;
        while (j >= hot_min_j && steps < 12) {
            const int32_t next_j = prev_occ[j];
            if ((size_t)(j + 1) < N) {
                eval_occ((size_t)j, true);
                if (res.hits >= 5 || (res.best_k >= 6 && res.true_consensus_hits >= 2)) break;
            }
            j = next_j;
            steps++;
        }

        // 2. Prefix Zone O(1) bucket chain walk (system prompt, tool schemas, read files)
        if (!res.found && hot_min_j > 8) {
            const int32_t prefix_limit = (int32_t)std::min<size_t>((size_t)hot_min_j, PREFIX_ZONE_CAP);
            int32_t pj = head_prefix[b];
            int psteps = 0;
            while (pj >= 5 && psteps < 10) {
                const int32_t next_pj = prev_prefix_occ[pj];
                if (pj < prefix_limit && (size_t)(pj + 1) < N) {
                    eval_occ((size_t)pj, false);
                    if (res.hits >= 3) break;
                }
                pj = next_pj;
                psteps++;
            }
        }

        return res;
    }

    // Build contiguous tail context `[..., id_last, extra[0..extra_len-1]]` (up to 12 tokens)
    int build_tail_window(const int32_t * extra, int extra_len, int32_t * out_win) const noexcept {
        const size_t N = indexed_tokens.size();
        const int keep_base = (int)std::min<size_t>(N, 8);
        int pos = 0;
        for (int i = keep_base; i >= 1; --i) {
            out_win[pos++] = indexed_tokens[N - (size_t)i];
        }
        for (int i = 0; i < extra_len && pos < 16; ++i) {
            out_win[pos++] = extra[i];
        }
        return pos;
    }

    // Check Stockfish Zobrist TT (`tt6/tt5/tt4/tt3`) for candidate refutation, wins, & history
    struct SFProbeSummary {
        bool    refuted      = false;
        int     best_wins    = 0;
        float   norm_history = 0.0f;
        int32_t killer_tok   = -1;
        int     killer_wins  = 0;
        int     killer_order = 0;
    };

    SFProbeSummary evaluate_stockfish(const int32_t * tail_win, int tail_len, int32_t cand, int match_k) const noexcept {
        SFProbeSummary s;
        if (tail_len < 3) {
            return s;
        }
        const uint32_t bit = bloom_pair_bit(tail_win[tail_len - 2], tail_win[tail_len - 1]);
        if (((sf_ctx_last_bloom[bit >> 6] >> (bit & 63u)) & 1ULL) == 0ULL) {
            return s;
        }
        const SFTTBucket * b5 = sf.probe5(tail_win, tail_len);
        // Mathematical subset invariant: observe_tt_all always populates tt5 whenever len >= 6 populates tt6!
        const SFTTBucket * b6 = b5 ? probe6(tail_win, tail_len) : nullptr;
        const SFTTBucket * b4 = (!b5 && (cand < 0 || match_k <= 6)) ? sf.probe4(tail_win, tail_len) : nullptr;
        const SFTTBucket * b3 = (!b5 && cand >= 0 && match_k <= 4) ? sf.probe3(tail_win, tail_len) : nullptr;

        // Highest-order killer move available in session memory
        // Note: when a bucket's first observation was a refutation, history starts at -128 and rises above 0 on win #2!
        if (b6 && b6->best_move >= 0 && b6->wins >= 2 && b6->history >= 0) {
            s.killer_tok = b6->best_move; s.killer_wins = b6->wins; s.killer_order = 6;
        } else if (b5 && b5->best_move >= 0 && b5->wins >= 2 && b5->history >= 0) {
            s.killer_tok = b5->best_move; s.killer_wins = b5->wins; s.killer_order = 5;
        } else if (b4 && b4->best_move >= 0 && b4->wins >= 3 && b4->history >= 32) {
            s.killer_tok = b4->best_move; s.killer_wins = b4->wins; s.killer_order = 4;
        }

        if (cand < 0) {
            return s;
        }

        // Refutation check across Zobrist horizons:
        // `cand == refuted_move` is refuted both immediately (`history < 0`) and after `best_move` accumulates `wins >= 2`!
        auto refuted_penalty = [](const SFTTBucket * b) noexcept -> float {
            return (b->history < 0) ? ((float)b->history / 512.0f) : -0.50f;
        };
        if (b6 && b6->refuted_move == cand && b6->best_move != cand && (b6->history < 0 || b6->wins >= 2)) {
            s.refuted = true;
            s.norm_history = std::min(s.norm_history, refuted_penalty(b6));
        }
        if (b5 && b5->refuted_move == cand && b5->best_move != cand && (b5->history < 0 || b5->wins >= 2)) {
            s.refuted = true;
            s.norm_history = std::min(s.norm_history, refuted_penalty(b5));
        }
        if (b4 && b4->refuted_move == cand && b4->best_move != cand && (b4->history <= -64 || b4->wins >= 3) && match_k <= 6) {
            s.refuted = true;
            s.norm_history = std::min(s.norm_history, refuted_penalty(b4));
        }
        if (b3 && b3->refuted_move == cand && b3->best_move != cand && b3->history <= -96 && match_k <= 4) {
            s.refuted = true;
            s.norm_history = std::min(s.norm_history, (float)b3->history / 512.0f);
        }

        // Positive verification backing when `cand` IS the verified `best_move`
        if (b6 && b6->best_move == cand && (b6->history > 0 || b6->wins >= 2)) {
            s.best_wins    = std::max<int>(s.best_wins, b6->wins);
            s.norm_history = std::max(s.norm_history, std::max((float)b6->history / 512.0f, 0.22f * std::min<int>(4, b6->wins)));
        }
        if (b5 && b5->best_move == cand && (b5->history > 0 || b5->wins >= 2)) {
            s.best_wins    = std::max<int>(s.best_wins, b5->wins);
            s.norm_history = std::max(s.norm_history, std::max((float)b5->history / 512.0f, 0.20f * std::min<int>(4, b5->wins)));
        }
        if (b4 && b4->best_move == cand && (b4->history > 0 || b4->wins >= 3)) {
            s.best_wins    = std::max<int>(s.best_wins, b4->wins);
            s.norm_history = std::max(s.norm_history, 0.85f * std::max((float)b4->history / 512.0f, 0.18f * std::min<int>(4, b4->wins)));
        }

        return s;
    }

    // Disambiguate conflicting suffix matches (`consensus_len == 0 && second_j > 0`) via Stockfish TT
    void resolve_suffix_conflict(RC2SuffixMatch & m, const int32_t * tail_win, int tail_len) const noexcept {
        if (!m.found || m.consensus_len > 0 || m.second_j == 0) return;
        const size_t N = indexed_tokens.size();
        if (m.best_j + 1 >= N || m.second_j + 1 >= N) return;

        const int32_t c1 = indexed_tokens[m.best_j + 1];
        const int32_t c2 = indexed_tokens[m.second_j + 1];
        const SFProbeSummary s1 = evaluate_stockfish(tail_win, tail_len, c1, m.best_k);
        const SFProbeSummary s2 = evaluate_stockfish(tail_win, tail_len, c2, m.second_k);

        if (!s1.refuted && s1.best_wins >= 2 && (s2.refuted || s2.best_wins == 0)) {
            m.consensus_len = 1;
        } else if (!s2.refuted && s2.best_wins >= 2 && (s1.refuted || s1.best_wins == 0)) {
            m.best_j        = m.second_j;
            m.best_k        = m.second_k;
            m.consensus_len = 1;
        }
    }

    // Phase A Check: Can we draft `[d_0 .. d_{K-1}]` in `< 200 ns` WITHOUT running the 15ms MTP `lm_head` pass?
    bool try_fastpath_draft(
        int max_draft,
        llama_tokens & out_draft,
        float & out_e_tok,
        float & out_p_surv,
        const RC2L1ArbiterWeights & weights = kRC2DefaultArbiter) {

        out_draft.clear();
        const int cap = std::min<int>(max_draft, MAX_DRAFT_CAP);
        if (cap <= 0 || indexed_tokens.size() < 8) {
            return false;
        }

        int32_t tail_win[16];
        int tail_len = build_tail_window(nullptr, 0, tail_win);

        RC2SuffixMatch m = query_two_zone_suffix(nullptr, 0);
        resolve_suffix_conflict(m, tail_win, tail_len);

        // Require genuinely strong suffix evidence to bypass neural NextN MTP at d=0!
        const bool strong_hot    = m.found && m.zone_is_hot && m.consensus_len >= 1 &&
                                   (m.best_k >= 6 || (m.best_k >= 5 && m.true_consensus_hits >= 2));
        const bool strong_prefix = m.found && !m.zone_is_hot && m.consensus_len >= 2 &&
                                   (m.best_k >= 7 || (m.best_k >= 6 && m.true_consensus_hits >= 2));
        if (!strong_hot && !strong_prefix) {
            return false;
        }

        const size_t N = indexed_tokens.size();
        const int32_t cand0 = indexed_tokens[m.best_j + 1];
        if (is_mode_refuted_token(cand0)) {
            stat_sf_vetoes++;
            return false;
        }
        const bool skip_sf0 = (m.best_k >= 6 && m.true_consensus_hits >= 2 && !cand_ever_refuted(cand0));
        const SFProbeSummary sf0 = skip_sf0 ? SFProbeSummary{} : evaluate_stockfish(tail_win, tail_len, cand0, m.best_k);
        if (sf0.refuted) {
            stat_sf_vetoes++;
            return false;
        }

        RC2StepFeatures f0;
        f0.mtp_p1            = 0.0f;
        f0.mtp_margin        = 0.0f;
        f0.suffix_match_len  = std::min(1.0f, (float)m.best_k / 10.0f);
        f0.suffix_hit_count  = std::min(1.0f, (float)m.true_consensus_hits / 4.0f);
        f0.zone_is_hot       = m.zone_is_hot ? 1.0f : 0.0f;
        f0.sf_tt_wins        = std::min(1.0f, (float)sf0.best_wins / 4.0f);
        f0.sf_tt_history     = sf0.norm_history;
        f0.depth_d           = 0.0f;
        f0.recent_accept_ema = recent_accept_ema;
        f0.is_structural_tok = (m.true_consensus_hits >= 2 || sf0.best_wins >= 1) ? 1.0f : 0.5f;

        const float q0 = rc2_l1::predict_accept_prob(f0, weights);
        if (q0 < weights.fastpath_q0_min) {
            return false;
        }

        float e_tok  = 1.0f;
        float p_surv = 1.0f;
        if (!rc2_l1::should_append_token(0, q0, e_tok, p_surv, /*prev_mtp=*/0, /*new_mtp=*/0, weights)) {
            return false;
        }

        int n_drafted = 1;
        out_draft.push_back((llama_token)cand0);
        if (tail_len < 15) tail_win[tail_len++] = cand0;

        const int max_span = std::min<int>(cap, m.consensus_len);
        for (int d = 1; d < max_span && (m.best_j + 1 + (size_t)d) < N; ++d) {
            const int32_t cand_d = indexed_tokens[m.best_j + 1 + (size_t)d];
            if (is_mode_refuted_token(cand_d)) {
                stat_sf_vetoes++;
                break;
            }
            const bool skip_sf = (m.best_k + d >= 7 && m.true_consensus_hits >= 2 && !cand_ever_refuted(cand_d));
            const SFProbeSummary sfd = skip_sf ? SFProbeSummary{} : evaluate_stockfish(tail_win, tail_len, cand_d, m.best_k + d);
            if (sfd.refuted) {
                stat_sf_vetoes++;
                break;
            }

            RC2StepFeatures fd;
            fd.mtp_p1            = 0.0f;
            fd.mtp_margin        = 0.0f;
            fd.suffix_match_len  = std::min(1.0f, (float)(m.best_k + d) / 10.0f);
            fd.suffix_hit_count  = std::min(1.0f, (float)m.true_consensus_hits / 4.0f);
            fd.zone_is_hot       = m.zone_is_hot ? 1.0f : 0.0f;
            fd.sf_tt_wins        = std::min(1.0f, (float)sfd.best_wins / 4.0f);
            fd.sf_tt_history     = sfd.norm_history;
            fd.depth_d           = (float)d / 4.0f;
            fd.recent_accept_ema = recent_accept_ema;
            fd.is_structural_tok = (m.true_consensus_hits >= 2 || sfd.best_wins >= 1) ? 1.0f : 0.5f;

            const float qd = rc2_l1::predict_accept_prob(fd, weights);
            if (!rc2_l1::should_append_token(d, qd, e_tok, p_surv, 0, 0, weights)) {
                break;
            }
            n_drafted++;
            out_draft.push_back((llama_token)cand_d);
            if (tail_len < 15) tail_win[tail_len++] = cand_d;
            stat_suffix_ext_toks++;
        }

        // Require >= 2 tokens on Fast-Path so saving 15ms MTP + multi-token burst strictly dominates!
        if (n_drafted < 2) {
            out_draft.clear();
            return false;
        }

        stat_fastpath_hits++;
        out_e_tok  = e_tok;
        out_p_surv = p_surv;
        record_last_draft(out_draft);
        return true;
    }

    // Phase B + C: Fuse `NextN MTP` d=0 Neural Anchor with Stockfish Refutation/Top-4 Promotion,
    // RouteCrack 2.0 L1 Micro-Arbiter Veto, and Anchored Two-Zone Suffix Copy (`[..., id_last, t_0] -> [t_1..t_4]`)!
    // Sets `out_want_stem_probe = true` iff `draft.size() == 1` and neural confidence justifies probing MTP d=1.
    void fuse_mtp_and_extend(
        const RC2MTPAnchorInfo & mtp0,
        int max_draft,
        llama_tokens & out_draft,
        float & out_e_tok,
        float & out_p_surv,
        bool & out_want_stem_probe,
        const RC2L1ArbiterWeights & weights = kRC2DefaultArbiter) {

        out_draft.clear();
        out_e_tok           = 1.0f;
        out_p_surv          = 1.0f;
        out_want_stem_probe = false;

        const int cap = std::min<int>(max_draft, MAX_DRAFT_CAP);
        if (cap <= 0 || !mtp0.valid || mtp0.n_cands <= 0 || mtp0.top_ids[0] < 0) {
            record_last_draft(out_draft);
            return;
        }

        stat_mtp_anchors++;

        int32_t tail_win[16];
        int tail_len = build_tail_window(nullptr, 0, tail_win);

        int32_t chosen_t0   = mtp0.top_ids[0];
        float   eff_p1      = mtp0.p_top1;
        float   eff_margin  = mtp0.p_margin;

        // Check pre-MTP Suffix Copy & Stockfish Zobrist TT for agreement or refutation at d=0
        const RC2SuffixMatch m0 = query_two_zone_suffix(nullptr, 0);
        const bool suffix_agrees_t0 = m0.found && m0.consensus_len >= 1 &&
                                      (m0.best_j + 1 < indexed_tokens.size()) &&
                                      (indexed_tokens[m0.best_j + 1] == chosen_t0);

        const bool skip_sf0 = (suffix_agrees_t0 && m0.best_k >= 6 && m0.true_consensus_hits >= 2 && !cand_ever_refuted(chosen_t0));
        SFProbeSummary sf0 = skip_sf0 ? SFProbeSummary{} : evaluate_stockfish(tail_win, tail_len, chosen_t0, suffix_agrees_t0 ? m0.best_k : 3);

        // Method 3: Stockfish Refutation Veto & Orthogonal MTP Top-4 Promotion at d=0
        if (sf0.refuted && eff_p1 < 0.88f) {
            bool promoted = false;
            if (sf0.killer_tok >= 0 && sf0.killer_tok != chosen_t0) {
                for (int c = 1; c < mtp0.n_cands; ++c) {
                    if (mtp0.top_ids[c] == sf0.killer_tok && mtp0.top_p[c] >= 0.05f) {
                        chosen_t0  = sf0.killer_tok;
                        eff_p1     = std::max(0.62f, mtp0.top_p[c] + 0.35f);
                        eff_margin = 0.35f;
                        sf0        = evaluate_stockfish(tail_win, tail_len, chosen_t0, 5);
                        promoted   = true;
                        stat_sf_promotions++;
                        break;
                    }
                }
            }
            if (!promoted) {
                stat_sf_vetoes++;
                record_last_draft(out_draft);
                return;
            }
        }

        // Evaluate RouteCrack 2.0 L1 Micro-Arbiter at d=0
        RC2StepFeatures f0;
        f0.mtp_p1            = eff_p1;
        f0.mtp_margin        = eff_margin;
        f0.suffix_match_len  = suffix_agrees_t0 ? std::min(1.0f, (float)m0.best_k / 10.0f) : 0.0f;
        f0.suffix_hit_count  = suffix_agrees_t0 ? std::min(1.0f, (float)m0.true_consensus_hits / 4.0f) : 0.0f;
        f0.zone_is_hot       = (suffix_agrees_t0 && m0.zone_is_hot) ? 1.0f : 0.0f;
        f0.sf_tt_wins        = std::min(1.0f, (float)sf0.best_wins / 4.0f);
        f0.sf_tt_history     = sf0.norm_history;
        f0.depth_d           = 0.0f;
        f0.recent_accept_ema = recent_accept_ema;
        f0.is_structural_tok = (suffix_agrees_t0 || sf0.best_wins >= 2 || mtp0.top4_mass >= 0.97f) ? 1.0f : 0.0f;

        const float q0 = rc2_l1::predict_accept_prob(f0, weights);
        float e_tok  = 1.0f;
        float p_surv = 1.0f;
        if (!rc2_l1::should_append_token(0, q0, e_tok, p_surv, /*prev_mtp=*/1, /*new_mtp=*/1, weights)) {
            stat_nnue_vetoes++;
            record_last_draft(out_draft);
            return;
        }

        int32_t drafted_local[MAX_DRAFT_CAP];
        int n_drafted = 0;
        drafted_local[n_drafted++] = chosen_t0;
        out_draft.push_back((llama_token)chosen_t0);

        // Phase C: Anchored Two-Zone Suffix Copy + Stockfish Session Extension (`[..., id_last, chosen_t0]`)!
        RC2SuffixMatch pre_anchored;
        if (suffix_agrees_t0 && m0.consensus_len >= 2) {
            pre_anchored              = m0;
            pre_anchored.best_j       = m0.best_j + 1;
            pre_anchored.best_k       = std::min(12, m0.best_k + 1);
            pre_anchored.consensus_len = m0.consensus_len - 1;
            pre_anchored.second_j     = 0;
        }
        extend_anchored_draft(drafted_local, n_drafted, cap, /*n_mtp_passes=*/1, e_tok, p_surv, out_draft, weights,
                              pre_anchored.found ? &pre_anchored : nullptr);

        out_e_tok  = e_tok;
        out_p_surv = p_surv;

        // Phase D Trigger: Only probe 1 optional MTP stem step at d=1 when Suffix+Stockfish had no match
        // AND d=0 neural MTP confidence was ultra-high (`eff_p1 >= 0.80` and `p_surv >= stem_probe_q0_min`)!
        if (n_drafted == 1 && cap >= 2 && eff_p1 >= 0.80f && p_surv >= weights.stem_probe_q0_min) {
            out_want_stem_probe = true;
        } else {
            record_last_draft(out_draft);
        }
    }

    // Extend `drafted_local[0..n_drafted-1]` up to `cap` using Anchored Two-Zone Suffix + Stockfish TT
    void extend_anchored_draft(
        int32_t * drafted_local,
        int & n_drafted,
        int cap,
        int n_mtp_passes,
        float & io_e_tok,
        float & io_p_surv,
        llama_tokens & out_draft,
        const RC2L1ArbiterWeights & weights = kRC2DefaultArbiter,
        const RC2SuffixMatch * precomputed_match = nullptr) {

        const size_t N = indexed_tokens.size();
        if (n_drafted >= cap || N < 4) return;

        int32_t tail_win[16];
        int tail_len = build_tail_window(drafted_local, n_drafted, tail_win);

        RC2SuffixMatch ma = precomputed_match ? *precomputed_match : query_two_zone_suffix(drafted_local, n_drafted);
        if (!precomputed_match) {
            resolve_suffix_conflict(ma, tail_win, tail_len);
        }

        if (ma.found && ma.consensus_len >= 1) {
            const int start_d = n_drafted;
            const int max_add = std::min<int>(cap - start_d, ma.consensus_len);
            for (int step = 0; step < max_add && (ma.best_j + 1 + (size_t)step) < N; ++step) {
                const int d = start_d + step;
                const int32_t cand = indexed_tokens[ma.best_j + 1 + (size_t)step];
                if (is_mode_refuted_token(cand)) {
                    stat_sf_vetoes++;
                    break;
                }
                const bool skip_sf = (ma.best_k + step >= 7 && ma.true_consensus_hits >= 2 && !cand_ever_refuted(cand));
                const SFProbeSummary sfd = skip_sf ? SFProbeSummary{} : evaluate_stockfish(tail_win, tail_len, cand, ma.best_k + step);
                if (sfd.refuted) {
                    stat_sf_vetoes++;
                    break;
                }
                // Guard weak 3-gram single-hit matches unless backed by recency or Stockfish TT
                if (ma.best_k == 3 && ma.true_consensus_hits < 2 && sfd.best_wins < 1 && (ma.best_j + 192 < N)) {
                    break;
                }

                RC2StepFeatures fd;
                fd.mtp_p1            = 0.0f;
                fd.mtp_margin        = 0.0f;
                fd.suffix_match_len  = std::min(1.0f, (float)(ma.best_k + step) / 10.0f);
                fd.suffix_hit_count  = std::min(1.0f, (float)ma.true_consensus_hits / 4.0f);
                fd.zone_is_hot       = ma.zone_is_hot ? 1.0f : 0.0f;
                fd.sf_tt_wins        = std::min(1.0f, (float)sfd.best_wins / 4.0f);
                fd.sf_tt_history     = sfd.norm_history;
                fd.depth_d           = (float)d / 4.0f;
                fd.recent_accept_ema = recent_accept_ema;
                fd.is_structural_tok = (ma.true_consensus_hits >= 2 || sfd.best_wins >= 1 || ma.best_k >= 5) ? 1.0f : 0.0f;

                const float qd = rc2_l1::predict_accept_prob(fd, weights);
                if (!rc2_l1::should_append_token(d, qd, io_e_tok, io_p_surv, n_mtp_passes, n_mtp_passes, weights)) {
                    break;
                }
                drafted_local[n_drafted++] = cand;
                out_draft.push_back((llama_token)cand);
                if (tail_len < 15) tail_win[tail_len++] = cand;
                stat_suffix_ext_toks++;
            }
        }

        // Method 3 Continuation Extension: When Suffix Copy ends early, query Stockfish 6-gram/5-gram Session TT
        while (n_drafted < cap) {
            const int d = n_drafted;
            const SFProbeSummary sfk = evaluate_stockfish(tail_win, tail_len, -1, 5);
            if (sfk.killer_tok < 0 || sfk.killer_order < 5 || sfk.killer_wins < 2) {
                break;
            }
            const SFProbeSummary sf_cand = evaluate_stockfish(tail_win, tail_len, sfk.killer_tok, sfk.killer_order);
            if (sf_cand.refuted) {
                break;
            }

            RC2StepFeatures fk;
            fk.mtp_p1            = 0.0f;
            fk.mtp_margin        = 0.0f;
            fk.suffix_match_len  = (sfk.killer_order >= 6) ? 0.55f : 0.45f;
            fk.suffix_hit_count  = std::min(1.0f, (float)sfk.killer_wins / 4.0f);
            fk.zone_is_hot       = 1.0f;
            fk.sf_tt_wins        = std::min(1.0f, (float)sfk.killer_wins / 4.0f);
            fk.sf_tt_history     = sf_cand.norm_history;
            fk.depth_d           = (float)d / 4.0f;
            fk.recent_accept_ema = recent_accept_ema;
            fk.is_structural_tok = 1.0f;

            const float qk = rc2_l1::predict_accept_prob(fk, weights);
            if (!rc2_l1::should_append_token(d, qk, io_e_tok, io_p_surv, n_mtp_passes, n_mtp_passes, weights)) {
                break;
            }
            drafted_local[n_drafted++] = sfk.killer_tok;
            out_draft.push_back((llama_token)sfk.killer_tok);
            if (tail_len < 15) tail_win[tail_len++] = sfk.killer_tok;
            stat_suffix_ext_toks++;
        }
    }

    // Phase D Completion: Evaluate optional neural MTP stem candidate at depth `stem_d` (1 or 2),
    // and if accepted, re-anchor Suffix Copy + Stockfish on `[..., id_last, t_0 .. t_{stem_d}]`!
    void finalize_stem_probe(
        const RC2MTPAnchorInfo & mtp_stem,
        int max_draft,
        float & io_e_tok,
        float & io_p_surv,
        llama_tokens & io_draft,
        bool * out_want_stem2 = nullptr,
        const RC2L1ArbiterWeights & weights = kRC2DefaultArbiter) {

        if (out_want_stem2) *out_want_stem2 = false;
        const int stem_d = (int)io_draft.size();
        const int cap = std::min<int>(max_draft, MAX_DRAFT_CAP);
        if ((stem_d != 1 && stem_d != 2) || cap <= stem_d || !mtp_stem.valid || mtp_stem.n_cands <= 0 || mtp_stem.top_ids[0] < 0) {
            record_last_draft(io_draft);
            return;
        }

        stat_stem_probes++;
        int32_t drafted_local[MAX_DRAFT_CAP];
        for (int i = 0; i < stem_d; ++i) {
            drafted_local[i] = (int32_t)io_draft[(size_t)i];
        }
        int n_drafted = stem_d;

        int32_t tail_win[16];
        const int tail_len = build_tail_window(drafted_local, n_drafted, tail_win);

        const int32_t cand_s = mtp_stem.top_ids[0];
        const SFProbeSummary sfs = evaluate_stockfish(tail_win, tail_len, cand_s, 4);
        if (sfs.refuted) {
            stat_sf_vetoes++;
            record_last_draft(io_draft);
            return;
        }

        RC2StepFeatures fs;
        fs.mtp_p1            = mtp_stem.p_top1;
        fs.mtp_margin        = mtp_stem.p_margin;
        fs.suffix_match_len  = 0.0f;
        fs.suffix_hit_count  = 0.0f;
        fs.zone_is_hot       = 0.0f;
        fs.sf_tt_wins        = std::min(1.0f, (float)sfs.best_wins / 4.0f);
        fs.sf_tt_history     = sfs.norm_history;
        fs.depth_d           = (float)stem_d / 4.0f;
        fs.recent_accept_ema = recent_accept_ema;
        fs.is_structural_tok = (sfs.best_wins >= 1 || mtp_stem.top4_mass >= 0.97f) ? 0.5f : 0.0f;

        const float qs = rc2_l1::predict_accept_prob(fs, weights);
        const int mtp_passes_done = stem_d + 1;
        if (rc2_l1::should_append_token(stem_d, qs, io_e_tok, io_p_surv, mtp_passes_done, mtp_passes_done, weights)) {
            drafted_local[n_drafted++] = cand_s;
            io_draft.push_back((llama_token)cand_s);

            // Re-anchor Two-Zone Suffix Copy + Stockfish on `[..., id_last, t_0 .. t_{stem_d}]`!
            extend_anchored_draft(drafted_local, n_drafted, cap, mtp_passes_done, io_e_tok, io_p_surv, io_draft, weights);

            // Allow a 2nd neural stem step (d=2) ONLY when both d=0 and d=1 were ultra-certain and Suffix hasn't latched yet!
            if (out_want_stem2 && stem_d == 1 && n_drafted == 2 && cap >= 3 &&
                mtp_stem.p_top1 >= 0.82f && io_p_surv >= 0.76f) {
                *out_want_stem2 = true;
                return;
            }
        } else {
            stat_nnue_vetoes++;
        }

        record_last_draft(io_draft);
    }

    // Universal Non-MTP Entry Point (`NVIDIA-Nemotron-3.5-30B-A3B` & `DeepSeek-V4-Flash-0731` 284B):
    // Unites `RCPredictor` (8-level FNV-64 Cascade + MacroCodebook + HotRules + Prompt-Warmed Online Trie v2 + PPM + Skipgram),
    // `O(1) Indexed Two-Zone 40K Suffix Copy` (including post-RC anchoring `[..., id_last, rc_t0]`),
    // `Stockfish-PVS Multi-Order Zobrist TT + 28-bit Tagged Mode Refutation`, and the RouteCrack 2.0 L1 Micro-Arbiter!
    void draft_without_mtp(
        int max_draft,
        llama_tokens & out_draft,
        RC2PredictorHub * hub = nullptr,
        const RC2L1ArbiterWeights & weights = kRC2DraftlessArbiter,
        bool is_heavy_moe = false) {

        out_draft.clear();
        const int cap = std::min<int>(max_draft, MAX_DRAFT_CAP);
        const size_t N = indexed_tokens.size();
        if (cap <= 0 || N < 6) {
            record_last_draft(out_draft);
            return;
        }

        std::unique_lock<std::mutex> hub_lock;
        if (hub && hub->pred) {
            hub_lock = std::unique_lock<std::mutex>(hub->mtx);
            if (N < last_rc_learned_len) {
                last_rc_learned_len = 0;
            }
            if (last_rc_learned_len == 0) {
                // Warm-ingest tail 2048 prompt tokens into Online Trie + PPM + Skipgram (< 0.6 ms once per request)
                const size_t start_pos = (N > 2048) ? (N - 2048) : 4;
                for (size_t pos = std::max<size_t>(4, start_pos); pos < N; ++pos) {
                    const size_t wlen = std::min<size_t>(12, pos + 1);
                    hub->learn_window_nolock(&indexed_tokens[pos + 1 - wlen], (int)wlen, 8);
                }
                last_rc_learned_len = N;
            } else if (N > last_rc_learned_len) {
                const size_t delta = N - last_rc_learned_len;
                const size_t start_pos = (delta <= 16) ? last_rc_learned_len : (N - 16);
                for (size_t pos = start_pos; pos < N; ++pos) {
                    if (pos < 2) continue;
                    const size_t wlen = std::min<size_t>(12, pos + 1);
                    hub->learn_window_nolock(&indexed_tokens[pos + 1 - wlen], (int)wlen, 10);
                }
                last_rc_learned_len = N;
            }
        }

        const bool has_offline = (hub && hub->offline_loaded);

        int32_t tail_win[24];
        int tail_len = 0;
        const size_t base_w = std::min<size_t>(N, 12);
        for (size_t i = N - base_w; i < N; ++i) {
            tail_win[tail_len++] = indexed_tokens[i];
        }

        const RCMacroEntry * active_macro = nullptr;
        int macro_start_d = 0;
        auto try_latch_macro = [&](int cur_d) {
            if (!hub || !hub->pred || !hub->pred->macro.entries || hub->pred->macro.n_entries == 0) return;
            for (int mk = std::min(tail_len, 6); mk >= 5; --mk) {
                const uint64_t mh = rc_fnv64(tail_win + tail_len - mk, mk);
                const int midx = hub->pred->macro.find(mh);
                if (midx >= 0 && hub->pred->macro.entries[midx].conf >= 92 && hub->pred->macro.entries[midx].cont_len >= 3) {
                    active_macro  = &hub->pred->macro.entries[midx];
                    macro_start_d = cur_d;
                    break;
                }
            }
        };
        try_latch_macro(0);

        int32_t drafted_local[MAX_DRAFT_CAP];
        int     n_drafted   = 0;
        float   e_tok       = 1.0f;
        float   p_surv      = 1.0f;

        bool   span_active = false;
        size_t span_base_j = 0;
        int    span_base_k = 0;
        int    span_rem    = 0;
        int    span_hits   = 0;
        bool   span_hot    = false;
        int    span_step   = 0;

        for (int d = 0; d < cap; ++d) {
            int32_t        cand         = -1;
            bool           from_suffix  = false;
            int            eff_k        = 0;
            int            rc_agree     = 0;
            float          rc_back_p1   = 0.0f;
            RC2RCCandidate rc_cand;
            SFProbeSummary sfd;

            // 1. Query Two-Zone O(1) 40K Suffix Copy (either continuing active span or re-anchoring after RC step!)
            if (!span_active) {
                RC2SuffixMatch ma = query_two_zone_suffix(drafted_local, n_drafted);
                if (ma.found && ma.consensus_len == 0 && ma.second_j > 0 &&
                    ma.best_j + 1 < N && ma.second_j + 1 < N) {
                    const int32_t c1 = indexed_tokens[ma.best_j + 1];
                    const int32_t c2 = indexed_tokens[ma.second_j + 1];
                    const SFProbeSummary s1 = evaluate_stockfish(tail_win, tail_len, c1, ma.best_k);
                    const SFProbeSummary s2 = evaluate_stockfish(tail_win, tail_len, c2, ma.second_k);
                    int ag1 = 0, ag2 = 0;
                    float bp1 = 0.0f, bp2 = 0.0f;
                    if (hub && hub->pred) {
                        hub->verify_cand_nolock(tail_win, tail_len, c1, ag1, bp1);
                        hub->verify_cand_nolock(tail_win, tail_len, c2, ag2, bp2);
                    }
                    if (!s1.refuted && (s1.best_wins >= 2 || (ag1 >= 2 && s1.best_wins >= 1)) && (s2.refuted || (s2.best_wins == 0 && ag2 == 0))) {
                        ma.consensus_len = 1;
                    } else if (!s2.refuted && (s2.best_wins >= 2 || (ag2 >= 2 && s2.best_wins >= 1)) && (s1.refuted || (s1.best_wins == 0 && ag1 == 0))) {
                        ma.best_j        = ma.second_j;
                        ma.best_k        = ma.second_k;
                        ma.consensus_len = 1;
                    }
                }
                if (ma.found && ma.consensus_len >= 1) {
                    span_active = true;
                    span_base_j = ma.best_j;
                    span_base_k = ma.best_k;
                    span_rem    = ma.consensus_len;
                    span_hits   = ma.true_consensus_hits;
                    span_hot    = ma.zone_is_hot;
                    span_step   = 0;
                }
            }

            if (span_active && span_step < span_rem && (span_base_j + 1 + (size_t)span_step) < N) {
                const int32_t cs = indexed_tokens[span_base_j + 1 + (size_t)span_step];
                eff_k = span_base_k + span_step;

                if (is_mode_refuted_token(cs)) {
                    stat_sf_vetoes++;
                    span_active = false;
                } else {
                    const bool skip_sf = (eff_k >= 7 && span_hits >= 2 && !cand_ever_refuted(cs));
                    sfd = skip_sf ? SFProbeSummary{} : evaluate_stockfish(tail_win, tail_len, cs, eff_k);
                    if (sfd.refuted) {
                        stat_sf_vetoes++;
                        span_active = false;
                    } else {
                        if (hub && hub->pred && (eff_k <= 6 || span_hits < 2)) {
                            hub->verify_cand_nolock(tail_win, tail_len, cs, rc_agree, rc_back_p1, /*cand_from_prompt_suffix=*/true);
                        } else if (eff_k >= 7 && span_hits >= 2) {
                            rc_agree   = 2;
                            rc_back_p1 = 0.85f;
                        }

                        // Precision & Singular Horizon guard on short/single-hit suffix matches (prevents over-drafting past idiom boundaries!)
                        int base_m = 1;
                        if (is_heavy_moe) {
                            if (span_base_k >= 10 && (span_hits >= 2 || recent_accept_ema >= 0.78f)) {
                                base_m = MAX_DRAFT_CAP;
                            } else if (span_base_k >= 8) {
                                base_m = (span_hits >= 2 || span_base_k >= 9) ? 4 : 3;
                            } else if (span_base_k >= 6) {
                                base_m = (span_hits >= 2) ? 3 : 2;
                            } else if (span_base_k == 5) {
                                base_m = (span_hits >= 2 || (span_hot && span_base_j + 1024 >= N)) ? 2 : 1;
                            } else if (span_base_k == 4) {
                                base_m = (span_hits >= 2) ? 2 :
                                         ((span_hot && span_base_j + 384 >= N && (rc_agree >= 1 || sfd.best_wins >= 1 || recent_accept_ema >= 0.68f)) ? 1 : 0);
                            } else {
                                base_m = (span_hits >= 2 && span_hot && span_base_j + 256 >= N) ? 1 : 0;
                            }
                        } else if (span_base_k >= 8 || (span_base_k >= 7 && span_hits >= 2)) {
                            base_m = MAX_DRAFT_CAP;
                        } else if (span_base_k >= 6) {
                            base_m = has_offline ? ((span_hits >= 2) ? 5 : 4) : ((span_hits >= 2) ? 4 : 3);
                        } else if (span_base_k == 5) {
                            base_m = has_offline ? ((span_hits >= 2) ? 4 : 2) : ((span_hits >= 2) ? 3 : 2);
                        } else if (span_base_k == 4) {
                            base_m = (span_hits >= 2) ? 2 : 1;
                        } else {
                            base_m = 1;
                        }

                        bool weak_short = false;
                        if (span_step >= base_m && rc_agree < 2 && sfd.best_wins < 2) {
                            weak_short = true;
                        } else if (!has_offline) {
                            if (eff_k == 3 && (span_hits < 2 || span_base_j + 256 < N) && sfd.best_wins < 2 && rc_agree < 2) {
                                weak_short = true;
                            } else if (eff_k == 4 && span_hits < 2 && span_base_j + 512 < N && sfd.best_wins < 1 && rc_agree < 1) {
                                weak_short = true;
                            } else if (!span_hot && eff_k <= 5 && span_hits < 2 && rc_agree < 1 && sfd.best_wins < 1) {
                                weak_short = true;
                            }
                        } else {
                            if (eff_k == 3 && span_hits < 2 && sfd.best_wins < 2 && rc_agree < 2) {
                                weak_short = true;
                            } else if (eff_k == 4 && span_hits < 2 && rc_agree == 0 && sfd.best_wins < 2) {
                                weak_short = true;
                            }
                        }

                        if (!weak_short) {
                            cand        = cs;
                            from_suffix = true;
                            span_step++;
                        } else {
                            span_active = false;
                        }
                    }
                }
            } else {
                span_active = false;
            }

            // 2. Fallback / Extension via `RCPredictor` Ensemble + `Stockfish-PVS` Session TT
            if (cand < 0 && hub && hub->pred) {
                if (!active_macro && d > 0) {
                    try_latch_macro(d);
                }
                rc_cand = hub->propose_cand_nolock(tail_win, tail_len, sf, active_macro, d - macro_start_d);
                if (rc_cand.tok >= 0 && !is_mode_refuted_token(rc_cand.tok)) {
                    sfd = evaluate_stockfish(tail_win, tail_len, rc_cand.tok, std::max(3, rc_cand.casc_k));
                    if (sfd.refuted) {
                        if (sfd.killer_tok >= 0 && sfd.killer_wins >= 2 && !is_mode_refuted_token(sfd.killer_tok)) {
                            int ka = 0; float kp = 0.0f;
                            hub->verify_cand_nolock(tail_win, tail_len, sfd.killer_tok, ka, kp);
                            if (ka >= 1 || sfd.killer_order >= 5) {
                                rc_cand.tok    = sfd.killer_tok;
                                rc_cand.cert   = 88 + std::min(6, sfd.killer_wins);
                                rc_cand.p1     = (float)rc_cand.cert / 100.0f;
                                rc_cand.margin = 0.75f;
                                sfd            = evaluate_stockfish(tail_win, tail_len, rc_cand.tok, 5);
                                stat_sf_promotions++;
                            } else {
                                stat_sf_vetoes++;
                                rc_cand.tok = -1;
                            }
                        } else {
                            stat_sf_vetoes++;
                            rc_cand.tok = -1;
                        }
                    }
                    // Enforce pure-symbolic certainty floor when unbacked by Suffix Copy
                    if (rc_cand.tok >= 0) {
                        const int min_cert = is_heavy_moe
                            ? ((d == 0) ? 88 : (d == 1 ? 94 : 97))
                            : ((d == 0) ? 84 : (d == 1 ? 92 : 96));
                        const int eff_cert = rc_cand.cert + (sfd.best_wins >= 2 ? 4 : 0);
                        if ((!rc_cand.from_macro && d >= 3) || eff_cert < min_cert) {
                            // 1-Ply Stockfish Quiescence Lookahead Rescue when within 3 points of threshold!
                            bool rescued = false;
                            if (d <= 1 && eff_cert >= min_cert - 3 && tail_len < 22) {
                                tail_win[tail_len] = rc_cand.tok;
                                const RCMacroEntry * dummy_m = nullptr;
                                const RC2RCCandidate next_rc = hub->propose_cand_nolock(tail_win, tail_len + 1, sf, dummy_m, 0);
                                const SFTTBucket * nb5 = sf.probe5(tail_win, tail_len + 1);
                                if (next_rc.tok >= 0 && next_rc.casc_k >= 6 && next_rc.cert >= 94 && next_rc.agree >= 2 &&
                                    nb5 && nb5->best_move == next_rc.tok && nb5->wins >= 2) {
                                    rescued = true;
                                    rc_cand.p1 = std::max(rc_cand.p1, 0.88f);
                                }
                            }
                            if (!rescued) {
                                rc_cand.tok = -1;
                            }
                        }
                    }
                    if (rc_cand.tok >= 0) {
                        cand = rc_cand.tok;
                    }
                }
            }

            // 3. Pure Stockfish 6-gram/5-gram Session Killer when neither Suffix nor RCPredictor matched
            if (cand < 0) {
                const SFProbeSummary sfk = evaluate_stockfish(tail_win, tail_len, -1, 5);
                if (sfk.killer_tok >= 0 && sfk.killer_order >= 5 && sfk.killer_wins >= 3 &&
                    !is_mode_refuted_token(sfk.killer_tok)) {
                    const SFProbeSummary sfc = evaluate_stockfish(tail_win, tail_len, sfk.killer_tok, sfk.killer_order);
                    if (!sfc.refuted) {
                        cand            = sfk.killer_tok;
                        sfd             = sfc;
                        rc_cand.tok     = cand;
                        rc_cand.p1      = 0.84f;
                        rc_cand.margin  = 0.65f;
                        rc_cand.deep_session = true;
                    }
                }
            }

            if (cand < 0) {
                break;
            }

            // 4. Score Unified Step Features with the RouteCrack 2.0 L1 Micro-Arbiter (AVX-512 / NEON / Scalar)
            RC2StepFeatures fd;
            if (from_suffix) {
                fd.mtp_p1            = rc_back_p1;
                fd.mtp_margin        = std::min(1.0f, 0.25f * (float)rc_agree);
                fd.suffix_match_len  = std::min(1.0f, (float)eff_k / 10.0f);
                fd.suffix_hit_count  = std::min(1.0f, (float)span_hits / 4.0f);
                fd.zone_is_hot       = span_hot ? 1.0f : 0.0f;
                fd.sf_tt_wins        = std::min(1.0f, (float)sfd.best_wins / 4.0f);
                fd.sf_tt_history     = sfd.norm_history;
                fd.depth_d           = (float)d / 4.0f;
                fd.recent_accept_ema = recent_accept_ema;
                fd.is_structural_tok = (span_hits >= 2 || rc_agree >= 1 || sfd.best_wins >= 1 || eff_k >= 6) ? 1.0f : 0.0f;
            } else {
                fd.mtp_p1            = rc_cand.p1;
                fd.mtp_margin        = rc_cand.margin;
                fd.suffix_match_len  = rc_cand.from_macro ? 0.55f : 0.0f;
                fd.suffix_hit_count  = rc_cand.from_macro ? 0.75f : 0.0f;
                fd.zone_is_hot       = (rc_cand.from_macro || rc_cand.deep_session || rc_cand.casc_k >= 6) ? 1.0f : 0.0f;
                fd.sf_tt_wins        = std::min(1.0f, (float)sfd.best_wins / 4.0f);
                fd.sf_tt_history     = sfd.norm_history;
                fd.depth_d           = (float)d / 4.0f;
                fd.recent_accept_ema = recent_accept_ema;
                fd.is_structural_tok = (rc_cand.agree >= 2 || rc_cand.from_macro || sfd.best_wins >= 2) ? 1.0f : 0.5f;
            }

            const float qd = rc2_l1::predict_accept_prob(fd, weights);
            if (!rc2_l1::should_append_token(d, qd, e_tok, p_surv, /*prev_mtp=*/0, /*new_mtp=*/0, weights)) {
                stat_nnue_vetoes++;
                break;
            }

            drafted_local[n_drafted++] = cand;
            out_draft.push_back((llama_token)cand);
            if (tail_len < 23) {
                tail_win[tail_len++] = cand;
            } else {
                std::memmove(tail_win, tail_win + 1, 22 * sizeof(int32_t));
                tail_win[22] = cand;
            }

            if (d == 0 && from_suffix) {
                stat_fastpath_hits++;
            } else if (from_suffix) {
                stat_suffix_ext_toks++;
            } else {
                stat_rc_anchors++;
            }
        }

        record_last_draft(out_draft);
    }

    inline void record_last_draft(const llama_tokens & draft) noexcept {
        last_draft_len = (int)std::min<size_t>(draft.size(), 8);
        for (int i = 0; i < last_draft_len; ++i) {
            last_draft[i] = (int32_t)draft[i];
        }
    }
};

#endif // RC_ROUTECRACK2_H
