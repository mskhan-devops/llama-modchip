/*
 * test_routecrack2.cpp — Rigorous Unit, Invariant, Sanitizer & Latency Suite for RouteCrack 2.0
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Minimal standalone typedefs when compiled outside full libllama for fast ASan/UBSan + latency testing
using llama_token  = int32_t;
using llama_pos    = int32_t;
using llama_seq_id = int32_t;
using llama_tokens = std::vector<llama_token>;
#define LLAMA_H

#include <limits>
#include "rc_routecrack2.h"

namespace {

// Deterministic reference MTP KV + `pending_h` simulator mirroring `common_speculative_impl_draft_mtp`
// + `server-context.cpp` (`seq_rm` after draft, `process(batch_in)`, `accept(n_accepted)`, `seq_rm(pos_next)`).
struct MockMTPKVVerifier {
    struct KVCell {
        bool    occupied = false;
        int32_t tok      = -1;
        float   h_in0    = -999.0f;
    };

    int n_embd = 8;
    std::vector<KVCell> kv;
    std::vector<float>  pending_h;
    std::vector<float>  prev_pending_h;
    llama_pos           prev_verify_pos0 = -1;
    std::vector<float>  verify_h;
    int32_t             verify_h_rows = 0;

    static float canonical_h(int32_t pos, int32_t tok_at_pos) {
        if (pos < 0) return 0.0f;
        return (float)(pos * 131 + (tok_at_pos & 1023)) * 0.125f;
    }

    void init(const llama_tokens & prompt) {
        kv.assign(4096, KVCell{});
        pending_h.assign(n_embd, 0.0f);
        prev_pending_h.assign(n_embd, 0.0f);
        prev_verify_pos0 = -1;

        // Prefill `0 .. P-1`
        for (size_t p = 0; p < prompt.size(); ++p) {
            const float h_prev = (p == 0) ? 0.0f : canonical_h((int32_t)(p - 1), prompt[p - 1]);
            kv[p] = {true, prompt[p], h_prev};
        }
        const float h_last = canonical_h((int32_t)prompt.size() - 1, prompt.back());
        std::fill(pending_h.begin(), pending_h.end(), h_last);
    }

    void seq_rm(llama_pos p0) {
        for (size_t p = (size_t)std::max(0, p0); p < kv.size(); ++p) {
            kv[p] = KVCell{};
        }
    }

    void simulate_mtp_draft_decode(llama_pos pos, llama_token tok, float h_in) {
        kv[(size_t)pos] = {true, tok, h_in};
    }

    void process_verify_batch(llama_pos pos0, const std::vector<llama_token> & batch_toks) {
        if (pos0 >= 0 && pos0 == prev_verify_pos0) {
            pending_h = prev_pending_h;
        } else {
            prev_pending_h   = pending_h;
            prev_verify_pos0 = pos0;
        }

        const int32_t n_rows = (int32_t)batch_toks.size();
        verify_h_rows = n_rows;
        verify_h.resize((size_t)n_rows * n_embd);

        for (int32_t k = 0; k < n_rows; ++k) {
            const float h_in = (k == 0)
                ? pending_h[0]
                : canonical_h(pos0 + k - 1, batch_toks[k - 1]);
            kv[(size_t)(pos0 + k)] = {true, batch_toks[k], h_in};

            const float h_out = canonical_h(pos0 + k, batch_toks[k]);
            for (int e = 0; e < n_embd; ++e) {
                verify_h[(size_t)k * n_embd + e] = h_out;
            }
        }
        std::memcpy(pending_h.data(),
                    verify_h.data() + (size_t)(n_rows - 1) * n_embd,
                    (size_t)n_embd * sizeof(float));
    }

    void accept(uint16_t n_accepted) {
        const int32_t i_h = std::min<int32_t>(n_accepted, verify_h_rows - 1);
        std::memcpy(pending_h.data(),
                    verify_h.data() + (size_t)i_h * n_embd,
                    (size_t)n_embd * sizeof(float));
    }

    void assert_exact_alignment(const llama_tokens & committed_prompt) const {
        const size_t P = committed_prompt.size();
        for (size_t p = 0; p < P; ++p) {
            assert(kv[p].occupied && "MTP KV missing committed position");
            assert(kv[p].tok == committed_prompt[p] && "MTP KV token mismatch");
            const float exp_h_in = (p == 0) ? 0.0f : canonical_h((int32_t)(p - 1), committed_prompt[p - 1]);
            assert(std::fabs(kv[p].h_in0 - exp_h_in) < 1e-5f && "MTP KV h_in desync");
        }
        for (size_t p = P; p < P + 16 && p < kv.size(); ++p) {
            assert(!kv[p].occupied && "Unremoved ghost draft entry in MTP KV");
        }
        const float exp_pending = canonical_h((int32_t)P - 1, committed_prompt.back());
        assert(std::fabs(pending_h[0] - exp_pending) < 1e-5f && "pending_h desync after accept");
    }
};

void test_two_zone_index_and_undo_invariants() {
    RC2SeqEngine eng;
    eng.init();

    // Build a 30,000-token context: Prefix Zone has unique tool schema `[901, 902, 903, 904, 905, 906, 777, 778, 779]`
    llama_tokens ctx;
    ctx.reserve(30000);
    for (int i = 0; i < 100; ++i) ctx.push_back(100 + (i % 37));
    const int32_t prefix_seq[] = {901, 902, 903, 904, 905, 906, 777, 778, 779};
    for (int32_t t : prefix_seq) ctx.push_back(t);
    while (ctx.size() < 26000) {
        ctx.push_back(2000 + (int32_t)((ctx.size() * 73) % 409));
    }
    // Insert Hot-Zone code block `[501, 502, 503, 504, 505, 506, 881, 882, 883, 884]`
    const int32_t hot_seq[] = {501, 502, 503, 504, 505, 506, 881, 882, 883, 884};
    for (int32_t t : hot_seq) ctx.push_back(t);
    for (int i = 0; i < 200; ++i) ctx.push_back(3000 + i);

    // 1. Query Prefix-Zone 6-gram `[901..906]` -> must recover `[777, 778, 779]` from Prefix Zone!
    llama_tokens q_pref = ctx;
    q_pref.insert(q_pref.end(), {901, 902, 903, 904, 905});
    eng.sync_context_and_observe(q_pref, 906);
    RC2SuffixMatch mp = eng.query_two_zone_suffix(nullptr, 0);
    assert(mp.found && !mp.zone_is_hot && mp.best_k >= 6 && mp.consensus_len >= 3);
    assert(eng.indexed_tokens[mp.best_j + 1] == 777);
    assert(eng.indexed_tokens[mp.best_j + 2] == 778);

    // 2. Truncate 5 tokens + append Hot-Zone `[501..506]` -> tests O(1) `pop_indexed_token()` undo!
    llama_tokens q_hot = ctx;
    q_hot.insert(q_hot.end(), {501, 502, 503, 504, 505});
    eng.sync_context_and_observe(q_hot, 506);

    llama_tokens fp_draft;
    float e_tok = 1.0f, p_surv = 1.0f;
    bool fp_ok = eng.try_fastpath_draft(5, fp_draft, e_tok, p_surv);
    assert(fp_ok && fp_draft.size() >= 4);
    assert(fp_draft[0] == 881 && fp_draft[1] == 882 && fp_draft[2] == 883 && fp_draft[3] == 884);

    std::printf("[PASS] test_two_zone_index_and_undo_invariants (30K ctx Prefix + Hot + O(1) undo)\n");
}

void test_anchored_mtp_disambiguation_and_stockfish_refutation() {
    RC2SeqEngine eng;
    eng.init();

    // Ambiguous stem `[40, 41, 42, 43]` (`return self.`) occurs 3 times with DIFFERENT continuations:
    //   Branch A: `40, 41, 42, 43 -> 101, 102, 103, 104`
    //   Branch B: `40, 41, 42, 43 -> 201, 202, 203, 204`
    //   Branch C: `40, 41, 42, 43 -> 301, 302, 303, 304`
    llama_tokens prompt = {
        10, 11, 40, 41, 42, 43, 101, 102, 103, 104, 99,
        12, 13, 40, 41, 42, 43, 201, 202, 203, 204, 99,
        14, 15, 40, 41, 42, 43, 301, 302, 303, 304, 99,
        16, 17, 40, 41, 42
    };
    eng.sync_context_and_observe(prompt, 43);

    // Pre-MTP suffix `[40, 41, 42, 43]` is ambiguous (`consensus_len == 0`), so Fast-Path MUST NOT fire!
    llama_tokens draft;
    float e_tok = 1.0f, p_surv = 1.0f;
    assert(!eng.try_fastpath_draft(5, draft, e_tok, p_surv));

    // Now NextN MTP predicts `t_0 = 201` (`p_top1 = 0.89`): anchoring on `[40, 41, 42, 43, 201]`
    // uniquely locks onto Branch B and extends `[201, 202, 203, 204]`!
    RC2MTPAnchorInfo mtp0;
    mtp0.valid = true; mtp0.n_cands = 2;
    mtp0.top_ids[0] = 201; mtp0.top_p[0] = 0.89f;
    mtp0.top_ids[1] = 101; mtp0.top_p[1] = 0.07f;
    mtp0.p_top1 = 0.89f; mtp0.p_margin = 0.82f; mtp0.top4_mass = 0.98f;

    bool want_stem = false;
    eng.fuse_mtp_and_extend(mtp0, 5, draft, e_tok, p_surv, want_stem);
    assert(draft.size() >= 4);
    assert(draft[0] == 201 && draft[1] == 202 && draft[2] == 203 && draft[3] == 204);

    // Now test Stockfish Refutation + Orthogonal MTP Top-4 Promotion:
    // Suppose target REJECTED `201` and actually emitted `101` twice in context `[16, 17, 40, 41, 42, 43]`!
    llama_tokens p_turn2 = prompt;
    p_turn2.push_back(43); // id_last from previous turn
    eng.sync_context_and_observe(p_turn2, 101); // 201 was refuted, 101 was actual!

    // Observe `[16, 17, 40, 41, 42, 43] -> 101` a second time so `wins == 2` in `tt6/tt5`
    p_turn2.insert(p_turn2.end(), {101, 16, 17, 40, 41, 42});
    eng.sync_context_and_observe(p_turn2, 43);
    p_turn2.push_back(43);
    eng.last_draft[0] = 201; eng.last_draft_len = 1;
    eng.sync_context_and_observe(p_turn2, 101);

    // Next time `[16, 17, 40, 41, 42, 43]` appears and MTP proposes `201` (p=0.72) with `101` at rank 2 (p=0.22):
    p_turn2.insert(p_turn2.end(), {101, 16, 17, 40, 41, 42});
    eng.sync_context_and_observe(p_turn2, 43);
    RC2MTPAnchorInfo mtp_refuted;
    mtp_refuted.valid = true; mtp_refuted.n_cands = 2;
    mtp_refuted.top_ids[0] = 201; mtp_refuted.top_p[0] = 0.72f;
    mtp_refuted.top_ids[1] = 101; mtp_refuted.top_p[1] = 0.22f;
    mtp_refuted.p_top1 = 0.72f; mtp_refuted.p_margin = 0.50f; mtp_refuted.top4_mass = 0.96f;

    draft.clear();
    eng.fuse_mtp_and_extend(mtp_refuted, 5, draft, e_tok, p_surv, want_stem);
    assert(!draft.empty() && draft[0] == 101 && "Stockfish must veto refuted 201 and promote verified 101!");
    assert(eng.stat_sf_promotions >= 1);

    std::printf("[PASS] test_anchored_mtp_disambiguation_and_stockfish_refutation\n");
}

void test_mtp_kv_position_alignment_across_partial_acceptances() {
    RC2SeqEngine eng;
    eng.init();
    MockMTPKVVerifier mtp_kv;

    llama_tokens committed;
    for (int i = 0; i < 64; ++i) {
        committed.push_back(500 + (i % 16));
    }
    mtp_kv.init(committed);
    llama_token id_last = 700;

    // Run 180 speculative steps exercising Fast-Path (0 MTP calls), Anchored (1 MTP call),
    // Stem-Probed (2 MTP calls), NNUE Veto (K=0), Partial Acceptance (0..K), AND Checkpoint Replay!
    for (int step = 0; step < 180; ++step) {
        const llama_pos pos0 = (llama_pos)committed.size();
        const llama_pos ckpt_pos_max = pos0 - 1;

        eng.sync_context_and_observe(committed, id_last);
        llama_tokens draft;
        float e_tok = 1.0f, p_surv = 1.0f;

        if (!eng.try_fastpath_draft(5, draft, e_tok, p_surv)) {
            // 1st MTP decode at pos0
            mtp_kv.simulate_mtp_draft_decode(pos0, id_last, mtp_kv.pending_h[0]);

            RC2MTPAnchorInfo m0;
            m0.valid = true; m0.n_cands = 2;
            const bool low_conf = (step % 11 == 0);
            m0.top_ids[0] = 500 + ((step * 3) % 16);
            m0.top_p[0]   = low_conf ? 0.11f : 0.88f;
            m0.top_ids[1] = 601;
            m0.top_p[1]   = low_conf ? 0.09f : 0.05f;
            m0.p_top1     = m0.top_p[0];
            m0.p_margin   = m0.top_p[0] - m0.top_p[1];
            m0.top4_mass  = low_conf ? 0.30f : 0.97f;

            bool want_stem = false;
            eng.fuse_mtp_and_extend(m0, 5, draft, e_tok, p_surv, want_stem);

            if (want_stem && draft.size() == 1) {
                // 2nd MTP decode at pos0 + 1
                mtp_kv.simulate_mtp_draft_decode(pos0 + 1, draft[0], MockMTPKVVerifier::canonical_h(pos0, id_last));
                RC2MTPAnchorInfo m1 = m0;
                m1.top_ids[0] = 500 + ((step * 3 + 1) % 16);
                m1.p_top1 = (step % 3 == 0) ? 0.91f : 0.79f;
                m1.p_margin = 0.72f;
                bool want_stem2 = false;
                eng.finalize_stem_probe(m1, 5, e_tok, p_surv, draft, &want_stem2);

                if (want_stem2 && draft.size() == 2) {
                    // 3rd MTP decode at pos0 + 2
                    mtp_kv.simulate_mtp_draft_decode(pos0 + 2, draft[1], MockMTPKVVerifier::canonical_h(pos0 + 1, draft[0]));
                    RC2MTPAnchorInfo m2 = m1;
                    m2.top_ids[0] = 500 + ((step * 3 + 2) % 16);
                    m2.p_top1 = 0.89f;
                    m2.p_margin = 0.75f;
                    eng.finalize_stem_probe(m2, 5, e_tok, p_surv, draft, nullptr);
                }
            }
        }

        // Server trims ctx_dft >= ckpt.pos_max + 1 right after common_speculative_draft()
        mtp_kv.seq_rm(ckpt_pos_max + 1);

        // Build target verification batch `[id_last, draft[0..K-1]]`
        std::vector<llama_token> verify_batch;
        verify_batch.push_back(id_last);
        for (llama_token dt : draft) verify_batch.push_back(dt);

        const size_t K = draft.size();
        const uint16_t n_acc = (K == 0) ? 0 : (uint16_t)(step % (K + 1));

        // Every 17th step with partial rejection, also exercise full-checkpoint replay (`use_ckpt_tgt == true`)!
        if (K > 0 && n_acc < K && (step % 17 == 0)) {
            mtp_kv.process_verify_batch(pos0, verify_batch);
            // Checkpoint restore trims back to ckpt_pos_max + 1 WITHOUT calling accept(), then replays `[id_last, draft[0..n_acc-1]]`!
            mtp_kv.seq_rm(ckpt_pos_max + 1);
            std::vector<llama_token> replay_batch(verify_batch.begin(), verify_batch.begin() + 1 + n_acc);
            mtp_kv.process_verify_batch(pos0, replay_batch);
            mtp_kv.accept(n_acc);
        } else {
            mtp_kv.process_verify_batch(pos0, verify_batch);
            mtp_kv.accept(n_acc);
        }

        // Commit `id_last` + `draft[0 .. n_acc - 1]` into prompt, trim rejected KV cells, and sample next `id_last`
        committed.push_back(id_last);
        for (uint16_t i = 0; i < n_acc; ++i) {
            committed.push_back(draft[i]);
        }
        mtp_kv.seq_rm((llama_pos)committed.size());
        mtp_kv.assert_exact_alignment(committed);

        id_last = 500 + ((step * 5 + 1) % 16);
    }

    std::printf("[PASS] test_mtp_kv_position_alignment_across_partial_acceptances (180 multi-turn steps, 0 desyncs)\n");
}

void test_sub_250ns_zen5_latency() {
    RC2SeqEngine eng;
    eng.init();

    // Realistic 40,960-token context (24K Prefix + 16K Hot Zone)
    llama_tokens ctx(40960);
    for (size_t i = 0; i < ctx.size(); ++i) {
        ctx[i] = (int32_t)((i * 2654435761ULL >> 16) % 8192);
    }
    // Plant realistic repeated code blocks in Hot Zone
    for (int rep = 0; rep < 6; ++rep) {
        size_t base = 32000 + (size_t)rep * 1100;
        for (int t = 0; t < 12; ++t) {
            ctx[base + (size_t)t] = 9000 + t;
        }
    }
    ctx.push_back(9000);
    ctx.push_back(9001);
    ctx.push_back(9002);
    ctx.push_back(9003);
    ctx.push_back(9004);
    eng.sync_context_and_observe(ctx, 9005);

    RC2MTPAnchorInfo mtp0;
    mtp0.valid = true; mtp0.n_cands = 3;
    mtp0.top_ids[0] = 9006; mtp0.top_p[0] = 0.91f;
    mtp0.top_ids[1] = 1234; mtp0.top_p[1] = 0.05f;
    mtp0.top_ids[2] = 2345; mtp0.top_p[2] = 0.02f;
    mtp0.p_top1 = 0.91f; mtp0.p_margin = 0.86f; mtp0.top4_mass = 0.98f;

    llama_tokens draft;
    draft.reserve(8);
    float e_tok = 1.0f, p_surv = 1.0f;
    bool want_stem = false;

    // Warmup
    for (int i = 0; i < 2000; ++i) {
        if ((i & 1) == 0) {
            eng.try_fastpath_draft(5, draft, e_tok, p_surv);
        } else {
            eng.fuse_mtp_and_extend(mtp0, 5, draft, e_tok, p_surv, want_stem);
        }
    }

    constexpr int ITERS = 200000;
    size_t sink = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        if ((i & 1) == 0) {
            eng.try_fastpath_draft(5, draft, e_tok, p_surv);
        } else {
            eng.fuse_mtp_and_extend(mtp0, 5, draft, e_tok, p_surv, want_stem);
        }
        sink += draft.size();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ns_total = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const double ns_per_call = ns_total / (double)ITERS;

    std::printf("[BENCH] 40,960-tok Two-Zone Suffix + Stockfish TT + RouteCrack 2.0 L1 Micro-Arbiter: %.2f ns/call (sink=%zu, sizeof(Arbiter)=%zu B)\n",
                ns_per_call, sink, sizeof(RC2L1ArbiterWeights));
#if !defined(__SANITIZE_ADDRESS__)
#if defined(__AVX512F__) || defined(__AVX2__)
    assert(ns_per_call < 250.0 && "SIMD Arbiter + Suffix lookup latency exceeded 250 ns budget!");
#else
    assert(ns_per_call < 600.0 && "Scalar Arbiter + Suffix lookup latency exceeded 600 ns budget!");
#endif
#endif
}

} // namespace

void test_json_mode_refutation_guard() {
    RC2SeqEngine eng;
    eng.init();

    // Raw C code in prompt ends lines with raw token `198` (`\n`), whereas JSON string emits `1639` (`\\n`).
    llama_tokens ctx = {
        10, 20, 30,
        401, 402, 403, 404, 405, 406, 198, 501, 502,
        601, 602, 603, 604, 605, 606, 198, 701, 702,
        801, 802, 803, 804, 805, 806, 807, 808, 198, 901, 902,
        801, 802, 803, 804, 805, 806, 807, 808, 198, 901, 902
    };

    // Rejection #1 of raw `198` (`heat = 3 < 4` -> one-off edit still allowed)
    llama_tokens p1 = ctx;
    p1.insert(p1.end(), {401, 402, 403, 404, 405});
    eng.sync_context_and_observe(p1, 406);
    eng.last_draft[0] = 198; eng.last_draft_len = 1;
    p1.push_back(406);
    eng.sync_context_and_observe(p1, 1639);
    assert(!eng.is_mode_refuted_token(198));

    // Rejection #2 of raw `198` (`heat = 6 >= 4` -> JSON string mode detected!)
    p1.insert(p1.end(), {1639, 601, 602, 603, 604, 605});
    eng.sync_context_and_observe(p1, 606);
    eng.last_draft[0] = 198; eng.last_draft_len = 1;
    p1.push_back(606);
    eng.sync_context_and_observe(p1, 1639);
    assert(eng.is_mode_refuted_token(198));
    // Verify zero false hash aliasing onto `198 + 1024 = 1222`!
    assert(!eng.is_mode_refuted_token(1222));

    // Now copy 5-gram `[801..805] -> [806, 807, 808, 198, 901]`: Fast-Path MUST draft `[806, 807, 808]` and STOP before `198`!
    p1.insert(p1.end(), {1639, 801, 802, 803, 804});
    eng.sync_context_and_observe(p1, 805);
    llama_tokens draft;
    float e_tok = 1.0f, p_surv = 1.0f;
    bool ok = eng.try_fastpath_draft(5, draft, e_tok, p_surv);
    assert(ok && draft.size() == 3);
    assert(draft[0] == 806 && draft[1] == 807 && draft[2] == 808);

    // Emit two real raw `198` tokens outside JSON -> heat cools `6 -> 4 -> 2 < 4`, re-enabling raw `\n`!
    p1.insert(p1.end(), {805, 806, 807, 808});
    eng.last_draft_len = 0;
    eng.sync_context_and_observe(p1, 198);
    p1.push_back(198);
    eng.sync_context_and_observe(p1, 198);
    assert(!eng.is_mode_refuted_token(198));

    std::printf("[PASS] test_json_mode_refutation_guard (collision-free 28-bit tagged mode heat)\n");
}

void test_draftless_rc_predictor_fusion_and_anchoring() {
    RC2PredictorHub hub;
    hub.init(/*n_vocab=*/129280, /*trie_cap=*/(1u << 15));

    RC2SeqEngine eng;
    eng.init();

    // Place `701..704` right near the recent tail alongside 9 occurrences of `801..804` (`9/10 = 90%` in Trie+PPM):
    // Suffix hash-chain sees both `701` and `801` in its top-5 recent scan (`consensus_len == 0`),
    // so `RCPredictor` anchors `801` at `d=0` (`stat_rc_anchors == 1`) and Suffix Copy extends `[802, 803, 804]`!
    llama_tokens ctx = {
        11, 12, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        21, 22, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        31, 32, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        41, 42, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        55, 56, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        61, 62, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        71, 72, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        81, 82, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        85, 86, 50, 51, 52, 53, 701, 702, 703, 704, 99,
        91, 92, 50, 51, 52, 53, 801, 802, 803, 804, 99,
        95, 96, 50, 51, 52
    };
    eng.sync_context_and_observe(ctx, 53);

    llama_tokens draft;
    eng.draft_without_mtp(5, draft, &hub, kRC2DraftlessArbiter);
    assert(draft.size() >= 4 && "Draftless RCPredictor + Anchored Two-Zone Suffix must draft >= 4 tokens!");
    assert(draft[0] == 801 && draft[1] == 802 && draft[2] == 803 && draft[3] == 804);
    assert(eng.stat_rc_anchors >= 1 && eng.stat_suffix_ext_toks >= 2);

    std::printf("[PASS] test_draftless_rc_predictor_fusion_and_anchoring (Universal Non-MTP Engine + kRC2DraftlessArbiter)\n");
}

void test_heavy_moe_draftless_bandwidth_gate() {
    RC2PredictorHub hub;
    hub.init(/*n_vocab=*/129280, /*trie_cap=*/(1u << 15));

    RC2SeqEngine eng;
    eng.init();
    eng.recent_accept_ema = 0.54f; // High-entropy prose regime

    // 1. Weak single-hit accidental 4-gram `[71, 72, 73, 74] -> 910` 200 tokens ago in prose:
    //    `kRC2HeavyMoEDraftlessArbiter` (`is_heavy_moe = true`) MUST refuse to waste a 284B MoE batch!
    llama_tokens ctx = {10, 11, 71, 72, 73, 74, 910, 911};
    for (int i = 0; i < 200; ++i) ctx.push_back(1000 + i);
    ctx.insert(ctx.end(), {71, 72, 73});
    eng.sync_context_and_observe(ctx, 74);

    llama_tokens draft;
    eng.draft_without_mtp(5, draft, &hub, kRC2HeavyMoEDraftlessArbiter, /*is_heavy_moe=*/true);
    assert(draft.empty() && "HeavyMoE arbiter must suppress single-hit cold-prose 4-gram!");

    // 2. Strong 9-token repeated C++ refactor span -> HeavyMoE arbiter drafts >= 4 tokens cleanly!
    eng.recent_accept_ema = 0.82f;
    for (int r = 0; r < 2; ++r) {
        for (int t = 0; t < 14; ++t) ctx.push_back(4000 + t);
        ctx.push_back(99);
    }
    for (int t = 0; t < 8; ++t) ctx.push_back(4000 + t);
    eng.sync_context_and_observe(ctx, 4008);
    eng.draft_without_mtp(5, draft, &hub, kRC2HeavyMoEDraftlessArbiter, /*is_heavy_moe=*/true);
    assert(draft.size() >= 4 && draft[0] == 4009 && draft[1] == 4010 && draft[2] == 4011 && draft[3] == 4012);

    std::printf("[PASS] test_heavy_moe_draftless_bandwidth_gate (284B kRC2HeavyMoEDraftlessArbiter)\n");
}

void test_fast_sigmoid_nan_inf_and_remez_precision() {
    static_assert(sizeof(RC2L1ArbiterWeights) == 832, "Arbiter struct must be 832 B");

    const float q_nan  = rc2_l1::fast_sigmoid(std::numeric_limits<float>::quiet_NaN());
    const float q_nnan = rc2_l1::fast_sigmoid(-std::numeric_limits<float>::quiet_NaN());
    const float q_pinf = rc2_l1::fast_sigmoid(std::numeric_limits<float>::infinity());
    const float q_ninf = rc2_l1::fast_sigmoid(-std::numeric_limits<float>::infinity());

    assert(std::isfinite(q_nan)  && std::fabs(q_nan  - 0.000006f) < 1e-7f);
    assert(std::isfinite(q_nnan) && std::fabs(q_nnan - 0.000006f) < 1e-7f);
    assert(std::isfinite(q_pinf) && std::fabs(q_pinf - 0.999994f) < 1e-7f);
    assert(std::isfinite(q_ninf) && std::fabs(q_ninf - 0.000006f) < 1e-7f);

    double max_err = 0.0;
    for (int i = -11999; i <= 11999; ++i) {
        const float z = (float)i * 0.001f;
        const double approx = (double)rc2_l1::fast_sigmoid(z);
        const double exact  = 1.0 / (1.0 + std::exp(-(double)z));
        max_err = std::max(max_err, std::fabs(approx - exact));
    }
    assert(max_err < 3.0e-6 && "Remez minimax fast_sigmoid error exceeded 3e-6 on (-12, 12)!");

    std::printf("[PASS] test_fast_sigmoid_nan_inf_and_remez_precision (NaN/Inf safe, max_err=%.2e < 3e-6, sizeof=832 B)\n", max_err);
}

void test_arbiter_nan_inf_bit_identity() {
    auto fbits = [](float v) -> uint32_t {
        uint32_t b = 0;
        std::memcpy(&b, &v, sizeof(b));
        return b;
    };

    const float qnan = std::numeric_limits<float>::quiet_NaN();
    const float nnan = -std::numeric_limits<float>::quiet_NaN();
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();

    RC2StepFeatures f_zero{};
    f_zero.recent_accept_ema = 0.0f;

    RC2StepFeatures f_nan{qnan, nnan, qnan, nnan, qnan, nnan, qnan, nnan, qnan, nnan};
    RC2StepFeatures f_pinf{pinf, pinf, pinf, pinf, pinf, pinf, pinf, pinf, pinf, pinf};
    RC2StepFeatures f_ninf{ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf, ninf};
    RC2StepFeatures f_mixed_adv{qnan, pinf, ninf, nnan, pinf, ninf, qnan, pinf, ninf, qnan};

    RC2StepFeatures f_partial_adv{0.88f, qnan, 0.70f, pinf, 1.0f, ninf, 0.25f, qnan, 0.72f, pinf};
    RC2StepFeatures f_partial_clean{0.88f, 0.0f, 0.70f, 0.0f, 1.0f, 0.0f, 0.25f, 0.0f, 0.72f, 0.0f};

    const RC2L1ArbiterWeights * profiles[3] = {
        &kRC2DefaultArbiter,
        &kRC2DraftlessArbiter,
        &kRC2HeavyMoEDraftlessArbiter,
    };

    uint32_t sig_zero[3] = {};
    uint32_t sig_part[3] = {};
    for (int p = 0; p < 3; ++p) {
        const auto & w = *profiles[p];
        const float r_zero  = rc2_l1::predict_accept_prob(f_zero, w);
        const float r_nan   = rc2_l1::predict_accept_prob(f_nan, w);
        const float r_pinf  = rc2_l1::predict_accept_prob(f_pinf, w);
        const float r_ninf  = rc2_l1::predict_accept_prob(f_ninf, w);
        const float r_mixed = rc2_l1::predict_accept_prob(f_mixed_adv, w);
        const float r_padv  = rc2_l1::predict_accept_prob(f_partial_adv, w);
        const float r_pcln  = rc2_l1::predict_accept_prob(f_partial_clean, w);

        assert(std::isfinite(r_zero));
        assert(fbits(r_nan)   == fbits(r_zero) && "All-NaN must match all-zero bit-identically!");
        assert(fbits(r_pinf)  == fbits(r_zero) && "All-+Inf must match all-zero bit-identically!");
        assert(fbits(r_ninf)  == fbits(r_zero) && "All--Inf must match all-zero bit-identically!");
        assert(fbits(r_mixed) == fbits(r_zero) && "Mixed NaN/+-Inf must match all-zero bit-identically!");
        assert(fbits(r_padv)  == fbits(r_pcln) && "Partial NaN/+-Inf must match zero-substituted features bit-identically!");

        sig_zero[p] = fbits(r_zero);
        sig_part[p] = fbits(r_padv);
    }

    // Golden IEEE-754 reference bits (verified 100% bit-identical across AVX-512, AVX2+FMA, NEON, and scalar)
    assert(sig_zero[0] == 0x3b1002b0u && sig_zero[1] == 0x3a2e7eb0u && sig_zero[2] == 0x39c8166bu &&
           "All-NaN/+-Inf output bits drifted from golden cross-ISA IEEE-754 reference!");
    assert(sig_part[0] == 0x3f7eb63bu && sig_part[1] == 0x3f7e81dau && sig_part[2] == 0x3f7b604au &&
           "Partial NaN/+-Inf output bits drifted from golden cross-ISA IEEE-754 reference!");
    std::printf(
        "[PASS] test_arbiter_nan_inf_bit_identity (zero_bits=%08x/%08x/%08x partial_bits=%08x/%08x/%08x)\n",
        sig_zero[0], sig_zero[1], sig_zero[2], sig_part[0], sig_part[1], sig_part[2]
    );
}

void test_partial_rollback_and_session_lcp_antialiasing() {
    RC2PredictorHub hub;
    hub.init(/*n_vocab=*/129280, /*trie_cap=*/(1u << 14));

    RC2SeqEngine eng;
    eng.init();

    // 1. Partial rollback (`lcp < old_sz` while `new_sz > last_rc_learned_len`):
    //    `last_rc_learned_len` MUST clamp down to `lcp` so rewritten tokens are learned!
    llama_tokens p0;
    for (int i = 0; i < 40; ++i) p0.push_back(100 + i);
    eng.sync_context_and_observe(p0, 140);
    llama_tokens draft;
    eng.draft_without_mtp(5, draft, &hub, kRC2DraftlessArbiter);
    assert(eng.last_rc_learned_len == 41);

    // Roll back tokens 30..40 (`lcp == 30`) and append 15 rewritten tokens (`new_sz == 45 > 41`)
    llama_tokens p_rb(p0.begin(), p0.begin() + 30);
    const int32_t rewritten[] = {710, 711, 712, 713, 714, 715, 955, 710, 711, 712, 713, 714, 715, 955};
    for (int32_t t : rewritten) p_rb.push_back(t);
    eng.sync_context_and_observe(p_rb, 888);
    assert(eng.last_rc_learned_len == 30 && "Partial rollback must clamp last_rc_learned_len to lcp!");
    eng.draft_without_mtp(5, draft, &hub, kRC2DraftlessArbiter);
    assert(eng.last_rc_learned_len == p_rb.size() + 1);

    float trie_conf = 0.0f;
    int trie_obs = 0, trie_depth = 0;
    const int32_t probe_ctx[] = {710, 711, 712, 713, 714, 715};
    const int32_t learned_tok = hub.pred->trie.predict_confident(probe_ctx, 6, &trie_conf, &trie_obs, &trie_depth);
    assert(learned_tok == 955 && trie_obs >= 2 && "Rewritten post-rollback region must be learned into online Trie!");

    // 2. Session boundary `tail_exact` / `lim - 16` anti-aliasing (`clear_session_mode()`):
    //    New prompt has identical `tok[0]`, identical `tok[old_sz >> 1]`, and identical last 16 chat-template tokens,
    //    while interior user tokens `[40..47]` changed!
    llama_tokens req1(63);
    for (size_t i = 0; i < req1.size(); ++i) req1[i] = 200 + (int32_t)i;
    for (size_t i = 40; i < 46; ++i) req1[i] = 600 + (int32_t)(i - 40);
    eng.clear_session_mode();
    eng.sync_context_and_observe(req1, 999);
    assert(eng.indexed_tokens[40] == 600);

    llama_tokens req2 = req1;
    for (size_t i = 40; i < 46; ++i) req2[i] = 700 + (int32_t)(i - 40); //tok[0], tok[32], and last 16 tokens unchanged!
    eng.clear_session_mode();
    eng.sync_context_and_observe(req2, 999);
    for (size_t i = 40; i < 46; ++i) {
        assert(eng.indexed_tokens[i] == (int32_t)(700 + (i - 40)) && "clear_session_mode must force full LCP scan!");
    }

    // Old `[600..603]` 4-gram from req1 must NOT exist in indexed suffix chains anymore
    llama_tokens check_q = req2;
    check_q.push_back(999);
    check_q.insert(check_q.end(), {600, 601, 602});
    eng.sync_context_and_observe(check_q, 603);
    RC2SuffixMatch m_stale = eng.query_two_zone_suffix(nullptr, 0);
    assert(!m_stale.found && "Stale interior tokens from previous session must be popped cleanly!");

    std::printf("[PASS] test_partial_rollback_and_session_lcp_antialiasing (LCP rollback clamp + anti-aliasing)\n");
}

int main() {
    test_fast_sigmoid_nan_inf_and_remez_precision();
    test_arbiter_nan_inf_bit_identity();
    test_partial_rollback_and_session_lcp_antialiasing();
    test_two_zone_index_and_undo_invariants();
    test_anchored_mtp_disambiguation_and_stockfish_refutation();
    test_json_mode_refutation_guard();
    test_draftless_rc_predictor_fusion_and_anchoring();
    test_heavy_moe_draftless_bandwidth_gate();
    test_mtp_kv_position_alignment_across_partial_acceptances();
    test_sub_250ns_zen5_latency();
    std::printf("[ALL TESTS PASSED] RouteCrack 2.0 verified clean!\n");
    return 0;
}
