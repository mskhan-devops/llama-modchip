/*
 * rc_arbiter_l1.h — RouteCrack 2.0 L1 Micro-Arbiter
 *
 * Cache-line aligned `10 -> 16 -> 1` ClippedReLU horizon & multi-method fusion micro-network
 * (`832 B` per profile = 13 cache lines, L1d-resident) with AVX-512, AVX2+FMA, ARM64 NEON, and
 * portable C++17 scalar execution paths. Evaluates marginal expected `tokens / ms` across:
 *   - `kRC2DefaultArbiter`          : Native NextN MTP + Two-Zone Suffix + Stockfish-PVS (`Qwen3.8`)
 *   - `kRC2DraftlessArbiter`        : Non-MTP Mamba-2 / Hybrid MoE (`NVIDIA-Nemotron-3.5-30B-A3B`)
 *   - `kRC2HeavyMoEDraftlessArbiter`: Non-MTP 256-Expert Ultra-Large MoE (`DeepSeek-V4-Flash-0731` 284B)
 */

#ifndef RC_ARBITER_L1_H
#define RC_ARBITER_L1_H

#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

struct RC2StepFeatures {
    float mtp_p1            = 0.0f; // [0, 1] NextN MTP top-1 prob (or RCPredictor cert/100 in draftless mode)
    float mtp_margin        = 0.0f; // [0, 1] NextN MTP margin (or RCPredictor ensemble specificity in draftless mode)
    float suffix_match_len  = 0.0f; // [0, 1] Two-Zone suffix match length min(1, k / 10)
    float suffix_hit_count  = 0.0f; // [0, 1] True consensus occurrences min(1, hits / 4)
    float zone_is_hot       = 0.0f; // {0, 1} 16K Hot Zone / Session Trie indicator
    float sf_tt_wins        = 0.0f; // [0, 1] Stockfish Zobrist TT verified wins min(1, wins / 4)
    float sf_tt_history     = 0.0f; // [-1, 1] Stockfish History Gravity (history / 512)
    float depth_d           = 0.0f; // [0, 1] Draft depth d / 4
    float recent_accept_ema = 0.72f;// [0, 1] Session EMA acceptance rate
    float is_structural_tok = 0.0f; // [0, 1] Cross-engine agreement / structural indicator
};

struct alignas(64) RC2L1ArbiterWeights {
    static constexpr int IN_DIM  = 10;
    static constexpr int HID_DIM = 16;

    alignas(64) float w1_col[IN_DIM][HID_DIM]; // 640 B (each row is 1x 512-bit ZMM / 4x 128-bit NEON regs)
    alignas(64) float b1[HID_DIM];             // 64 B
    alignas(64) float w2[HID_DIM];             // 64 B
    float b2;
    float cost_conservatism;
    float mtp_pass_cost;
    float rollback_cost;
    float fastpath_q0_min;
    float stem_probe_q0_min;
    float target_batch_cost[6];                // k = 0..5 drafted tokens (batch = 1..6)
};

// Profile 1: MTP-Equipped Flagships (`Qwen3.8-Flash-Next` 64GB & 32GB)
static constexpr RC2L1ArbiterWeights kRC2DefaultArbiter = {
    /* .w1_col = */ {
        { 1.334935f, 0.350000f, 0.850000f, 0.450000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.250000f, 0.000000f, 0.000000f, 0.500000f, -0.950000f, -2.400000f },
        { 0.250000f, 1.350000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.900000f, -2.200000f, 2.084636f, 1.550000f, 0.950000f, 1.850000f, -0.850000f, 0.000000f, -0.300000f, 0.000000f, 1.450000f, 0.500000f, -1.600000f, -2.800000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.300000f, 0.350000f, 1.350000f, 0.300000f, -0.900000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.650000f, 0.000000f, -0.800000f, 0.000000f, 0.200000f, 0.000000f, 0.550000f, 0.350000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.850000f, -1.500000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.800000f, 1.400000f, 0.000000f, 0.650000f, 0.000000f, 0.400000f, -1.200000f, -2.200000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.950000f, -2.200000f, 0.450000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 1.600000f, -0.200000f, -0.250000f, 0.000000f, 0.000000f, 1.300000f, 0.000000f, 0.000000f, 0.000000f, 0.750000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.350000f, -1.450000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.650000f, 0.000000f, 0.000000f, 0.000000f, 0.250000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.950000f, 0.350000f, 0.000000f, 0.000000f, -0.900000f }
    },
    /* .b1 = */ { -0.334935f, -0.180000f, -0.950000f, -0.420000f, -1.084636f, -1.020000f, -0.820000f, -0.880000f, -0.220000f, -0.620000f, -0.080000f, -0.850000f, -0.880000f, -1.100000f, 1.050000f, 0.920000f },
    /* .w2 = */ { 1.830700f, 1.432600f, 0.911800f, -0.994800f, 3.147800f, 1.723600f, 1.213700f, 1.416600f, -0.666500f, 1.682000f, -3.600000f, 1.422600f, 1.928000f, 0.850000f, -1.621600f, -4.090400f },
    /* .b2 = */ -0.733500f,
    /* .cost_conservatism = */ 1.03760f,
    /* .mtp_pass_cost     = */ 0.13000f,
    /* .rollback_cost     = */ 0.02500f,
    /* .fastpath_q0_min   = */ 0.89500f,
    /* .stem_probe_q0_min = */ 0.85500f,
    /* .target_batch_cost = */ { 1.035000f, 1.330000f, 1.620000f, 1.905000f, 2.185000f, 2.460000f }
};

// Profile 2: Medium/Fast Non-MTP Flagships (`NVIDIA-Nemotron-3.5-30B-A3B` 16GB)
static constexpr RC2L1ArbiterWeights kRC2DraftlessArbiter = {
    /* .w1_col = */ {
        { 2.693240f, 0.350000f, 0.850000f, 0.450000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.250000f, 0.000000f, 0.000000f, 0.500000f, -0.950000f, -2.400000f },
        { 0.250000f, 1.350000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.900000f, -2.200000f, 2.105263f, 1.550000f, 0.950000f, 1.850000f, -0.850000f, 0.000000f, -0.300000f, 0.000000f, 1.450000f, 0.500000f, -1.600000f, -2.800000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.300000f, 0.350000f, 1.350000f, 0.300000f, -0.900000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.650000f, 0.000000f, -0.800000f, 0.000000f, 0.200000f, 0.000000f, 0.550000f, 0.350000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.850000f, -1.500000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.800000f, 1.400000f, 0.000000f, 0.650000f, 0.000000f, 0.400000f, -1.200000f, -2.200000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.950000f, -2.200000f, 0.450000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 1.600000f, -0.200000f, -0.250000f, 0.000000f, 0.000000f, 1.300000f, 0.000000f, 0.000000f, 0.000000f, 0.750000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.350000f, -1.450000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.650000f, 0.000000f, 0.000000f, 0.000000f, 0.250000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.950000f, 0.350000f, 0.000000f, 0.000000f, -0.900000f }
    },
    /* .b1 = */ { -1.693240f, -0.180000f, -0.950000f, -0.420000f, -1.105263f, -1.020000f, -0.820000f, -0.880000f, -0.220000f, -0.620000f, -0.080000f, -0.850000f, -0.880000f, -1.100000f, 1.050000f, 0.920000f },
    /* .w2 = */ { 2.191400f, 1.187900f, 1.525700f, -2.312700f, 3.102300f, 2.042000f, 1.510700f, 1.522000f, -1.056300f, 2.150000f, -4.665400f, 0.962100f, 1.956500f, 0.615700f, -2.661800f, -3.851400f },
    /* .b2 = */ -1.109000f,
    /* .cost_conservatism = */ 1.05200f,
    /* .mtp_pass_cost     = */ 0.00000f,
    /* .rollback_cost     = */ 0.03000f,
    /* .fastpath_q0_min   = */ 0.00000f,
    /* .stem_probe_q0_min = */ 1.00000f,
    /* .target_batch_cost = */ { 1.000000f, 1.315000f, 1.625000f, 1.930000f, 2.230000f, 2.525000f }
};

// Profile 3: Bandwidth-Bound Ultra-Large MoE Flagships (`DeepSeek-V4-Flash-0731` 284B, > 35 GiB)
static constexpr RC2L1ArbiterWeights kRC2HeavyMoEDraftlessArbiter = {
    /* .w1_col = */ {
        { 2.898551f, 0.350000f, 0.850000f, 0.450000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.250000f, 0.000000f, 0.000000f, 0.500000f, -0.950000f, -2.400000f },
        { 0.250000f, 1.350000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.900000f, -2.200000f, 2.399232f, 1.550000f, 0.950000f, 1.850000f, -0.850000f, 0.000000f, -0.300000f, 0.000000f, 1.450000f, 0.500000f, -1.600000f, -2.800000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.300000f, 0.350000f, 1.350000f, 0.300000f, -0.900000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.650000f, 0.000000f, -0.800000f, 0.000000f, 0.200000f, 0.000000f, 0.550000f, 0.350000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.850000f, -1.500000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, -0.800000f, 1.400000f, 0.000000f, 0.650000f, 0.000000f, 0.400000f, -1.200000f, -2.200000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.950000f, -2.200000f, 0.450000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 1.600000f, -0.200000f, -0.250000f, 0.000000f, 0.000000f, 1.300000f, 0.000000f, 0.000000f, 0.000000f, 0.750000f, 0.000000f, 0.000000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 1.350000f, -1.450000f, 0.000000f },
        { 0.000000f, 0.000000f, 0.650000f, 0.000000f, 0.000000f, 0.000000f, 0.250000f, 0.000000f, 0.000000f, 0.000000f, 0.000000f, 0.950000f, 0.350000f, 0.000000f, 0.000000f, -0.900000f }
    },
    /* .b1 = */ { -1.898551f, -0.180000f, -0.950000f, -0.420000f, -1.399232f, -1.020000f, -0.820000f, -0.880000f, -0.220000f, -0.620000f, -0.080000f, -0.850000f, -0.880000f, -1.100000f, 1.050000f, 0.920000f },
    /* .w2 = */ { 1.625200f, 1.460000f, 1.951400f, -1.981800f, 2.974600f, 1.273200f, 1.579900f, 1.490200f, -1.651900f, 2.250000f, -4.753300f, 1.610000f, 1.915700f, 0.635500f, -2.750300f, -4.283000f },
    /* .b2 = */ -1.180000f,
    /* .cost_conservatism = */ 1.08440f,
    /* .mtp_pass_cost     = */ 0.00000f,
    /* .rollback_cost     = */ 0.03200f,
    /* .fastpath_q0_min   = */ 0.00000f,
    /* .stem_probe_q0_min = */ 1.00000f,
    /* .target_batch_cost = */ { 1.000000f, 1.395000f, 1.785000f, 2.170000f, 2.550000f, 2.925000f }
};

static_assert(sizeof(RC2L1ArbiterWeights) == 832, "RouteCrack 2.0 L1 Micro-Arbiter must be 832 B (13 cache lines)");

namespace rc2_l1 {

static inline float sanitize_feature(float v) noexcept {
    return std::isfinite(v) ? v : 0.0f;
}

static inline float fast_sigmoid(float z) noexcept {
    if (!(z > -12.0f)) return 0.000006f; // catches z <= -12.0f and NaN without UB
    if (z >= 12.0f)    return 0.999994f;
    // IEEE-754 range-reduced degree-5 Remez minimax 2^f on [0, 1] (< 3e-6 max error, zero libm call)
    const float x = -z * 1.4426950408889634f; // -z * log2(e)
    const int32_t xi = (int32_t)std::floor(x);
    const float f = x - (float)xi;
    const float p = 1.0f + f * (0.69315309f + f * (0.24015362f + f * (0.05582825f + f * (0.00898898f + f * (0.00187682f)))));
    const uint32_t bits = ((uint32_t)(xi + 127)) << 23;
    float scale;
    std::memcpy(&scale, &bits, sizeof(float));
    return 1.0f / (1.0f + p * scale);
}

static inline float predict_accept_prob(
    const RC2StepFeatures & f,
    const RC2L1ArbiterWeights & w = kRC2DefaultArbiter) noexcept {

    // Pre-sanitize all 10 input features (NaN / +-Inf -> 0.0f) so AVX-512, AVX2+FMA,
    // ARM64 NEON, and scalar paths produce bit-identical results on adversarial inputs.
    const float x[RC2L1ArbiterWeights::IN_DIM] = {
        sanitize_feature(f.mtp_p1),
        sanitize_feature(f.mtp_margin),
        sanitize_feature(f.suffix_match_len),
        sanitize_feature(f.suffix_hit_count),
        sanitize_feature(f.zone_is_hot),
        sanitize_feature(f.sf_tt_wins),
        sanitize_feature(f.sf_tt_history),
        sanitize_feature(f.depth_d),
        sanitize_feature(f.recent_accept_ema),
        sanitize_feature(f.is_structural_tok)
    };

#if defined(__AVX512F__)
    __m512 acc0 = _mm512_loadu_ps(w.b1);
    __m512 acc1 = _mm512_setzero_ps();
    for (int j = 0; j < RC2L1ArbiterWeights::IN_DIM; j += 2) {
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(x[j]),     _mm512_loadu_ps(w.w1_col[j]),     acc0);
        acc1 = _mm512_fmadd_ps(_mm512_set1_ps(x[j + 1]), _mm512_loadu_ps(w.w1_col[j + 1]), acc1);
    }
    const __m512 acc  = _mm512_add_ps(acc0, acc1);
    const __m512 zero = _mm512_setzero_ps();
    const __m512 one  = _mm512_set1_ps(1.0f);
    const __m512 h    = _mm512_min_ps(_mm512_max_ps(acc, zero), one);
    const __m512 prod = _mm512_mul_ps(h, _mm512_loadu_ps(w.w2));
    // Exact 16 -> 8 -> 4 -> 2 -> 1 pairwise reduction matching AVX2, NEON, and scalar
    const __m256 p256 = _mm256_add_ps(_mm512_castps512_ps256(prod), _mm512_extractf32x8_ps(prod, 1));
    const __m128 p128 = _mm_add_ps(_mm256_castps256_ps128(p256), _mm256_extractf128_ps(p256, 1));
    const __m128 p64  = _mm_add_ps(p128, _mm_movehl_ps(p128, p128));
    const __m128 p32  = _mm_add_ss(p64, _mm_movehdup_ps(p64));
    const float  z    = w.b2 + _mm_cvtss_f32(p32);
    return fast_sigmoid(z);
#elif defined(__AVX2__) && defined(__FMA__)
    __m256 a0 = _mm256_loadu_ps(w.b1 + 0);
    __m256 a1 = _mm256_loadu_ps(w.b1 + 8);
    for (int j = 0; j < RC2L1ArbiterWeights::IN_DIM; ++j) {
        const __m256 vx = _mm256_set1_ps(x[j]);
        a0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w.w1_col[j] + 0), a0);
        a1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(w.w1_col[j] + 8), a1);
    }
    const __m256 zero = _mm256_setzero_ps();
    const __m256 one  = _mm256_set1_ps(1.0f);
    a0 = _mm256_min_ps(_mm256_max_ps(a0, zero), one);
    a1 = _mm256_min_ps(_mm256_max_ps(a1, zero), one);
    const __m256 p256 = _mm256_add_ps(_mm256_mul_ps(a0, _mm256_loadu_ps(w.w2 + 0)), _mm256_mul_ps(a1, _mm256_loadu_ps(w.w2 + 8)));
    const __m128 p128 = _mm_add_ps(_mm256_castps256_ps128(p256), _mm256_extractf128_ps(p256, 1));
    const __m128 p64  = _mm_add_ps(p128, _mm_movehl_ps(p128, p128));
    const __m128 p32  = _mm_add_ss(p64, _mm_movehdup_ps(p64));
    const float  z    = w.b2 + _mm_cvtss_f32(p32);
    return fast_sigmoid(z);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t a0 = vld1q_f32(w.b1 + 0);
    float32x4_t a1 = vld1q_f32(w.b1 + 4);
    float32x4_t a2 = vld1q_f32(w.b1 + 8);
    float32x4_t a3 = vld1q_f32(w.b1 + 12);
    for (int j = 0; j < RC2L1ArbiterWeights::IN_DIM; ++j) {
        const float xj = x[j];
        if (xj != 0.0f) {
            a0 = vfmaq_n_f32(a0, vld1q_f32(w.w1_col[j] + 0),  xj);
            a1 = vfmaq_n_f32(a1, vld1q_f32(w.w1_col[j] + 4),  xj);
            a2 = vfmaq_n_f32(a2, vld1q_f32(w.w1_col[j] + 8),  xj);
            a3 = vfmaq_n_f32(a3, vld1q_f32(w.w1_col[j] + 12), xj);
        }
    }
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one  = vdupq_n_f32(1.0f);
    a0 = vminq_f32(vmaxq_f32(a0, zero), one);
    a1 = vminq_f32(vmaxq_f32(a1, zero), one);
    a2 = vminq_f32(vmaxq_f32(a2, zero), one);
    a3 = vminq_f32(vmaxq_f32(a3, zero), one);
    const float32x4_t p02  = vaddq_f32(vmulq_f32(a0, vld1q_f32(w.w2 + 0)), vmulq_f32(a2, vld1q_f32(w.w2 + 8)));
    const float32x4_t p13  = vaddq_f32(vmulq_f32(a1, vld1q_f32(w.w2 + 4)), vmulq_f32(a3, vld1q_f32(w.w2 + 12)));
    const float32x4_t p128 = vaddq_f32(p02, p13);
    const float32x2_t p64  = vadd_f32(vget_low_f32(p128), vget_high_f32(p128));
    const float       z    = w.b2 + (vget_lane_f32(p64, 0) + vget_lane_f32(p64, 1));
    return fast_sigmoid(z);
#else
    alignas(64) float h[RC2L1ArbiterWeights::HID_DIM];
    for (int i = 0; i < RC2L1ArbiterWeights::HID_DIM; ++i) {
        h[i] = w.b1[i];
    }
    for (int j = 0; j < RC2L1ArbiterWeights::IN_DIM; ++j) {
        const float xj = x[j];
        if (xj != 0.0f) {
            for (int i = 0; i < RC2L1ArbiterWeights::HID_DIM; ++i) {
                h[i] += xj * w.w1_col[j][i];
            }
        }
    }
    float prod[RC2L1ArbiterWeights::HID_DIM];
    for (int i = 0; i < RC2L1ArbiterWeights::HID_DIM; ++i) {
        const float hi = !(h[i] > 0.0f) ? 0.0f : (h[i] >= 1.0f ? 1.0f : h[i]);
        prod[i] = hi * w.w2[i];
    }
    float s8[8], s4[4], s2[2];
    for (int i = 0; i < 8; ++i) s8[i] = prod[i] + prod[i + 8];
    for (int i = 0; i < 4; ++i) s4[i] = s8[i] + s8[i + 4];
    s2[0] = s4[0] + s4[2];
    s2[1] = s4[1] + s4[3];
    const float z = w.b2 + (s2[0] + s2[1]);
    return fast_sigmoid(z);
#endif
}

static inline float expected_cycle_cost(
    int k,
    int n_mtp_passes,
    float p_all_accepted,
    const RC2L1ArbiterWeights & w = kRC2DefaultArbiter) noexcept {
    const int kc = k < 0 ? 0 : (k > 5 ? 5 : k);
    const float rb = (kc > 0) ? w.rollback_cost * (1.0f - p_all_accepted) : 0.0f;
    return (float)n_mtp_passes * w.mtp_pass_cost + w.target_batch_cost[kc] + rb;
}

static inline bool should_append_token(
    int d,
    float q_d,
    float & io_e_tok,
    float & io_p_surv,
    int prev_mtp_passes,
    int new_mtp_passes,
    const RC2L1ArbiterWeights & w = kRC2DefaultArbiter) noexcept {

    if (d < 0 || d >= 5) return false;
    const float prev_cost = expected_cycle_cost(d, prev_mtp_passes, io_p_surv, w);
    const float next_surv = io_p_surv * q_d;
    const float next_tok  = io_e_tok + next_surv;
    const float raw_next_cost = expected_cycle_cost(d + 1, new_mtp_passes, next_surv, w);
    const float adj_next_cost = prev_cost + (raw_next_cost - prev_cost) * w.cost_conservatism;

    if (next_tok * prev_cost > io_e_tok * adj_next_cost) {
        io_e_tok  = next_tok;
        io_p_surv = next_surv;
        return true;
    }
    return false;
}

} // namespace rc2_l1

#endif // RC_ARBITER_L1_H
