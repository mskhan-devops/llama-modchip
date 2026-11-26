# NVIDIA Nemotron (`30B-A3B` & `H-8B`) — Optimization Guide

> **Back to:** [Main Overview (`README.md`)](../README.md) · [LUT-VQ3-64 Quality Gate (`benchmarks/QUALITY_GATE_VQ3_64.md`)](../benchmarks/QUALITY_GATE_VQ3_64.md) · [Hardware & Chip Matrix (`docs/HARDWARE_OPTIMIZATIONS.md`)](HARDWARE_OPTIMIZATIONS.md) · [DeepSeek-V4-Flash 284B Guide (`docs/DEEPSEEK_V4_OPTIMIZATIONS.md`)](DEEPSEEK_V4_OPTIMIZATIONS.md) · [Full Paper (`ARCHITECTURE.md`)](../ARCHITECTURE.md)

---

## 1. Architectural Profile: `Nemotron-3.5-30B-A3B` at 256K Context

`NVIDIA-Nemotron-3.5-Lightning-30B-A3B` (`nemotron_h_moe`) is a **52-layer Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE** model (`30.2B` total parameters, **`3.0B` active per token**):
- **23 Mamba-2 State-Space Layers (`SSM`)**: Maintain a constant-size $O(1)$ recurrent state (`18.4 MB` total across all 23 layers) whether context is `1K` or `256K` tokens.
- **6 Grouped-Query Attention Layers (`GQA`, `2` KV heads per layer = `12` KV heads total)**: Only **6 out of 52 layers** allocate a growing KV cache. At **`262,144`-token (`256K`) context**, the KV cache occupies **`384 MB` (`Q4_0`) to `432 MB` (`EAKV Q8_0/Q4_0`)** (`624 MB` with static `-ctk q8_0 -ctv q4_0`) — compared to `7.1–16.0 GB` on standard 48-layer GQA Transformers.
- **23 Sparse MoE Layers (`128` routed experts/layer, `top-6` active + `1` double-width shared expert)**: `~1.4–1.8 GB` of active weights read per token during `N=1` decode.

---

## 2. Summary of Optimizations Applied to `Nemotron-3.5-30B-A3B`

| Optimization Layer | Technique Implemented | Target Bottleneck Solved | Measured Gain (`Zen 5` / `Threadripper` / `Vulkan` / `Metal`) |
| :--- | :--- | :--- | :--- |
| **1. Quantization (`LUT-VQ3-64`)** | Block-64 3D Lloyd-Max Vector Quantizer (`GGML_TYPE_VQ3_64 = 45`, `24B / 64w = 3.00 bpw`) via AVX-512 VBMI `VPERMI2B` (`patches/llama-modchip-v3.2.patch`) | Reduces MoE `up_exps` footprint by `-33.3%` vs `Q4_0` and `-20.0%` vs `Fractal-64` without scalar 2–3 bit grid loss (`+3.42 dB` SQNR over `Q2_K`) | **`13.30 GiB` shipped Hybrid (`13.08 GiB` full `up_exps` / `16.04 GiB` RSS)**, **`24.18 tok/s` E2E (`+23.7%` vs `Q4_0`)**, **`70.19%` PVS acc**, **`100/100` Gate** |
| **2. Quantization (`MaxQ-16GB`)** | 100% Donor-Free Standard `ggml` (`max_type = 42`, `0` custom types: `{F32, Q4_0, Q8_0, MXFP4, Q2_0}`) | Universal 16 GB deployment on unmodified upstream `llama.cpp` (`46eb1742d+` / `de32b268c+`) and `Q2_0`-capable runners (Apple Metal, CUDA, Vulkan) | **`13.39 GiB` (`~15.0G` cold `N=1` / `14.304 GiB` warm gate)**, **`22.64 tok/s`** cold cg `N=1` (`188.3` `pp512`), blind `wiki-alt` PPL **`7.31`** |
| **3. Quantization (`Fractal-64`)** | Block-64 2-Level Hierarchical Non-Linear `GGML_TYPE_FRACTAL_64 = 44` (`30B / 64w = 3.75 bpw`) from `BF16` master weights | `1856 × 2688` expert dimensions are not divisible by `256`, causing `Q3_K`/`IQ3_S` in `llama.cpp` to fall back to `Q4_0` | **`17.60 GiB -> 14.36 GiB` (`-6.01 GiB` idle RAM)**, **`+28.0%` CPU decode (`25.02 tok/s`)**, **`78.60 tok/s` M3 Max Metal ([PR #24](https://github.com/synchopate/llama-modchip/pull/24))**, **`100/100` Gate** |
| **4. Quantization (`Fractal-MXFP4`)** | Asymmetric Block-32 `MXFP4` (`4.25 bpw`) on routed MoE experts + `Q8_0`/`Q4_0` on GQA/SSM backbone + `_MM_HINT_NTA` | Bandwidth bottleneck on CPU & VRAM capacity on `16 GB` GPUs | **`15.95 GiB`**, **`28.20 agg / 42.18 W1 tok/s` `RouteCrack 2.0` CPU**, **`75.1–80.8 tok/s` on AMD Vulkan Hybrid** |
| **5. SIMD Kernels (`FRACTAL_64` & `VQ3_64`)** | Single-cycle AVX-512 VBMI `VPERMI2B` (`VQ3_64 = 45`) and AVX-512 `vprorvq` / AVX2 variable bit-shifts fused into `vpmaddwd` (`FRACTAL_64 = 44`) | Scalar bit-unpacking overhead in sub-4-bit hierarchical blocks | **`+55%` `FRACTAL_64` kernel speedup** (`86.5 ms -> 55.8 ms` per 10M blocks), register-resident 384-byte `VQ3_64` codebook |
| **6. Speculative Engine (`RouteCrack 2.0`)** | Unified `--spec-type routecrack2`: `832 B` L1 Micro-Arbiter (`kRC2DraftlessArbiter`) + `RCPredictor` + Two-Zone 40K Suffix + `192 KB` Stockfish-PVS Zobrist TT with 28-bit Tagged Mode Refutation ([`rc_routecrack2.h`](../speculative/cpp/rc_routecrack2.h)) | 1B neural draft models read `3.0 GB` from DDR5 (`40 ms`) and flush L3 cache; eliminates `W2` JSON mode-bleed divergence | **`28.20 agg / 42.18 W1 tok/s`** (`Fractal-MXFP4` ref; `+24.8%` vs `N=1`, `+10.5%` `W2`, `100%` exact `W1+W2`) on 16C Zen 5; up to **`60.4 tok/s`** `M=7` / **`111.62 tok/s`** on Threadripper 7985WX |
| **7. Cache Isolation (`_MM_HINT_NTA`)** | Bidirectional Non-Temporal L3 bypass on cold MoE weights and KV tokens older than `LLAMA_KV_L3_WINDOW=2048` | Streaming `1.4–1.8 GB/tok` of MoE weights evicts hot Mamba-2 state and recent KV tokens from `64 MB` L3 | Keeps `18.4 MB` Mamba-2 state + `4.88 MiB` hot KV window in L3 cache |
| **8. Virtual Memory (`2MB HugePages`)** | `MADV_HUGEPAGE` (`THP=madvise`) on weight mmap + compute arenas + `2MB`-aligned `posix_memalign` | `17.6 GiB` in `4KB` pages = `4.5M` PTEs, thrashing the `3,072`-entry Zen 5 L2 STLB (`14 ms/tok` page-walk stall) | Expands L2 STLB coverage to **`6.14 GB`**, cutting dTLB page-walk latency from `14.0 ms` to **`0.8 ms/tok` (`-99.6%` misses)** |
| **9. Weight Tile Repacking (`q4_0_8x8`)** | In-memory AVX-512 VNNI (`VPDPBUSD`) / AVX2 interleaved `8×8` matrix tile repacking at model load | Cache-line splits and nibble-unpacking stalls during GEMV/GEMM | **`+19%` raw GEMV throughput** and **`241–306.8 tok/s` (`pp512`) prefill** on 16C Zen 5 |
| **10. KV Cache (`EAKV + VPI`)** | Entropy-Guided Asymmetric KV (`Q8_0` Keys for RoPE phase precision, `Q4_0` Values) + Virtual Positional Indexing | FP16 KV cache growth and RoPE degradation at `128K–256K` context | Full **`256K` context in `432 MB`**, `100%` passkey needle retrieval |
| **11. Agentic Loop (`HARP-Cache` Slot-0)** | Per-UID `/dev/shm` memory-mapped KV + Mamba-2 recurrent state snapshot + SSE Gateway auto-restore | Re-evaluating Claude Code's `22.2K`-token 40-tool system prompt takes `130.8 s` on CPU | **`18.9 ms` warm turn restore** |
| **12. Dynamic MoE Pruning (`Top-K 6->4`)** | Opt-in norm-preserving dynamic MoE router truncation (`LLAMA_MOE_TOP_K=4` environment variable at server startup) | Reading 6 routed experts per layer on bandwidth-constrained DDR4/DDR5 | **`-33%` routed MoE DRAM traffic** (opt-in boot-time setting; `v3.2.0` defaults to unpruned `Top-6`) |

---

## 3. Verified Quality & Speed Gate (`100 / 100` Score)

All four quantization profiles (`Baseline-Q4_0`, `Fractal-MXFP4`, `Fractal-64-BF16`, and `LUT-VQ3-64`) pass the automated grading suite [`benchmarks/eval_quality_and_tools.py`](../benchmarks/eval_quality_and_tools.py) (full reports in [`benchmarks/QUALITY_GATE_VQ3_64.md`](../benchmarks/QUALITY_GATE_VQ3_64.md) and [`benchmarks/FRACTAL_MOE_QUALITY_GATE.md`](../benchmarks/FRACTAL_MOE_QUALITY_GATE.md)):

| Metric / Suite ([`benchmarks/eval_quality_and_tools.py`](../benchmarks/eval_quality_and_tools.py)) | `Baseline-Q4_0` (`17.60 GiB`) | `Fractal-MXFP4` (`15.95 GiB`) | `Fractal-64-BF16` (`14.36 GiB`) | **`LUT-VQ3-64` (`13.30` / `13.08 GiB`¹)** |
| :--- | :---: | :---: | :---: | :---: |
| **1. Agentic Tool-Calling (`30` pts — 5 multi-tool JSON calls)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`¹** |
| **2. Executable Python Code Gen (`30` pts — sandbox-tested)** | `30.0 / 30.0` | `30.0 / 30.0` | `30.0 / 30.0` | **`30.0 / 30.0`¹** |
| **3. Long-Context Needle (`8.5K`) & Symbolic Math (`20` pts)** | `20.0 / 20.0` | `20.0 / 20.0` | `20.0 / 20.0` | **`20.0 / 20.0`¹** |
| **Served Gate Decode (`Stockfish-PVS` E2E, Suite 4)** | `19.54 tok/s` *(`66.7%` acc)* | **`26.08 tok/s`** *(`61.1%` acc)* | **`25.02 tok/s`** *(`59.1%` acc)* | **`24.18 tok/s`¹ (`+23.7%` vs `Q4_0`, `70.2%` acc)** |
| **In-Context Refactor Burst (`M=7`) / GPU-Hybrid** | **`40.91–60.40 tok/s`** *(`48.39` avg `M=7`)* | **`75.1–80.8 tok/s`** *(AMD Vulkan Hybrid)* | **`71.3–78.6 tok/s`** *(Apple M3/M4 Max Metal)* | — *(Gate E2E measured on `13.08 GiB` abl.¹)* |
| **Peak Resident RAM (`VmRSS` @ `256K` Context)** | `21.71 GiB` | `18.86 GiB` | `17.27 GiB` | **`16.04 GiB`¹ (`-5.67 GiB` vs `Q4_0`)** |

*(¹ `eval_nemotron_vq3_64.json` [`24.18 tok/s`, `16.04 GiB` `VmRSS`, `100/100` Gate] was recorded on the `13.08 GiB` 23-layer `--all-up-exps` `VQ3` ablation; continuous `WikiText-2` PPL and `BF16` Logit KLD in §3.2 below are measured directly on the shipped `13.30 GiB` Hybrid GGUF.)*

### 3.1 Empirical Paired KL-Divergence & Logit Fidelity (`Q4_0` Reference Baseline)

Independent paired evaluation conducted by `@khairullinr` on an Apple M4 Max (`36 GB` LPDDR5X-8533, Metal) using `llama-perplexity --kl-divergence` on a Python source-code corpus (`12` chunks × `2048` ctx, `~12.3k` scored tokens, `-ngl 99 -fa 1 -b 256 -ub 256`, [Issue #16](https://github.com/synchopate/llama-modchip/issues/16#issuecomment-6752457578)):

- **Reference Model**: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf` (`17.60 GiB`, nominal `4.50 bpw`). Note this measures fidelity relative to `Q4_0`, not unquantized `BF16`.
- **Target Quantization**: `Fractal-64-BF16` (`14.36 GiB`, nominal `3.75 bpw`, `-18.42%` file footprint).
- **Top-1 (Argmax) Token Agreement**: **`94.347 ± 0.208%`** (reported as `Same top p` by `llama-perplexity`). In ~19 out of 20 token positions, `Fractal-64` selects the exact same greedy token as the `Q4_0` baseline.
- **Kullback-Leibler Divergence (KLD)**:
  - **Mean KLD**: **`0.056606 ± 0.001964 nats`**
  - **90th Percentile KLD**: **`0.139480 nats`**
  - **95th Percentile KLD**: **`0.280968 nats`**
  - **99th Percentile KLD**: **`0.845352 nats`**
  - **Median KLD**: **`0.000829 nats`**
  - *Distribution profile*: On the vast majority of tokens, predicted logit distributions are nearly identical (median KLD `< 0.001 nats`), with divergence concentrated in a tail of roughly 5–10% of positions.
- **Per-Token Log-Likelihood Correlation**: **`97.28%`** (`Cor(ln PPL(Q), ln PPL(base))`).
- **Paired Perplexity & Probability Shifts**:
  - **Mean PPL(base)**: `2.082848 ± 0.037092`
  - **Mean PPL(Q)**: `2.123941 ± 0.039133`
  - **Mean Paired $\ln(\text{PPL}(Q)/\text{PPL}(\text{base}))$**: `0.019537 ± 0.004266` (**`+1.97%`**)
  - **Mean Paired $\text{PPL}(Q) - \text{PPL}(\text{base})$**: `0.041092 ± 0.009110`
  - **Median $\Delta p$**: **`-0.000%`** (no systematic probability bias)
  - **RMS $\Delta p$**: **`8.133 ± 0.197%`**
- *Methodological Caveats*: Single Python code corpus (low entropy compared to open prose); tokens within context windows are correlated, so asymptotic variance may exceed i.i.d. estimates.

### 3.2 Direct Uncompressed `BF16` (`58.84 GiB`) `WikiText-2` Perplexity & Multi-Step Logit KLD (`v2.5.4`)

To measure quantization and abliteration fidelity directly against the uncompressed `58.84 GiB` `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` master checkpoint without intermediate `Q4_0` quantization noise, [`benchmarks/eval_perplexity_and_kld.py`](../benchmarks/eval_perplexity_and_kld.py) evaluates teacher-forced perplexity on uncompressed `WikiText-2` (`wiki.test.raw`, `24 × 512 = 12,288` tokens) alongside 32-probe step-0 and prefix-matched multi-step rollout Top-64 Logit KL-Divergence (`T = 1.0`, raw JSON artifacts in [`benchmarks/results/`](../benchmarks/results/)):

| `Nemotron-3.5-30B-A3B` Checkpoint | Disk Size | **`WikiText-2` `PPL ± σ` (`12,288` tok)** | **32-Probe Step-0 KLD vs. True `BF16`** | **Prefix-Matched Multi-Step KLD vs. True `BF16`** | **Same-Family KLD (`Step-0` / `Matched`)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **True `BF16` Master** | `58.84 GiB` | **`6.9582 ± 0.24151`** *(Ref)* | `0.0000 nats` (`100%` Ref) | `0.0000 nats` (`100%` Ref) | — |
| **`Fractal-MXFP4` *(from `Q4_0`)*** | `15.95 GiB` | `7.2927 ± 0.25382` (`+4.81%`) | `0.1623 nats` (`78.1%` Top-1, `96.9%` Top-5) | `0.1204 nats` (`155` pos, `88.4%` Top-1, `97.4%` Top-5) | — |
| **`Fractal-64` *(direct from `BF16`)*** | `14.36 GiB` | **`7.3610 ± 0.26017`** (`+5.79%`) | **`0.1254 nats`** (`81.2%` Top-1, **`100%` Top-5**) | **`0.0929 nats`** (`167` pos, **`89.8%` Top-1, `99.4%` Top-5**) | `0.0000` *(Ref)* |
| **`VQ3-64` *(Aligned Default)*** | `13.30 GiB` | **`7.4400 ± 0.26263`** (`+6.92%`) | **`0.1495 nats`** (`78.1%` Top-1, `96.9%` Top-5) | **`0.0944 nats`** (`149` pos, `85.2%` Top-1, `98.0%` Top-5) | **`0.0522 / 0.0889 nats`** vs. `Fractal-64` |
| **`Abliterated-VQ3-64` *(Opt-In)*** | `13.30 GiB` | `7.7721 ± 0.28352` (`+11.70%`) | `0.1431 nats` (`75.0%` Top-1, `100%` Top-5) | `0.1109 nats` (`141` pos, `84.4%` Top-1, `99.3%` Top-5) | **`0.0308 / 0.0312 nats`** vs. `VQ3-64` *(`87.5% / 92.4%` Top-1, `100%` Top-5)* |

---

## 4. Quick Commands for Nemotron (`v3.2.0`)

```bash
# 1. Install & SHA-256 verify pre-quantized Nemotron v3.2.0 checkpoints from Hugging Face
./install.sh --preset nemotron-vq3         # 13.30 GiB CPU-Turbo (types 44/45, AVX-512 VBMI + RouteCrack 2.0)
./install.sh --preset nemotron-maxq-16gb   # 13.39 GiB standard ggml <=42 (upstream llama.cpp & Q2_0-capable runners)
./install.sh --preset nemotron-fractal     # 14.36 GiB Fractal-64 (type 44, AVX2 / Apple Metal)

# 2. Launch interactive streaming chat or run the 100-Point Quality & Tool-Calling Gate
./modchip chat
./modchip eval

# 3. Install llama-quantize + companion tools (--all-tools) for custom standard-ggml quantization
./modchip download-bin --all-tools
./modchip quantize /path/to/bf16.gguf /path/to/out-q4_0.gguf Q4_0
```

### Trunk-only quantization (official Q4_0 experts kept)

The routed experts dominate file size, while the always-active trunk dominates decode. To
lower only the trunk and copy the experts at their existing type:

```bash
./scripts/build_trunk_quant.sh --profile trunk-q5 --input /path/to/official-q4_0.gguf --output /path/to/Trunk-Q5.gguf
./scripts/build_trunk_quant.sh --profile trunk-q4 --input /path/to/official-q4_0.gguf --output /path/to/Trunk-Q4.gguf
```

`trunk-q5` puts the token embedding, Mamba-2 `ssm_in`/`ssm_out` and shared experts at
Q5_0; `trunk-q4` puts the same tensors at Q4_0. Both keep the attention projections,
router and output head at Q8_0 and read the routed-expert type from the GGUF header, so
llama-quantize copies those tensors instead of requantizing them. On the official Q4_0
(600 paired BFCL v3 prompts), Q5_0 measured +11% decode at KLD 0.016 with BFCL unchanged
at 80.5%; Q4_0 measured +18% at KLD 0.034 but lost 8.7 BFCL points, from the Mamba-2 and
shared-expert weights rather than attention.
