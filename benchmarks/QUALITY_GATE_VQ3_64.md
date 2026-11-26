# `llama-modchip v3.2.0` 3-Tier Quality, Signal Precision & `RouteCrack 2.0` Parity Gate

- **Release:** `v3.2.0` (Zero-Collision `ggml` ABI `41..45` · Unified `patches/llama-modchip-v3.2.patch` on `llama.cpp @ 46eb1742d` / `b12604`)
- **Target Flagship Fleet:**
  - **64 GB & 32 GB Tiers — `Qwen3.8-Flash-Next`** (`176.5B` total / `5B` active, `48` blocks: `36` Gated DeltaNet + `12` GQA + `512` routed experts + `320M`-row PLE + `blk.48` NextN MTP)
  - **64 GB Tier (`284B`) — `DeepSeek-V4-Flash-0731-MaxQ-v4`** (`284.33B` total / `13.17B` active, `43` MLA + `256+1s` MoE layers, unpruned `Top-6`, 100% donor-free standard `ggml` tensor types; requires `llama-modchip v3.2.0+` loader for `ConeCollapse` `ffn_down_exps.bias`)
  - **16 GB Tier — `NVIDIA-Nemotron-3.5-Lightning-30B-A3B`** (`30.2B` total / `3.0B` active, `52` blocks: `23` Mamba-2 SSM + `23` Sparse MoE + `6` GQA Attention, unpruned `Top-6`; `VQ3-64` CPU-Turbo, `Fractal-64` Modchip Pro & `MaxQ-16GB` 100% unmodified-upstream `ggml`)
- **Reference Testbed:** AMD Ryzen 9 9955HX (`16` Zen 5 cores, `64 MB` dual-CCD L3 cache, `74.0 GiB` Linux container RAM, DDR5-5600, Transparent HugePages `madvise`, `-t 15 -tb 15 --poll 0 --poll-batch 0`, `0 GPU`), with explicit Linux Cgroup v2 `MemorySwapMax=0` capacity enforcement (`60 GiB` for 64 GB tier, `28.5 GiB` for 32 GB tier, `15.0 GiB` for `MaxQ-16GB`).
- **Evaluation Suites:**
  1. **Blind `wiki-alt.txt` Perplexity & Logit KLD (`0.000000%` 8/13-gram calibration overlap):** `40 × 512` blind chunks on `Qwen3.8-Flash-Next` (vs. donor-free `BF16` reference `2.7766 ± 0.0549`) and `Nemotron-30B` (vs. `16.07 GiB` `MXFP4` reference `6.5962 ± 0.1729`); full `51`-chunk (`143,565` tokens) and strictly unseen `49`-chunk (`3..51`, `137,935` tokens) `AtomicChat/dsv4-eval-artifacts` `ref5632.kld` evaluation on `DeepSeek-V4 284B`.
  2. **Continuous `WikiText-2` & 32-Probe / Rollout Top-64 Logit KLD ([`benchmarks/eval_perplexity_and_kld.py`](eval_perplexity_and_kld.py)):** Anchored directly to the `58.84 GiB` True `BF16` master on `Nemotron-30B`.
  3. **`RouteCrack 2.0` Head-to-Head Throughput & Greedy Parity Gate:** Raw 4-workload (`W1` Claude Code refactor, `W2` multi-tool JSON, `W3` greenfield C99, `W4` bilingual IT/EN prose) JSON transcripts in [`results/qwen38_routecrack2_v31.json`](results/qwen38_routecrack2_v31.json), [`results/dsv4_routecrack2_v31.json`](results/dsv4_routecrack2_v31.json), and [`results/nemotron_routecrack2_v31.json`](results/nemotron_routecrack2_v31.json).
  4. **Functional Tool-Calling, Coding & Security Audit ([`benchmarks/eval_quality_and_tools.py`](eval_quality_and_tools.py), [`benchmarks/eval_hard60_and_refusal.py`](eval_hard60_and_refusal.py)).**

---

## 1. Zero-Collision `v3.2.0` `ggml` ABI & Format Signal Fidelity

`llama-modchip v3.2.0` preserves all upstream `ggml` tensor types (`0..43`) untouched and places custom block-64 CPU/Metal quantization formats at dedicated zero-collision slots `44` and `45`:

| `ggml` Enum ID | Symbol (`ggml.h`) | Effective `bpw` | Bytes / 64 W | Direct `BF16` SQNR | Cosine Sim vs. `BF16` | Backends |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `41` | `GGML_TYPE_Q1_0` *(Upstream)* | `1.250` | `10.0 B` | *RaBitQ PLE* | — | AVX-512 / AVX2 / Metal / CUDA |
| `42` | `GGML_TYPE_Q2_0` *(Upstream)* | `2.250` | `18.0 B` | *Signed-scale* | — | AVX-512 / AVX2 / Metal / CUDA |
| `43` | `GGML_TYPE_F8_E4M3` *(Upstream)* | `8.000` | `64.0 B` | *FP8 E4M3* | — | Upstream intact |
| **`44`** | **`GGML_TYPE_FRACTAL_64`** *(Modchip)* | **`3.750`** | **`30.0 B`** | **`17.75 dB`** (`+1.45 dB` vs `Q3_K`) | **`99.15%`** | AVX-512 / AVX2 / NEON / **Apple Metal** |
| **`45`** | **`GGML_TYPE_VQ3_64`** *(Modchip)* | **`3.000`** | **`24.0 B`** | **`13.57 dB`** (`+3.42 dB` vs `Q2_K`) | **`97.78%`** | AVX-512 VBMI (`VPERMI2B`) / AVX2 / NEON / Scalar |

*(BF16 reconstruction fidelity measured on `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16.gguf` `blk.1.ffn_up_exps.weight`, $N = 4,194,304$ weights; Standard `Q4_0`: `21.09 dB` / `99.61%`, `Q3_K`: `16.30 dB` / `98.84%`, `Q2_K`: `10.15 dB` / `95.37%`.)*

---

## 2. `Qwen3.8-Flash-Next` (`MaxQ-64GB` & `MaxQ-32GB`) — Blind `wiki-alt.txt` Quality Gate (`3.43` PPL)

Evaluated on `40 × 512` blind `wiki-alt.txt` chunks (`0.000000%` 8-gram and 13-gram overlap with calibration) against the donor-free `BF16` reference logit base (`Mean PPL(base) = 2.776624 ± 0.054868`):

| Checkpoint | Exact LLM Bytes (`GiB`) | Trunk + PLE (`GiB`) | Built-In MTP | `Mean PPL(Q) ± SE` | `Mean KLD ± SE` | `Median KLD` | `Same Top-1 (%)` | Peak `VmHWM` (`-c 32768`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`MaxQ-64GB`** *(64 GB Flagship)* | **`57,969,748,352 B`** (`53.99 GiB`) | `43.26 + 10.73` (`512/512` exps) | **Yes (`blk.48`)** | **`3.4325 ± 0.0708`** *(`3.4383` repack)* | **`0.4971 ± 0.0099`** | **`0.1060`** | **`78.314 ± 0.408%`** | **`45.08 GiB`** *(`< 60.0 GiB` cgroup)* |
| **`MaxQ-64GB-Abliterated`** *(`v3.0-r2` Heretic-2)* | **`57,969,748,352 B`** (`53.99 GiB`) | `43.26 + 10.73` (`27` edited + MTP) | **Yes (`blk.48`)** | *`0.0818` KLD vs. base (`0/100` refusals)* | *(`1,058 / 1,085` bit-identical)* | — | — | **`46.34 GiB`** *(`RssAnon = 2.31 GiB`)* |
| `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` | `58,408,584,928 B` (`54.40 GiB`) | `27.58 + 26.82` (`256/512` pruned) | No | `5.0916 ± 0.1214` | `0.9302 ± 0.0156` | `0.2964` | `69.382 ± 0.456%` | `38.55 GiB` (`--no-repack`) |
| **`MaxQ-32GB`** *(32 GB Default)* | **`38,341,683,424 B`** (`35.71 GiB`) | **`24.98 + 10.73`** (`512/512` exps) | **Yes (`blk.48`)** | **`5.8120 ± 0.1402`** *(`5.8041` repack)* | **`1.0069 ± 0.0149`** | **`0.3986`** | **`66.275 ± 0.468%`** | **`26.91–27.44 GiB <= 28.5 GiB`** |
| **`MaxQ-32GB-Abliterated`** *(`v3.0-r2` Heretic-2)* | **`38,341,683,424 B`** (`35.71 GiB`) | **`24.98 + 10.73`** (`32` edited + MTP) | **Yes (`blk.48`)** | *`0.0818` KLD vs. base (`0/100` refusals)* | *(`1,095 / 1,127` bit-identical)* | — | — | **`28.40 GiB <= 28.5 GiB`** |

- **`MaxQ-64GB` vs. Competitor (`~54 GiB` Class):** Compressing the `320M`-row PLE table from `26.82 GiB` (`IQ4_NL`) to `10.73 GiB` (`PCA-Wiener D=256 + RaBitQ Q1_0`) reinvests `+16.09 GiB` into `512/512` unpruned experts, `FP64 Woodbury DualGram Q2_0`, and built-in `blk.48` MTP — yielding **`-1.6591` PPL**, **`-46.6%` Mean KLD**, **`-64.3%` Median KLD**, and **`+8.93 pp` Top-1 agreement** vs. `ISTA-DASLab GSQ-RCO-Coder-IQ1_M` (note: a coder-tuned `256/512`-pruned checkpoint evaluated on general Wikipedia `wiki-alt.txt`) while `0.41 GiB` smaller.
- **`MaxQ-32GB` Cold-Cache Cgroup v2 Certification:** Passes cold-cache Linux Cgroup v2 (`drop_caches=3`, `Cached=0.3 GiB`, `memory.max = 28.5 GiB`, `MemorySwapMax = 0`, `oom_kill = 0`, `n = 3` cold restarts in [`results/qwen38_32gb_cgroup28g_v311.json`](results/qwen38_32gb_cgroup28g_v311.json)) at **`28.08 GiB VmHWM` / `28.26 GiB` cgroup peak** (`26.11` / `26.79 GiB` `N=1`, `pgsteal = 0`, `cg.file = 25.14 GiB` out of `35.71 GiB` on disk — proving **`> 10.5 GiB` PLE stays lazily unfaulted**; `14.58 ± 0.32 tok/s` `RouteCrack 2.0` vs `10.00 ± 0.01` `N=1`; `26.91–27.44 GiB` in shorter-context gate; `0 / 18` third-party HF `Qwen3.8` GGUFs fit under `28.5 GiB`). Note that `MaxQ-32GB` applies 42-layer `ConeCollapse` width pruning (`K=640->512` with `ffn_down_exps.bias`, requiring `llama-modchip v3.1.1+`) and trades quality (`5.8120` PPL, `1.0069` KLD vs. `3.4325` / `0.4971` on 64 GB) to fit 32 GB RAM. At `1.63 bpw`, run with `--temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05` (automated by `./modchip`) for loop-free generation.
- **`--repack` Equivalence Gate:** `batch = 1` `--repack` vs. `--no-repack` yields **`KLD = 0.00000 ± 0.00000`** with `RssAnon` reduced `5.4×` (`9.81 GiB -> 1.81 GiB`) via zero-copy 3D MoE `mmap` preservation.

---

## 3. `RouteCrack 2.0` (`v3.1.1`) Throughput, Cold-Cache Cgroup v2 Suite, & Target Parity

Verified across all three flagship families (`T = 0`, 16 Zen 5 cores, `0 GPU`, `74.0 GiB` container): uncapped single-run `W1–W4` head-to-head logs in `benchmarks/results/*_routecrack2_v31.json`, plus **all four cold-cache Linux Cgroup v2 (`drop_caches=3`, `Cached=0.3 GiB`, `MemorySwapMax=0`, `oom_kill=0`, `n = 3` independent cold restarts each) JSONs**:
- **Certified strictly below cap (`pgsteal = 0`, `workingset_refault = 0`):**
  - [`results/qwen38_64gb_cgroup60g_v311.json`](results/qwen38_64gb_cgroup60g_v311.json) (`60.0 GiB` cap): `N=1` **`8.78 ± 0.02`** (`46.53 GiB` cg.peak) vs `RC2` **`15.64 ± 0.05 tok/s`** (`W1 22.00 ± 0.19`, `48.35 GiB` cg.peak)
  - [`results/qwen38_32gb_cgroup28g_v311.json`](results/qwen38_32gb_cgroup28g_v311.json) (`28.5 GiB` cap): `N=1` **`10.00 ± 0.01`** (`26.79 GiB` cg.peak) vs `RC2` **`14.58 ± 0.32 tok/s`** (`W1 15.31`, `28.26 GiB` cg.peak — tight `0.24 GiB` headroom at 32K, `>10.5 GiB` PLE unfaulted)
  - [`results/nemotron_16gb_cgroup15g_v311.json`](results/nemotron_16gb_cgroup15g_v311.json) (`15.0 GiB` cap, `N=1`): **`22.64 ± 0.03 tok/s`** (`~15.03 GiB` needed, `0.2 GiB` refault — nearly fits `15.0 GiB`; higher than earlier unpinned `-t 15 --poll 0` `15.10 t/s` `llama-bench` gate due to pinned `16C` `taskset` + `q8_0/q4_0` KV)
- **Exceeds cap at `-c 32768` (OOM-free via continuous host-ZFS-ARC-backed file reclaim; bare-metal throughput unverified):**
  - [`results/dsv4_64gb_cgroup60g_v311.json`](results/dsv4_64gb_cgroup60g_v311.json) (`60.0 GiB` cap): requires `~60.3–60.7 GiB` (`>= 61 GiB` free RAM for zero refaults); completes OOM-free via `10.7–30.7 GiB`/run ZFS-ARC-backed file reclaim (`8.88 ± 0.25` RC2 / `7.68 ± 0.01` N=1)
  - [`results/nemotron_16gb_cgroup15g_v311.json`](results/nemotron_16gb_cgroup15g_v311.json) (`15.0 GiB` cap, `RC2`): requires `~15.75 GiB` (`>= 16 GiB` free RAM for zero refaults); completes OOM-free via `8.4–9.2 GiB`/run ZFS-ARC-backed file reclaim (`26.28 ± 0.19`)

| Flagship Checkpoint | Pure `N=1` (`none`) | `v3.0` Default | **`RouteCrack 2.0` (`v3.1.1`)** | `W1` / `W2` / `W3` / `W4` (`tok/s`) | Accept (`Mean Len`) | Exact Token Match vs. `N=1` (`T=0`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.8-Flash-Next-MaxQ-64GB`** ([`JSON`](results/qwen38_routecrack2_v31.json)) | `8.67 t/s` (`8.50/8.68/8.68/8.68`) | `13.94 t/s` (`draft-mtp`: `16.61/16.84/13.24/10.76`) | **`15.34 tok/s`** (`+77.0%` vs `N=1`, `+10.1%` vs `v3.0`) | **`21.38` / `17.38` / `12.63` / `12.78`** | **`79.95%`** (`3.66`) | `W1`: `254/254`, `W2`: `254/254`, `W3`: `4/254`, `W4`: `48/254` *(batch-matmul fp tie drift; plain `MTP-1` also `4/254` on `W3`)* |
| **`DeepSeek-V4-Flash MaxQ-v4`** ([`JSON`](results/dsv4_routecrack2_v31.json)) | `6.80 t/s` (`6.48/6.70/6.90/7.01`) | `7.73 t/s` (`ngram-map-k`: `11.90/7.39/6.74/6.55`) | **`7.79 tok/s`** (`+14.5%` vs `N=1`, `+0.84%` vs `v3.0` — tied within noise) | **`12.31` / `7.55` / `6.74` / `6.49`** | **`85.92%`** (`2.97`) | `W1`: `191/191`, `W2`: `191/191`, `W3`: `34/191`, `W4`: `48/191` *(fixes `v3.0` `W2` JSON divergence)* |
| **`Nemotron-3.5-30B (Fractal-MXFP4 ref)`** ([`JSON`](results/nemotron_routecrack2_v31.json)) | `22.60 t/s` (`22.20/22.57/22.69/22.61`) | `27.13 t/s` (`ngram-map-k`: `42.54/26.08/22.65/23.80`) | **`28.20 tok/s`** (`+24.8%` vs `N=1`, `+3.96%` vs `v3.0` — within noise) | **`42.18` / `28.83` / `22.97` / `24.72`** | **`78.59%`** (`3.35`) | `W1`: `255/255`, `W2`: `255/255`, `W3`: `17/253`, `W4`: `173/255` *(measured on `16.07 GiB` `Fractal-MXFP4` ref; fixes `v3.0` `W2` `57/255`)* |

---

## 4. `DeepSeek-V4-Flash-0731-MaxQ-v4` (`284.33B`, `57.99 GiB`, 100% Standard `ggml` Types)

Built 100% donor-free from `deepseek-ai/DeepSeek-V4-Flash-0731` (`7872f01b1d1fe23eabc4c98b48bffcef5a386062`) with **zero custom tensor types** (`1,362` standard `ggml` tensors: `9` full-width $K=2048$ anchor layers + `34` $K=1792$ ConeCollapse layers with shrunk centroid bias `ffn_down_exps.bias` requiring `llama-modchip v3.1.1+` + FP64 Woodbury DualGram $N_C=384$), executing natively on CPU, Apple Metal, CUDA, and Vulkan via `llama-modchip`. *(Note: the historical `27.0 t/s` Apple M5 Max measurement in [`apple-silicon/results/dsv4-m5max.md`](../apple-silicon/results/dsv4-m5max.md) was collected on the earlier withdrawn `DeepSeek-V4-Fractal` `56.56 GiB` build, not `MaxQ-v4`.)*

### Full 51-Chunk & Strictly Unseen 49-Chunk (`3..51`) `AtomicChat` `wiki-alt.txt` (`ref5632.kld`)

| Publisher & `DeepSeek-V4-Flash-0731` Checkpoint | Disk Size | Resident in `64 GB` RAM? | Chunk-12 `wiki-alt.txt` `PPL ± σ` | Full 51-Chunk `wiki-alt.txt` `PPL` (`143,565` tok) |
| :--- | :---: | :---: | :---: | :---: |
| `bullerwins` `MXFP4_MOE-BF16` *(Full Reference)* | `150.75 GiB` | No (`192 GB+`) | `4.4474 ± 0.0587` | **`4.5381 ± 0.0292`** *(`8×5090`; `4.3406` on `H100`)* |
| `prometheusAIR` `IQ2_XXS-slim` | `78.84 GiB` | No (`96 GB+`) | `5.5360 ± 0.0754` | `5.5165 ± 0.0365` |
| `antirez` `IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-0731` | `80.76 GiB` | No (`96 GB+`) | `5.5484 ± 0.0762` | `5.6025 ± 0.0377` |
| `unsloth` `UD-IQ1_S` | `76.87 GiB` | No (`96 GB+`) | — | `5.9734` |
| `AtomicChat` `AD-IQ1_M` | `65.42 GiB` | No (`>64 GiB` OOM) | — | `6.3813` |
| **`Synchopate` `MaxQ-v4-0731` *(100% Donor-Free Default)*** | **`57.99 GiB`** | `~60.5 GiB` WS (needs `>= 61G` free; `60G` cg OOM-free via ARC reclaim, [`JSON`](results/dsv4_64gb_cgroup60g_v311.json)) | **`6.8026 ± 0.0977`** | **`6.8592 ± 0.0482`** (`Mean KLD = 0.6485`, `Median = 0.1838`, `Top-1 = 73.283%`; [`log`](results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_v4.log)) |
| `Synchopate` `MaxQ-Fractal-0731` *(Historical `v2.5.3`)* | `56.93 GiB` | `74 GiB` host (`58.13 GiB` `VmRSS`; legacy) | `7.3180 ± 0.1039` | `7.2977 ± 0.0504` (`Mean KLD = 0.7400`, `Top-1 = 70.397%`; [`log`](results/atomicchat_wikialt_ctx5632_51c_dsv4_maxq_fractal.log)) |
| `DevQuasar` `Q2_K` *(uncalibrated outlier)* | `95.99 GiB` | No (`128 GB+`) | `9.4237 ± 0.1402` | `9.1127 ± 0.0647` |

- **Strictly Unseen 49-Chunk Split (`Chunks 3..51`, `137,935` tokens):** Excluding diagnostic chunks `1..2`, donor-free `MaxQ-v4` achieves **`PPL(Q) = 6.8471`** (`-6.12%` vs. `MaxQ-Fractal` `7.2938`), **`Mean KLD = 0.625743 nats`** (`-12.84%`), **`RMS Δp = 26.726%`**, and **`Same top p = 73.595%`** (`+2.894 pp`), significantly outperforming an internal `AD-BF16 + AD-IQ1_M` donor hybrid ablation (`MaxQ-v3`) on KLD ($p = 6.58 \times 10^{-5}$), $\Delta p$ ($p = 8.84 \times 10^{-10}$), and Top-1 ($p = 1.26 \times 10^{-13}$).

---

## 5. `NVIDIA-Nemotron-3.5-30B-A3B` (`16 GB` Tier — `VQ3-64`, `Fractal-64`, & `MaxQ-16GB`)

### Blind `wiki-alt.txt` (`40 × 512`), True-`BF16` `WikiText-2` (`12,288` tok), & Cgroup Certification

| `Nemotron-3.5-30B-A3B` Checkpoint | `ggml` ABI | Disk Size | Blind `wiki-alt` `PPL` (`Mean KLD` vs `MXFP4`) | `WikiText-2` `PPL ± σ` (`12,288` tok) | KLD vs. True `BF16` (`Step-0` / Matched) | Gate / Cgroup Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **True `BF16` Master** *(`fd513c91...`)* | `BF16` | `58.84 GiB` | — | **`6.9582 ± 0.24151`** *(Ref)* | **`0.0000 nats`** | `100.0 / 100` |
| `MXFP4` Reference *(Donor-Free)* | Std (`39`) | `16.07 GiB` | `6.5962 ± 0.1729` *(Ref)* | `7.2927 ± 0.25382` *(from `Q4_0`)* | `0.1623 / 0.1204 nats` | `100.0 / 100` |
| **`Fractal-64`** *(AVX2 / Metal)* | **Type `44`** | `14.36 GiB` | **`6.6533`** (`0.0688` KLD) | **`7.3610 ± 0.26017`** | **`0.1254 / 0.0929 nats`** | `100.0 / 100` · Metal `78.6 t/s` |
| **`VQ3-64`** *(16 GB CPU-Turbo Default)* | **Types `44/45`** | **`13.30 GiB`** | **`6.7614`** (`0.0871` KLD) | **`7.4400 ± 0.26263`** | **`0.1495 / 0.0944 nats`** | **`100.0 / 100`** · `8/8` Audit |
| **`MaxQ-16GB`** *(Universal Std `ggml`)* | **Std (`<= 42`)** | **`13.39 GiB`** | **`7.3129 ± 0.1923`** (`0.1722` KLD, `83.43%` Top-1) | — | — | **`14.304 GiB <= 15.0 GiB` Cgroup** (`188.3 pp` / `27.8 t/s` refactor) |
| **`Abliterated-VQ3-64`** *(Opt-In)* | **Types `44/45`** | `13.30 GiB` | — | `7.7721 ± 0.28352` | `0.1431 / 0.1109 nats` (`0.0308` vs `VQ3`) | `8 / 8` Security Audit |

*(Note: `VQ3-64`, `Fractal-64`, and `Abliterated-VQ3-64` were migrated cleanly to zero-collision `v3.0+` ABI types `44`/`45` via zero-recompute 47-byte header updates with zero payload weight changes.)*

---

## 6. Reproducibility Commands (`v3.2.0`)

```bash
# 1. Build unified llama-modchip v3.2.0 engine with quantizer & backend kernel tests
BUILD_TESTS=1 ./scripts/build_zen5_llama.sh ./workspace/llama.cpp
./workspace/llama.cpp/build/bin/test-quantize-fns
./workspace/llama.cpp/build/bin/test-backend-ops test -b CPU

# 2. Or download all prebuilt v3.2.0 Linux x86_64 binaries (including llama-perplexity & llama-bench)
./modchip download-bin --all-tools

# 3. Download & SHA-256 verify any 3-Tier checkpoint from Hugging Face
./install.sh --preset qwen38-64gb --yes
./install.sh --preset qwen38-32gb --yes
./install.sh --preset dsv4-maxq --yes
./install.sh --preset nemotron-vq3 --yes
./install.sh --preset nemotron-maxq-16gb --yes

# 4. Run WikiText-2 / wiki-alt Perplexity & 32-Probe Logit KLD
python3 benchmarks/eval_perplexity_and_kld.py \
  --llama-perplexity ./workspace/bin/llama-perplexity \
  --model ./workspace/models/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf \
  --ctx-size 512 --chunks 24 \
  --url http://127.0.0.1:8789 \
  --ref-logits benchmarks/results/nemotron_bf16_ref_logits.json \
  --rollout-steps 8

./workspace/bin/llama-perplexity \
  -m ./workspace/models/dsv4-maxq-v4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf \
  -f wiki-alt.txt -c 5632 -b 5632 --repack -fa on -t 15 -tb 15 --poll 0 --ppl-output-type 1

# 5. Run the 100-Point Functional Gate & RouteCrack 2.0 C++ ASan/UBSan Suite
python3 benchmarks/eval_quality_and_tools.py --url http://127.0.0.1:8789 --target-tps 15.0
g++ -std=c++17 -O2 -Wall -Wextra -fsanitize=address,undefined -Ispeculative/cpp speculative/cpp/test_routecrack2.cpp -o /tmp/test_rc2 && /tmp/test_rc2
```
