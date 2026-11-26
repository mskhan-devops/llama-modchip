# llama-modchip on Windows — Porting Notes & Benchmarks (`v3.2.0`)

**Project:** `llama-modchip` `v3.2.0` (`Qwen3.8-Flash-Next-MaxQ-64GB` `53.99 GiB` & `MaxQ-32GB` `35.71 GiB`, `DeepSeek-V4-Flash-0731-MaxQ-v4` `57.99 GiB`, and `NVIDIA-Nemotron-3.5-30B-A3B` `VQ3-64` `13.30 GiB` / `MaxQ-16GB` `13.39 GiB` / `Abliterated-VQ3-64` `13.30 GiB`)
**Test machine:** AMD Ryzen 7 9800X3D (8C/16T, 96 MB L3) · 64 GB DDR5 · NVIDIA RTX 4070 Ti 12 GB · NVMe PCIe 5.0
**OS:** Windows 11 Pro · engine built with MinGW-w64 GCC (UCRT64) + CMake + Ninja, `-march=native` (AVX-512F/BW/VBMI/VNNI/BF16)
**Original Port & 1M-Context Verification:** [`@MR1KRAKEN`](https://github.com/MR1KRAKEN) ([PR #27](https://github.com/synchopate/llama-modchip/pull/27), [Issue #28](https://github.com/synchopate/llama-modchip/issues/28)); x64 AVX-512 ABI & `VQ3` update in `commit 54c89bf` ([Issue #31](https://github.com/synchopate/llama-modchip/issues/31)); unified `v3.2.0` `RouteCrack 2.0` + Quantized Split-KV + Native HARP-Cache v2 rebase on `llama.cpp` `46eb1742d` (`b12604`)

---

## 1. Custom vs. Standard Tensor Types in `v3.2.0`

In `v3.2.0`, **`Qwen3.8-Flash-Next-MaxQ-64GB/32GB`**, **`DeepSeek-V4-Flash-0731-MaxQ-v4`**, and **`NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB`** use **100% standard `ggml` tensor types** (`Q1_0 = 41`, `Q2_0 = 42`, `F8_E4M3 = 43` untouched), while CPU-Turbo `Nemotron-VQ3-64` (`13.30 GiB`) uses zero-collision **`GGML_TYPE_FRACTAL_64 = 44`** (`30 B / 64 W`) and **`GGML_TYPE_VQ3_64 = 45`** (`24 B / 64 W`). Stock upstream `llama.cpp` binaries do not register types `44`/`45` (and lack `v3.2.0`'s `RouteCrack 2.0` + `RouteCrack 2.0 L1 Micro-Arbiter`, Quantized Split-KV Decode, Native C++ Chained-Prefix HARP-Cache v2, AVX-512 GDN recurrence, zero-copy 3D MoE `mmap` repack preservation, and `deepseek4` ConeCollapse loader).

## 2. What had to change for Windows

The shipped POSIX scripts (`install.sh`, `modchip`, `scripts/build_zen5_llama.sh`) rely on Linux/macOS APIs (`taskset`, `/dev/shm/llama_modchip_<uid>.bin`, `sys/mman.h`), and GCC/MinGW-w64 on Windows x64 differs from Linux System V AMD64 in both ABI register preservation and PE/COFF `.rdata` alignment.

Five source and toolchain fixes are integrated into [`patches/llama-modchip-v3.2.patch`](../patches/llama-modchip-v3.2.patch) and [`windows/build_patched_engine.ps1`](../windows/build_patched_engine.ps1):

| # | File | Problem | Fix |
| :-- | :--- | :--- | :--- |
| 1 | `ggml/src/ggml-cpu/ops.cpp` | `#include <sys/mman.h>` + POSIX `/dev/shm` block | Guarded by `#if defined(__linux__)`; on Windows the attention policy comes from env (`LLAMA_ATTN_WINDOW`, `LLAMA_ATTN_SINKS`, `LLAMA_KV_L3_WINDOW`) |
| 2 | `ggml/src/ggml-cpu/ggml-cpu.c` | `__thread` and `__attribute__((aligned(64)))` are GCC-only | `MC_TLS` = `__declspec(thread)`; buffer aligned with `__declspec(align(64))` under `_MSC_VER` |
| 3 | `ggml/src/ggml-cpu/arch/x86/quants.c` | `__builtin_ctz(m)` inside the AVX-512 path | MSVC path uses `_BitScanForward` (`mc_ctz32`) |
| 4 | `src/llama-graph.cpp` | `ggml_dsv4_skip_csa_comp` is defined in a C file but referenced from C++ | C linkage declared under `_MSC_VER`; MinGW behaviour unchanged |
| 5 | `patches/llama-modchip-v3.2.patch` & `windows/build_patched_engine.ps1` | On Windows x64, `XMM6..XMM15` are callee-saved (`128-bit`), PE/COFF `.rdata` only guarantees 16-byte section alignment (`vmovdqa64` faults on 64-byte tables), and default `1–2 MB` Windows thread stacks overflow during deep graph builds ([Issue #31](https://github.com/synchopate/llama-modchip/issues/31)) | Replaced aligned 512-bit constant loads with unaligned `_mm512_loadu_si512` (`vmovdqu64`), isolated heavy AVX-512 kernels to avoid GCC UCRT64 `XMM6..XMM15` prologue/epilogue clobbers, and set `-Wl,--stack,16777216` (`16 MB` stack) |

One runtime constraint: for `deepseek4`/MLA (and recommended for `qwen4exp`/GQA) the KV cache uses **matching K and V types** (`-ctk q8_0 -ctv q8_0`).

## 3. Build

```powershell
# Unified v3.2.0 CPU engine (supports Qwen3.8, DeepSeek-V4 284B, & Nemotron-30B — default):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1

# Optional MSVC + CUDA hybrid build:
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1 -Cuda
```

The script clones `ggml-org/llama.cpp` at **`46eb1742d`** (`46eb1742db6d2caec1bfa848c5702b1bdcc815e5`, `b12604`), applies the single unified `patches/llama-modchip-v3.2.patch` (which includes `vq3_codebook.h` for `FRACTAL_64 = 44` / `VQ3_64 = 45`, `RouteCrack 2.0` + `RouteCrack 2.0 L1 Micro-Arbiter` headers in `common/`, `rc_harp_radix.h` in `tools/server/`, Quantized Split-KV Decode, `qwen4exp` + `deepseek4` + `nemotron_h_moe` + `inkling2` support, and Windows x64 portability guards), and builds a statically linked `windows/bin-modchip/llama-server.exe`.

Key flags: `-DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_EXE_LINKER_FLAGS="-static -Wl,--stack,16777216" -D_WIN32_WINNT=0x0A00`.

## 4. Measured performance (`Ryzen 9 9950X3D` & `Ryzen 7 9800X3D`, `DeepSeek-V4-Flash 284B`)

**AMD Ryzen 9 9950X3D** (`16C/32T`, `2 CCDs`, `128 MB` L3, `128 GB` DDR5-6000, `-t 16`, KV `q8_0`, flash-attn on, `DeepSeek-V4-Flash-0731-Abliterated-Fractal` `56.93 GiB` — [`@kevinblo`, Issue #33](https://github.com/synchopate/llama-modchip/issues/33)):
- **CPU, Top-K 6 (unpruned default)**: **`27.55 ± 0.05 tok/s` `pp512`** · **`8.11 ± 0.07 tok/s` `tg128`** (`+28%` decode and `+29%` prefill vs 1-CCD `9800X3D` Top-6 thanks to dual-CCD GMI3 read bandwidth).

**AMD Ryzen 7 9800X3D** (`8C/16T`, `96 MB` L3, `64 GB` DDR5, `llama-bench -r 3`, `-t 8`, KV `q8_0`, flash-attn on — [`@MR1KRAKEN`, Issue #28](https://github.com/synchopate/llama-modchip/issues/28)):

| Configuration | pp512 | tg128 |
| :--- | ---: | ---: |
| CPU, Top-K 6 (unpruned) | 21.3 tok/s | 6.35 ± 0.29 tok/s |
| CPU, Top-K 4 (pruned) | 21.1 tok/s | 8.04 ± 0.45 tok/s |
| CPU, 16 threads (SMT) | — | 6.41 tok/s (no gain) |
| GPU offload, 5 MoE layers in VRAM | 5.2 | 2.65 tok/s |
| GPU offload, dense-only | 3.7 | 2.71 tok/s |

Context scaling: 32K → 6.7, 128K → 6.9, 256K → 6.6 tok/s. The full **1M context (1,048,576)** starts and runs cleanly (~16 GB RAM free on a 64 GB host).

## 5. Why a 12 GB GPU does not help `DeepSeek-V4 284B`

1. MoE experts occupy ~55 GB of the 57.99 GiB `MaxQ-v4` checkpoint — only 5–7 of 43 layers fit in 12 GB VRAM (~15% of the work), so PCIe transfer synchronization between GPU and CPU layers costs more than partial MoE offload saves.
2. Note also that custom `GGML_TYPE_FRACTAL_64` (`type 44`) and `GGML_TYPE_VQ3_64` (`type 45`) kernels (used on `Nemotron VQ3-64` / `Fractal-64`) are CPU-only on Windows/Linux (no CUDA/Vulkan shader; Apple Metal supports `type 44`). For CUDA/Vulkan hybrid GPU+CPU offload on 16 GB models, select 100% standard `ggml` checkpoints (`Nemotron MaxQ-16GB` `13.39 GiB`, `Fractal-MXFP4`, or `Q4_0`) via `windows\AVVIA_MODCHIP.bat`.

Net result: on a 12 GB card `DeepSeek-V4 284B` is faster on CPU alone. (For 30B/14B/8B standard `Q4_0`, `MXFP4`, or `MaxQ-16GB` checkpoints where the dense trunk runs natively on CUDA/Vulkan, use `windows\AVVIA_MODCHIP.bat` instead.)

## 6. Operational notes & pitfalls

1. **Never set `DSV4_MOE_TOPK` / `LLAMA_MOE_TOP_K` in the environment for `DeepSeek-V4`.** During load the engine derives `ffn_gate_tid2eid` dimensions from that variable; because the GGUF stores 6 hash-routed experts, any other value aborts with `expected 4, 129280, got 6, 129280`. Drive runtime Top-K pruning exclusively through per-user `DSV4_MOE_CFG_PATH` (`$LOCALAPPDATA\llama-modchip\dsv4_moe_cfg.bin` via `modchip_dsv4.ps1 -Fast`).
2. **Repetition penalties and DRY sampling:** Do **not** pass `--dry-sequence-breaker none` — DRY then penalizes repeated chat-template tokens and corrupts output. If enabling DRY, keep default sequence breakers (`--repeat-penalty 1.1 --repeat-last-n 256`).
3. **Long-prompt prefill timeouts:** At ~15–22 tok/s prefill on 8 cores, a 6,000-token cold prompt takes several minutes before the first token. Raise HTTP client stream idle timeouts above 300 s, or pair `llama-server.exe` with `--slot-save-path` so recurring system/tool prefixes restore in milliseconds.
4. **One server per port:** Context size and MoE placement are fixed at startup; changing presets requires restarting `llama-server.exe`.
5. **`cache_reuse` log notice:** `cache_reuse is not supported by this context` is expected for `deepseek4`; LCP prefix slot reuse still works normally.

## 7. Summary

All three `v3.2.0` RAM tiers — `Qwen3.8-Flash-Next` (`MaxQ-64GB` `53.99 GiB` & `MaxQ-32GB` `35.71 GiB`), `DeepSeek-V4-Flash 284B` (`MaxQ-v4` `57.99 GiB`, 100% standard `ggml`), and `NVIDIA-Nemotron-3.5-30B-A3B` (`VQ3-64` `13.30 GiB` types `44`/`45`, `MaxQ-16GB` `13.39 GiB` standard `ggml`, and `Fractal-64` `14.36 GiB` type `44`) — compile and run natively on Windows 11 via `windows\build_patched_engine.ps1`, `windows\modchip_dsv4.ps1`, and `windows\modchip_win.ps1`, delivering **8.11 tok/s unpruned `Top-6` decode / 27.55 tok/s `pp512` prefill on Ryzen 9 9950X3D** and **6.35–8.04 tok/s pure decode / 21.3 tok/s prefill up to 1M context on an 8-core Ryzen 7 9800X3D in 64 GB RAM**.
