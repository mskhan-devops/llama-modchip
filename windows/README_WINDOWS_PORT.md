# Unified `llama-modchip v3.2.0` Engine on Windows (`Qwen3.8`, `DeepSeek-V4 284B`, & `Nemotron-30B`)

In **`v3.2.0`**, **`Qwen3.8-Flash-Next-MaxQ-64GB`** (`53.99 GiB`), **`Qwen3.8-Flash-Next-MaxQ-32GB`** (`35.71 GiB`), **`DeepSeek-V4-Flash-0731-MaxQ-v4`** (`57.99 GiB`), and **`NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB`** (`13.39 GiB`) use **100% standard `ggml` tensor types** (`Q1_0 = 41`, `Q2_0 = 42`, `F8_E4M3 = 43` untouched), while CPU-Turbo **`NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64`** (`13.30 GiB`) uses zero-collision **`GGML_TYPE_FRACTAL_64 = 44`** (`30 B / 64 W`) and **`GGML_TYPE_VQ3_64 = 45`** (`24 B / 64 W`).

`AVVIA_MODCHIP.bat` (`modchip_win.ps1`) remains the GPU+CPU hybrid launcher for 16 GB standard GGUFs (`Qwen3`, `Nemotron Q4_0` / `MaxQ-16GB`, `Nemotron-H-8B`). For the unified `v3.2.0` CPU engine (**`RouteCrack 2.0` `--spec-type routecrack2`** + Quantized Split-KV + Native HARP-Cache v2 across `Qwen3.8-MaxQ-64GB/32GB`, `DeepSeek-V4-Flash 284B` ConeCollapse, and `Nemotron VQ3-64` AVX-512 VBMI), use the workflow below.

## 1. Build the unified `v3.2.0` patched engine

With MSYS2 MinGW-w64 UCRT64 (`gcc >= 14`), Git, CMake, and Ninja on `PATH`:

```powershell
# Unified v3.2.0 engine (supports Qwen3.8, DeepSeek-V4 284B & Nemotron-30B — default):
powershell -ExecutionPolicy Bypass -File windows\build_patched_engine.ps1
```

`build_patched_engine.ps1` clones `ggml-org/llama.cpp` at pinned commit **`46eb1742d`** (`46eb1742db6d2caec1bfa848c5702b1bdcc815e5`, `b12604`), applies [`patches/llama-modchip-v3.2.patch`](../patches/llama-modchip-v3.2.patch) (including `RouteCrack 2.0` + `RouteCrack 2.0 L1 Micro-Arbiter`, Quantized Split-KV Decode, Native C++ Chained-Prefix HARP-Cache v2, `FRACTAL_64 = 44`, `VQ3_64 = 45`, `qwen4exp` AVX-512 GDN + 3D MoE `mmap` repack preservation, `deepseek4` ConeCollapse, unaligned `vmovdqu64` `.rdata` loads, Windows x64 `XMM6..XMM15` ABI safety, and `16 MB` stack size `-Wl,--stack,16777216`), and outputs static `windows\bin-modchip\llama-server.exe`, `llama-cli.exe`, and `llama-bench.exe`.

## 2. Run

Place the GGUF shards (`Qwen3.8-Flash-Next-MaxQ-64GB-*.gguf`, `Qwen3.8-Flash-Next-MaxQ-32GB-*.gguf`, `DeepSeek-V4-Flash-0731-MaxQ-v4-*.gguf`, or `NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf`) in `windows\models\` and run:

```bat
windows\AVVIA_DSV4.bat
```

`modchip_dsv4.ps1` detects physical cores and RAM, writes per-user `$LOCALAPPDATA\llama-modchip\dsv4_moe_cfg.bin` (`DSV4_MOE_CFG_PATH`), and starts an OpenAI-compatible server at `http://127.0.0.1:8080/v1` (pass `-OpenBrowser` to launch the browser UI automatically).

Useful flags for `windows\modchip_dsv4.ps1`: `-Ctx 65536`, `-Fast` (`Top-4` $L_2$-quadrature pruning), `-KvType q4_0`, `-NPredict N`, `-Bench`.
`windows\make_shortcuts.ps1` generates desktop shortcuts for context presets from `32K` up to `1M`.

## 3. Key constraints

- `deepseek4` / MLA requires matching K and V KV-cache types (`-ctk q8_0 -ctv q8_0` or `-ctk q4_0 -ctv q4_0`; mixed `-ctk q8_0 -ctv q4_0` is rejected).
- Never export `DSV4_MOE_TOPK` or `LLAMA_MOE_TOP_K` in the environment when loading `DeepSeek-V4-Flash`; control Top-K pruning strictly via per-user `DSV4_MOE_CFG_PATH` (`-Fast`).
- Raise HTTP client stream idle timeouts above 300 s for multi-thousand-token cold prompts on 8-core CPUs.

Full engineering report and Ryzen 7 9800X3D benchmarks: [`docs/WINDOWS_PORT.md`](../docs/WINDOWS_PORT.md).
