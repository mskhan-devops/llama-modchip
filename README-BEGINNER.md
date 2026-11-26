# llama-modchip — Beginner's Guide (plain English)

This is a **human-friendly companion** to the project's big `README.md`, `ARCHITECTURE.md` and `FAQ.md`.
It explains what the scripts do, which ones to run on **your** laptop, and in what order.

Everything below was checked against the v3.2.0 files in this repository on **2026-11-25**.
All speed numbers marked "estimate" are rough planning estimates, not measurements on your machine.

---

## 0. The 30-second summary

`llama-modchip` is a **patched version of `llama.cpp`** (the popular open-source engine that runs
GGUF language-model files) plus a set of helper scripts.

Normal `llama.cpp` is already fast on a GPU. This project's specialty is making **large Mixture-of-Experts
(MoE) models run on ordinary CPU RAM** — its tagline is "0 GPUs Required". It does this with:

| Feature | Plain-English explanation |
|---|---|
| **RouteCrack 2.0** (`--spec-type routecrack2`) | Guesses several upcoming tokens and verifies them quickly. Like a typist who writes a whole word while you check it. |
| **HARP / FNV radix cache** | Remembers previous conversations so repeated prefixes don't have to be recomputed. |
| **EAKV + quantized KV cache** (`-ctk q8_0 -ctv q4_0`) | Stores the model's "short-term memory" in 8-bit/4-bit instead of 16-bit, so more context fits in less RAM. |
| **`--repack`, core pinning, `--poll 0`** | Rearranges model weights and avoids CPU threads fighting each other. |
| **`/dev/shm` slot saving** | Keeps conversation snapshots in RAM-backed `/dev/shm` for very fast switching back to an old chat. |
| **Custom quant types** (`FRACTAL_64`, `VQ3_64`) | Extra-small weight formats. **CPU-only on Linux/Windows** — no CUDA support. |

Important: **you do not need these patches to run a normal GGUF on a GPU.** Stock `llama.cpp` already does that.
You need this project if you want its CPU-oriented tricks, its supported MaxQ model builds, the live
control SHM interface, or the web gateway.

---

## 1. Your machine (auto-detected facts)

| Item | Detected value | Consequence |
|---|---|---|
| CPU | Intel Core i5-12450HX, 8 physical cores (4P+4E), 12 logical threads, up to 4.4 GHz | Use **7 threads** (`-t 7 -tb 7`), not 12 |
| SIMD | **AVX2, FMA, F16C, BMI2, AVX-VNNI**; **no AVX-512** | Use AVX2 builds; skip AVX-512/VBMI presets |
| L3 cache | 12 MB (one domain) | Profiler sets `LLAMA_KV_L3_WINDOW=512` |
| RAM | 23 GiB total, ~18 GiB available, **no swap** | Models above ~16 GB are tight; keep context short |
| GPU | RTX 4050 Laptop, **6141 MiB** (~5.7 GB free), compute capability **8.9 / sm_89** | Only ~3–5 GB models fit fully; need `GGML_CUDA=ON` + `-ngl` |
| Driver / toolkit | Driver 595.99.02 (CUDA 13.2 capable); local `nvcc` is **CUDA 12.0** | Project docs use CUDA 12.4 — upgrade toolkit before a CUDA build |
| Disk | 237 GB free NVMe, `/dev/shm` = 12 GB | Plenty of room for models; `/dev/shm` slot saves are fine |
| OS | Linux Mint 22.3 (Ubuntu 24.04 base) | `install.sh` Linux path applies |

> **Note from this review:** your shell history shows `./install.sh` and `cd workspace/llama.cpp`,
> but at review time there was **no `workspace/` folder, no `start_server.sh`, and no
> `.modchip_workspace`** in `/home/mskhan/llama-modchip`. Either the build was removed or it was
> made with a different `--workspace`. The printed server command in section 5 points at
> `workspace/bin/llama-server-nemotron`, which currently does not exist. Run one of the installs in
> section 3 first.

---

## 2. How the pieces fit together

```
 Your GGUF model file
        │
        ▼
 scripts/llama_modchip.py  ── reads GGUF header + CPU info, writes 48-byte SHM control file
        │                     and prints the recommended llama-server command
        ▼
 workspace/bin/llama-server-nemotron  ── patched llama.cpp engine (the thing that generates tokens)
        │                                  listens on http://127.0.0.1:8789
        ▼
 scripts/chat.py  ── terminal chat client (shows tok/s and draft acceptance)
        │
        ▼ (optional)
 gateway/sse_gateway.py  ── web chat + telemetry HUD on http://127.0.0.1:8787
```

And the two "front doors":

- **`./modchip`** — the all-in-one CLI (`setup`, `start`, `chat`, `bench`, `tui`, `stop`, …).
- **`./install.sh`** — the installer that downloads/builds the engine and generates `start_server.sh`.

---

## 3. Building / installing (three options)

All commands below assume:

```bash
cd /home/mskhan/llama-modchip
```

### Option A — Fastest to get running: prebuilt CPU binary, use your own models

```bash
./install.sh --prebuilt --model /home/mskhan/LLMs/NeoHorse-1-4B-Q5_K_M.gguf --yes
```

What happens: detects AVX2 → downloads the SHA-256-verified
`llama-server-modchip-linux-x86_64-avx2` binary from the v3.2.0 release → installs it as
`workspace/bin/llama-server-nemotron` → generates `start_server.sh` and `start_gateway.sh`.
**No model is downloaded** because you passed `--model`. This engine is **CPU-only**.

### Option B — Fastest CPU generation: compile from source

```bash
./install.sh --source --model /home/mskhan/LLMs/NeoHorse-1-4B-Q5_K_M.gguf --yes
```

What happens: clones pinned `ggml-org/llama.cpp` commit `46eb1742d` (build b12604) into
`workspace/llama.cpp`, applies `patches/llama-modchip-v3.2.patch`, configures CMake with
`-DGGML_AVX512=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_OPENMP=ON`,
builds `llama-server`, `llama-cli`, `llama-bench`, `llama-quantize`, `llama-perplexity`, and copies
`llama-server` to `workspace/bin/llama-server-nemotron`.

Equivalent manual build (useful if the installer's preset prompt gets in the way):

```bash
cd /home/mskhan/llama-modchip
PORTABLE_X86_ISA=avx2 ./scripts/build_zen5_llama.sh /home/mskhan/llama-modchip/workspace/llama.cpp
```

`PORTABLE_X86_ISA=avx2` sets `-march=haswell -mno-avx512f`. Build-and-run on this same laptop can
instead leave it unset so CMake's `-march=native` is used (slightly better, less portable).

### Option C — GPU (CUDA) build for the RTX 4050

The Linux installer **never enables CUDA by itself** — it has no `nvidia-smi` check, no `-ngl`, and
defaults `GGML_CUDA=OFF`. You must opt in:

```bash
cd /home/mskhan/llama-modchip

# 1. Enable the CUDA backend during the patched build (engine only, no model download)
GGML_CUDA=ON ./install.sh --source --skip-model --yes

# 2. Pin the CUDA architecture to your RTX 4050 (compute capability 8.9) and rebuild
cmake -S workspace/llama.cpp -B workspace/llama.cpp/build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build workspace/llama.cpp/build -j 8 --target llama-server llama-cli llama-bench
cp workspace/llama.cpp/build/bin/llama-server workspace/bin/llama-server-nemotron

# 3. Start with GPU offload explicitly — the generated start_server.sh has NO -ngl flag
MODCHIP_EXTRA_ARGS="-ngl 99" ./modchip start /home/mskhan/LLMs/NeoHorse-1-4B-Q5_K_M.gguf
```

Notes:

- The repo's Linux builder exposes `GGML_CUDA` but does **not** set `-DCMAKE_CUDA_ARCHITECTURES`;
  step 2 above fixes that. The Windows PowerShell builder defaults to `Arch = "89"`, which is why 89
  is correct here.
- Your installed `nvcc` is CUDA **12.0**; project docs use CUDA **12.4**. If CMake or nvcc complains,
  install a newer CUDA toolkit (12.4+ or 13.x — your driver supports up to 13.2) and retry with a
  clean build directory (`RESET_LLAMA_TREE=1` and/or delete `workspace/llama.cpp/build`).
- The custom `FRACTAL_64`/`VQ3_64` quant types have **no CUDA kernels**. Use standard quant files
  (Q4_K, Q5_K, MXFP4, Q4_0, …) for GPU work.

### What `install.sh` actually does, step by step

1. **Bootstrap** — if launched via `curl … | bash`, it clones this repo and re-runs itself locally.
2. **Parse options** (see table below) and create `workspace/{bin,models,routecrack,cache}`.
3. **Audit hardware** — counts physical cores, RAM, CPU flags, `/dev/shm`, THP state. It also runs
   `scripts/check_cpu_readiness.py`, but only for *display* — the installer ignores its advice.
   On your machine it silently picks preset **`nemotron-maxq-16gb`** (23 GiB RAM + no AVX-512).
4. **Install the engine** — prebuilt download + SHA check, or source build with the patch.
5. **Get a model** — download preset shards with size/SHA-256 verification, or use `--model`,
   or `--skip-model`.
6. **Prepare RouteCrack tables** — only for Nemotron presets (others use `RC_TABLES_DIR=/nonexistent`).
7. **Generate `start_server.sh` and `start_gateway.sh`**, save `.modchip_workspace`, and print next steps.

### `install.sh` options

| Flag | Meaning |
|---|---|
| `--workspace <dir>` | Build/model directory. Default `/home/mskhan/llama-modchip/workspace` here. |
| `--preset <id>` | One of `qwen38-64gb`, `qwen38-32gb`, `dsv4-maxq`, `nemotron-vq3`, `nemotron-maxq-16gb`, `nemotron-ablit`, plus aliases `1`–`8`. |
| `--model <path>` | Use a local GGUF instead of downloading. |
| `--skip-model` | Install engine/scripts only. |
| `--prebuilt` / `--source` | Force prebuilt binary / local compile. |
| `--yes`, `-y` | Non-interactive; accept hardware-derived defaults. |
| `--accept-abliterated` | Required consent in `--yes` mode for "ablit" presets. |
| `--with-mmproj` | Also fetch the 0.85 GiB Qwen3.8 vision projector. |
| `--corpus <dir>` | Your source code, used to seed Nemotron n-gram tables. |
| `--port <port>` | Server port (default 8789). |

There is **no** `--cuda`, `--cpu-only`, `--avx2`, `--avx512` or `--native` flag — those are Unknown
Options and will abort. Use environment variables instead: `GGML_CUDA=ON`, `GGML_NATIVE=0`,
`PORTABLE_X86_ISA=avx2|avx512`, `GGML_VULKAN=ON`, `INSTALL_DEPS=1` (this last one is the only path
that uses `sudo`, to apt-install build tools).

---

## 4. `scripts/llama_modchip.py` — the profiler / command generator

```bash
python3 /home/mskhan/llama-modchip/scripts/llama_modchip.py \
  --model /home/mskhan/LLMs/Qwen3.6-28B-REAP20-A3B-Q4_K_S.gguf \
  --apply-shm --print-cmd
```

**It does not generate tokens, does not start a server, and does not resize `/dev/shm`.** It is a dry run.

Behind the scenes:

1. Parses arguments (see table below).
2. `probe_hardware()` — reads `/proc/cpuinfo`, sysfs L3 topology, `/proc/meminfo`, THP setting.
   On your laptop: 8 physical cores → `decode_threads = batch_threads = 7`, `taskset -c 0,2,4,6,8-11`.
3. `inspect_gguf()` — reads just the GGUF header (magic bytes, metadata). For your Qwen3.6 file it
   found: `qwen35moe`, 40 blocks, hybrid 30 SSM + 10 GQA layers, MoE 8-of-205 experts, 262144 native
   context, Q4_K weights, 15.12 GiB file size. **No model weights are loaded.**
4. `compute_modchip_profile()` — decides 7 threads, 512-token L3 KV window, exact attention
   (`LLAMA_ATTN_WINDOW=0`), RouteCrack enabled with 84/92/96 confidence gates, `q8_0/q4_0` KV,
   `--repack`, and hybrid-SSM checkpoints.
5. `--apply-shm` — `write_modchip_shm()` packs **12 × 32-bit integers = 48 bytes** (magic `MODC`)
   and writes them **in place** to:
   - `/run/user/1000/llama_modchip.bin`
   - `/dev/shm/llama_modchip_1000.bin`

   Both are mode `0600`. A *running* patched `llama-server` has these files mmap'd, so changing the
   contents changes its behaviour live (this is what `./modchip tui` does). Nothing is mounted,
   no hugepages are allocated, and `/dev/shm` size is not changed.
6. `--print-cmd` — prints the ready-to-paste `llama-server` command. It does **not** execute it.

### Exact command produced for your Qwen3.6 file

```bash
RC_TABLES_DIR=/nonexistent LLAMA_ATTN_WINDOW=0 LLAMA_ATTN_SINKS=16 LLAMA_SWA_STRIDE=1 \
LLAMA_KV_L3_WINDOW=512 LLAMA_KV_DISABLE_CHUNKING=1 LLAMA_RC_ENABLED=1 \
LLAMA_RC_CERT_D0=84 LLAMA_RC_CERT_D1=92 LLAMA_RC_CERT_D2=96 \
OMP_PROC_BIND=close OMP_PLACES=cores \
taskset -c 0,2,4,6,8-11 \
/home/mskhan/llama-modchip/workspace/bin/llama-server-nemotron \
  --model /home/mskhan/LLMs/Qwen3.6-28B-REAP20-A3B-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8789 --no-webui --offline \
  -c 262144 -np 1 -t 7 -tb 7 --poll 0 --poll-batch 0 \
  -cb -ub 2048 -b 2048 --flash-attn auto --load-mode none \
  --cache-type-k q8_0 --cache-type-v q4_0 --repack \
  --cache-reuse 256 --slot-save-path /dev/shm \
  --ctx-checkpoints 64 --checkpoint-min-step 128 \
  --spec-type routecrack2 \
  --reasoning-format deepseek --reasoning-budget 0
```

Reading the important parts:

| Part | Why |
|---|---|
| `RC_TABLES_DIR=/nonexistent` | Never load Nemotron vocabulary tables into a Qwen model. |
| `taskset -c 0,2,4,6,8-11` | Pin to physical cores, skip SMT siblings. |
| `-t 7 -tb 7 --poll 0 --poll-batch 0` | Repo rule: physical cores − 1, no busy polling. Claimed +25.9% prefill / +31.3% decode. |
| `-c 262144` | ⚠️ **Too big for 18 GiB available RAM.** This is the model's native context, copied blindly. Change to `-c 8192` or `-c 16384`. |
| `-ub 2048 -b 2048` | Microbatch/batch. On this RAM size try `-ub 512 -b 1024`. |
| `-ctk q8_0 -ctv q4_0` | Compressed KV cache, about 8.3 KB/token for this model. 262K tokens ≈ **2 GB KV** on top of 15.1 GB weights. |
| `--slot-save-path /dev/shm` | Restore old chats from RAM quickly. |
| `--spec-type routecrack2` | Speculative decoding. Benchmark against `--spec-type none`. |
| `--no-webui --offline` | No built-in web UI, no network calls. Use the separate gateway for a web UI. |
| **No `-ngl`** | This script is CPU-only by design. Add `-ngl` manually for CUDA. |

### All profiler flags

| Flag | Meaning |
|---|---|
| `--model`, `-m` | GGUF to inspect (required unless `--audit`). |
| `--audit m1.gguf m2.gguf` | Compare several models' compatibility. |
| `--apply-shm` | Write the 48-byte live-control file. |
| `--print-cmd` | Print the launch command without running it. |
| `--port` | Server port to embed (default 8789 here). |
| `--swa N` | Opt into sliding-window attention of N tokens (default `0` = exact). |
| `--json` | Machine-readable profile. |
| `--bench` | Run A/B test with `llama-bench` (needs `--bench-bin`). |
| `--bench-bin <path>` | Path to `llama-bench`. |
| `--tui` | Interactive curses controller for live SHM knobs. |

---

## 5. `./modchip` — the everyday CLI

| Command | What it does |
|---|---|
| `./modchip setup` | Calls `install.sh` (engine + model selection). |
| `./modchip download-bin --all-tools` | Downloads SHA-verified server, CLI and bench binaries. |
| `./modchip start [model.gguf]` | Starts `start_server.sh` + gateway in the background; waits for `/health`. Runs the installer first if nothing is installed. |
| `./modchip stop` | Stops both background processes. |
| `./modchip restart [model.gguf]` | Stop + start. |
| `./modchip chat -p "hello"` | Terminal chat with live tok/s telemetry. |
| `./modchip bench` | Two-stage speed test + scorecard. |
| `./modchip eval` | Quality / tool-calling / needle benchmark. |
| `./modchip web` | Opens the web HUD at `http://127.0.0.1:8787`. |
| `./modchip tui` | Live controller: `SPACE` toggles knobs, `A` re-syncs SHM, `Q` quits. |
| `./modchip status` | Server health, THP state, per-user SHM info. |
| `./modchip checksums` | Pinned SHA-256 list. |
| `./modchip claude` | Wire Claude Code CLI to the local engine. |
| `./modchip quantize` | Quantize helper. |

Typical session:

```bash
cd /home/mskhan/llama-modchip
MODCHIP_CTX_SIZE=16384 ./modchip start /home/mskhan/LLMs/NeoHorse-1-4B-Q5_K_M.gguf
./modchip chat
./modchip web          # optional, opens HUD on :8787
./modchip stop
```

Useful environment overrides for `./modchip start`:
`MODCHIP_CTX_SIZE`, `MODCHIP_UBATCH_SIZE`, `MODCHIP_SLOT_SAVE_PATH`, `MODCHIP_SPEC_TYPE`,
`MODCHIP_EXTRA_ARGS` (the easiest way to inject `-ngl 99`), `LLAMA_RC_ENABLED=0`, `SERVER_PORT`.

---

## 6. All other scripts in this repo

| Path | Purpose | When to use |
|---|---|---|
| `scripts/llama_modchip.py` | Hardware + GGUF profiler, SHM writer, command generator | Before every model change |
| `scripts/check_cpu_readiness.py` | Human-readable CPU/RAM/bandwidth report and preset advice | First-time hardware check; `./modchip status` |
| `scripts/_cpu_topology.py` | Helper for L3 domains / physical core enumeration (imported by the two above) | Rarely directly |
| `scripts/chat.py` | HTTP chat client; shows TTFT, decode tok/s, draft acceptance | `./modchip chat` |
| `scripts/build_trunk_quant.sh` | Re-quantize trunk tensors (Nemotron tooling) | Advanced model prep |
| `scripts/build_zen5_llama.sh` | Clone + patch + CMake build for llama-modchip | Source installs |
| `scripts/gguf_tensor_type.py` | Report dominant tensor quant type in a GGUF | Debugging quant compatibility |
| `scripts/prewarm_agent_prefix.py` | Restore slot 0 and warm an agent prompt via gateway | Agent workflows |
| `scripts/fill_kv_cache_256k.py` | Fills KV up to 256K in steps and logs speed/acceptance | Long-context stress tests |
| `scripts/preflight_release_check.py` | Release-CI gate (version, hashes, invariants) | Maintainers only |
| `gateway/sse_gateway.py` | Web chat + OpenAI/Anthropic-compatible proxy + HUD | `./modchip web` |
| `speculative/` | RouteCrack research code and C++ headers mirrored into the patch | Developers only |
| `kv_cache/` | HARP, EAKV, FNV acceleration tests | Developers only |
| `benchmarks/` | Quality gates, perplexity/KLD eval, result JSONs | Reading claimed results |
| `windows/`, `apple-silicon/`, `fedora/`, `systemd/` | Platform-specific launchers | Other OSes only |

---

## 7. Which of YOUR models to use

All paths are in `/home/mskhan/LLMs/`. Estimates assume patched AVX2 build, 7 threads, 4–32K context,
`q8_0/q4_0` KV, flash attention on. Your GPU has ~5.7 GB free.

| Model | Size | Best use | Suggested launch | Est. tok/s |
|---|---:|---|---|---|
| `NeoHorse-1-4B-Q5_K_M.gguf` | 3.07 GB | 🏆 **Fastest overall** — full GPU | `-ngl 99 -c 16384 -fa on -ctk q8_0 -ctv q4_0` | 30–70 |
| `LFM2.5-8B-A1B-MXFP4_MOE.gguf` | 5.35 GB | Fast MoE, tight full-GPU fit | `-ngl 99 -c 4096 -ub 256 -fa on -ctk q4_0 -ctv q4_0`; lower `ngl` if OOM | 20–50 GPU / 8–20 hybrid |
| `ornith-1.0-9b-Q4_K_M.gguf` | 5.63 GB | Nearly fills VRAM; compare partial offload vs CPU | `-ngl 40–80 -c 4096 -ub 256` | 8–25 partial / 4–10 CPU |
| `gpt-oss-20b-MXFP4.gguf` | 12.1 GB | Good RAM-tier quality; standard ggml, GPU-friendly | CPU-first: `-ngl 0 -c 32768 -ub 512` | 5–12 CPU |
| `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | 14.2 GB | Best quality-per-RAM MoE candidate | `-ngl 0 -c 16384 -ub 512 --repack` | 4–12 |
| `LFM2.5-24B-A2B-Q4_K_M.gguf` | 14.5 GB | Another strong RAM-tier MoE | same as above | 4–12 |
| `Qwen3.6-28B-REAP20-A3B-Q4_K_S.gguf` | 15.1 GiB | Tight; hybrid SSM+MoE. CPU only | `-ngl 0 -c 8192 -ub 512 -ctk q4_0 -ctv q4_0` | 3–9 |
| `qwen3.8-flash-next-Q4.gguf` | 15.9 GiB | ⚠️ Highest swap/refault risk here | `-ngl 0 -c 4096 -ub 256` | 2–6 |

Notes:

- **Only the 3–5 GB models can fully fit in 6 GB VRAM** once KV cache and compute buffers are counted.
- The repo's official presets are *different files*: `Qwen3.8-Flash-Next-MaxQ-32GB/64GB`,
  `DeepSeek-V4-MaxQ-v4` and `Nemotron-3.5-30B` (VQ3/MaxQ/Fractal). Your files are standard Q4/MXFP4
  GGUFs. They load fine, but published RouteCrack speedups may not transfer.
- LFM2 and gpt-oss have **no model-specific preset** in this repo; they are just standard GGUFs.
- Qwen3.6 appears only in a community leaderboard row, not as an official preset.
- Avoid `Nemotron VQ3-64` on this laptop — it requires AVX-512 VBMI. `MaxQ-16GB` (13.39 GiB) is the
  right 16 GB preset for AVX2.

Reference points from the project (different, bigger hardware): CPU-only laptop Ryzen 7 5700U with
Qwen3.6 A3B Q2_K_XL reached ~14.8 tok/s; Qwen3.8 MaxQ-64GB on 16-core Zen 5 reached 15.34 tok/s
with RouteCrack vs 8.67 without. Treat these as upper context, not expectations for your laptop.

---

## 8. Speed checklist (highest impact first)

1. **Keep context small.** Speed falls hard as KV grows; the repo measured an 82% decode drop from
   <1K tokens to 13.4K tokens on a 30B-A3B model. Start at 4K–8K for the 15–17 GB models.
2. **Put small models fully on GPU** with `-ngl 99` (requires the CUDA build). For MoE models on
   bigger GPUs the repo prefers `-ngl 99 --n-cpu-moe N -ot '\.ffn_.*_exps\.=CPU' --load-mode none`,
   but that is documented for 8–16 GB GPUs, not 6 GB.
3. **Use 7 threads**: `-t 7 -tb 7 --poll 0 --poll-batch 0`. Test `-t 4` too — hybrid P/E cores
   sometimes prefer fewer threads.
4. **Compressed KV + flash attention**: `-ctk q8_0 -ctv q4_0 -fa on` (or q4_0/q4_0 if tight).
5. **Keep `--repack` and `--load-mode none`** with the patched engine. On *unpatched* upstream
   `llama.cpp`, use `--no-repack` for Qwen3.8 models.
6. **Set `RC_TABLES_DIR=/nonexistent` for every non-Nemotron model** and A/B test
   `--spec-type routecrack2` vs `--spec-type none`.
7. **Transparent HugePages = `madvise`** (already the case here). Do not use `always`.
   Optional, needs sudo, resets on reboot:
   ```bash
   echo madvise       | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
   echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
   ```
8. **Close browsers/VMs** when using the 15–17 GB models — you have no swap, so pressure means OOM.
9. **Benchmark, don't guess**:
   ```bash
   ./modchip bench
   python3 scripts/llama_modchip.py -m MODEL.gguf --bench --bench-bin workspace/bin/llama-bench
   python3 scripts/chat.py -p "Write 200 words about KV cache"   # shows live decode tok/s
   ```

---

## 9. Common problems

| Symptom | Fix |
|---|---|
| `workspace/bin/llama-server-nemotron: No such file` | Run `./install.sh --prebuilt --model … --yes` or `./modchip download-bin`. |
| `illegal instruction` | You built AVX-512 for an AVX2-only CPU. Rebuild with `PORTABLE_X86_ISA=avx2`. |
| Process killed / severe slowdown on 16–17 GB models | Reduce `-c` to 4096–8192, `-ub` to 256, use `MODCHIP_SLOT_SAVE_PATH=/home/mskhan/llama-modchip/workspace/cache` instead of `/dev/shm`. |
| `cache_reuse is not supported by this context` | Expected for some DeepSeek-style architectures; not fatal. |
| CUDA build fails with nvcc errors | Toolkit is 12.0; project docs use 12.4. Install newer toolkit, clean `build/`, add `-DCMAKE_CUDA_ARCHITECTURES=89`. |
| Weird results loading Nemotron tables with Qwen | Always export `RC_TABLES_DIR=/nonexistent`. |
| Web UI missing | `--no-webui` is deliberate; use `./modchip web` (gateway on :8787). |
| Want to see what SHM currently says | `xxd /run/user/1000/llama_modchip.bin` (48 bytes; first 4 = `CDOM`). |

---

## 10. Where to read more

- `README.md` — features, RAM tiers, hardware table, community leaderboard.
- `ARCHITECTURE.md` — deep dive into all 15 optimizations and benchmark math.
- `FAQ.md` — runtime defaults and warnings (Q1–Q16).
- `docs/HARDWARE_OPTIMIZATIONS.md` — CPU/GPU backend matrix.
- `speculative/ROUTECRACK_TECHNIQUES.md` — how RouteCrack 2.0 works.
- `benchmarks/QUALITY_GATE_VQ3_64.md` — quality-gate results.
- `paper/llama-modchip-paper.pdf` — the paper.

*Guide written for the v3.2.0 release and the detected hardware shown in section 1.*
