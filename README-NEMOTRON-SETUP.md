# Nemotron MaxQ-16GB — Quick Setup

Official optimized checkpoint for this laptop (i5-12450HX / 23 GB RAM / AVX2).
Run these after your internet connection is stable.

## 1. Full setup (source build)

```bash
cd /home/mskhan/llama-modchip

# Clean up empty folders left by the interrupted run
rm -rf /home/mskhan/llama-modchip/workspace

# Clone + patch + build engine, download and verify the 13.4 GB model,
# build RouteCrack tables, and generate start_server.sh / start_gateway.sh
./install.sh --preset nemotron-maxq-16gb --source --yes
```

Optional — keep a log you can watch from another terminal:

```bash
cd /home/mskhan/llama-modchip
./install.sh --preset nemotron-maxq-16gb --source --yes > /tmp/opencode/nemotron-install.log 2>&1

# In another terminal:
tail -f /tmp/opencode/nemotron-install.log
```

## 2. If the llama.cpp git clone keeps failing

```bash
cd /home/mskhan/llama-modchip
rm -rf /home/mskhan/llama-modchip/workspace
./install.sh --preset nemotron-maxq-16gb --prebuilt --yes
```

This skips the git clone and compile; it only downloads a small verified binary and the model.
RouteCrack still works in online mode (no offline n-gram tables).

## 3. Verify and run

```bash
cd /home/mskhan/llama-modchip

# Show the recommended launch command for the new model
python3 scripts/llama_modchip.py \
  -m workspace/models/NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB.gguf \
  --apply-shm --print-cmd

# Start server (:8789) and web gateway/HUD (:8787)
./modchip start

# Chat in terminal, or open http://127.0.0.1:8787 for the web HUD
./modchip chat
./modchip web

# Benchmark tokens/sec, then stop
./modchip bench
./modchip stop
```

## Resume after an interrupted download

Re-run the same command; the model download resumes and verified files are reused:

```bash
cd /home/mskhan/llama-modchip
./install.sh --preset nemotron-maxq-16gb --source --yes
```

Only delete `workspace/llama.cpp` manually if the git clone itself is corrupted.

## CUDA build (RTX 4050 Laptop / sm_89)

Rebuild with the GPU architecture pinned to 89 so nvcc does not try `-arch=native`:

```bash
cd /home/mskhan/llama-modchip

rm -rf workspace/llama.cpp/build

cmake -S workspace/llama.cpp -B workspace/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_NUMBER=12604 \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
  -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF -DGGML_AVX512_VNNI=OFF \
  -DGGML_OPENMP=ON \
  "-DCMAKE_C_FLAGS=-march=haswell -mno-avx512f" \
  "-DCMAKE_CXX_FLAGS=-march=haswell -mno-avx512f" \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build workspace/llama.cpp/build -j8 \
  --target llama-server llama-cli llama-bench llama-perplexity llama-quantize

cp workspace/llama.cpp/build/bin/llama-server workspace/bin/llama-server-nemotron
export LD_LIBRARY_PATH=/home/mskhan/llama-modchip/workspace/llama.cpp/build/bin:$LD_LIBRARY_PATH

workspace/bin/llama-server-nemotron --version | grep -i cuda
workspace/bin/llama-server-nemotron --list-devices

MODCHIP_EXTRA_ARGS="-ngl 99" ./modchip start /home/mskhan/LLMs/NeoHorse-1-4B-Q5_K_M.gguf

MODCHIP_EXTRA_ARGS="-ngl 99" ./modchip start /home/mskhan/LLMs/Qwen3.6-28B-REAP20-A3B-Q4_K_S.gguf

MODCHIP_EXTRA_ARGS="-ngl 99" ./modchip start /home/mskhan/LLMs/LFM2.5-24B-A2B-Q4_K_M.gguf


```
