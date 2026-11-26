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
