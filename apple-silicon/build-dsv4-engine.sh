#!/usr/bin/env bash
# build-dsv4-engine.sh — Build unified llama-modchip v3.2 engine on Apple Silicon (Metal + ARM64 NEON SDOT + RouteCrack 2.0 + Quantized Split-KV + HARP-Cache v2).
# Supports Qwen3.8-Flash-Next, DeepSeek-V4-Flash, Nemotron-3.5-30B, and Inkling-2-Mini with a single source tree.
#
# Environment:
#   LLAMA_SRC  existing llama.cpp clone (default: workspace/llama.cpp-src, cloned if missing)
#   DEST       worktree to create (default: workspace/llama.cpp-v3; must not exist)
#   JOBS       build parallelism (default: number of CPUs)
#   BUILD      0 = prepare the source tree only
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
BASE=46eb1742db6d2caec1bfa848c5702b1bdcc815e5
LLAMA_SRC="${LLAMA_SRC:-$REPO/workspace/llama.cpp-src}"
DEST="${DEST:-$REPO/workspace/llama.cpp-v3}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

if [[ ! -d "$LLAMA_SRC/.git" && ! -f "$LLAMA_SRC/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
fi
if ! git -C "$LLAMA_SRC" cat-file -e "${BASE}^{commit}" 2>/dev/null; then
    git -C "$LLAMA_SRC" fetch --depth=1 origin "$BASE"
fi
if [[ -e "$DEST" ]]; then
    echo "$DEST already exists; remove it (git -C $LLAMA_SRC worktree remove $DEST) or set DEST" >&2
    exit 1
fi

git -C "$LLAMA_SRC" worktree add --detach "$DEST" "$BASE"
cd "$DEST"

git apply "$REPO/patches/llama-modchip-v3.2.patch"

if [[ "${BUILD:-1}" == 0 ]]; then
    echo "[build-modchip-v3] source tree ready in $DEST (BUILD=0)" >&2
    exit 0
fi

export HF_UI_VERSION="${HF_UI_VERSION:-b12604}"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_NUMBER=12604 -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_CURL=OFF
cmake --build build -j "$JOBS" --target llama-server llama-bench llama-cli llama-perplexity llama-quantize test-backend-ops test-quantize-fns test-llama-archs

echo "[build-modchip-v3] built $DEST/build/bin" >&2
