#!/usr/bin/env bash
# run-server.sh — llama-server for Nemotron 3.5 Lightning 30B-A3B on Apple Silicon (tuned on an M4 Pro, 24 GB).
#
# The Metal GPU does all the work. The x86 parts of this repo (AVX-512 repack, NTA prefetch,
# hugepages, CCD pinning, the CPU SWA patch) have no effect on a Mac, so stock llama.cpp is used.
#
# Environment overrides:
#   MODEL, MTP_MODEL   GGUF paths (default: the ggml-org files in the LM Studio model folder)
#   PORT               default 8789 (the port that gateway/sse_gateway.py expects)
#   CTX                context size in tokens, default 131072 (262144 is possible, see README)
#   SPEC               routecrack2 (default) | none | ngram | ngram-map-k | mtp
#   N_CPU_MOE          MoE layers to keep on the CPU side. Empty = let --fit decide.
#   UBATCH             physical batch size for prefill, default 1024. 2048 prefills ~15% faster on short
#                      prompts, but at 128K context its 1.8 GiB compute buffer causes a Metal out-of-memory error.
#   CACHE_RAM          host prompt cache in MiB, default 1024 (llama.cpp default is 8192)
#   CHECKPOINTS        Mamba-2 context checkpoints per slot, default 8 (llama.cpp default is 32)
#   KV_TYPE            KV cache type for K and V, default q8_0
#   GATEWAY            1 = raw output for gateway/sse_gateway.py (--skip-chat-parsing). The gateway
#                      parses <think> and <tool_call> text itself. Default 0 (native parsing).
# Extra arguments are passed to llama-server unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-}"
if [[ -z "$WORKSPACE_DIR" && -f "${REPO_ROOT}/.modchip_workspace" ]]; then
    WORKSPACE_DIR="$(head -n 1 "${REPO_ROOT}/.modchip_workspace" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' || true)"
fi
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}/workspace}"

MODEL_DIR="${MODEL_DIR:-$HOME/.lmstudio/models/ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF}"
if [[ -z "${MODEL:-}" && -f "${WORKSPACE_DIR}/run/default_model" ]]; then
    SAVED_MODEL="$(head -n 1 "${WORKSPACE_DIR}/run/default_model" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' || true)"
    [[ -n "$SAVED_MODEL" && -f "$SAVED_MODEL" ]] && MODEL="$SAVED_MODEL"
fi
MODEL="${MODEL:-$MODEL_DIR/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}"
MTP_MODEL="${MTP_MODEL:-$MODEL_DIR/mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf}"
PORT="${PORT:-8789}"
CTX="${CTX:-131072}"
SPEC="${SPEC:-routecrack2}"
N_CPU_MOE="${N_CPU_MOE:-}"
UBATCH="${UBATCH:-1024}"
CACHE_RAM="${CACHE_RAM:-1024}"
CHECKPOINTS="${CHECKPOINTS:-8}"
KV_TYPE="${KV_TYPE:-q8_0}"
GATEWAY="${GATEWAY:-0}"
if [[ -z "${LLAMA_SERVER:-}" ]]; then
    if [[ -x "${WORKSPACE_DIR}/bin/llama-server-nemotron" ]]; then
        LLAMA_SERVER="${WORKSPACE_DIR}/bin/llama-server-nemotron"
        export DYLD_LIBRARY_PATH="${WORKSPACE_DIR}/llama.cpp/build/bin:${DYLD_LIBRARY_PATH:-}"
    elif [[ -x "${WORKSPACE_DIR}/llama.cpp/build/bin/llama-server" ]]; then
        LLAMA_SERVER="${WORKSPACE_DIR}/llama.cpp/build/bin/llama-server"
    elif [[ -x "${REPO_ROOT}/llama.cpp/build/bin/llama-server" ]]; then
        LLAMA_SERVER="${REPO_ROOT}/llama.cpp/build/bin/llama-server"
    else
        LLAMA_SERVER="llama-server"
        echo "[run-server] WARNING: using 'llama-server' from PATH; FRACTAL_64 (44) / VQ3_64 (45) GGUFs and RouteCrack 2.0 require './modchip setup' (or bash apple-silicon/setup.sh)." >&2
    fi
fi
echo "[run-server] binary=$LLAMA_SERVER" >&2

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL" >&2
    exit 1
fi

MODEL_BASE="$(basename "$MODEL")"
MODEL_BASE_LOWER="$(echo "$MODEL_BASE" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODEL_BASE_LOWER" != *nemotron* ]]; then
    export RC_TABLES_DIR="/nonexistent"
fi

if [[ -z "${MODEL_ALIAS:-}" ]]; then
    if [[ "$MODEL_BASE_LOWER" == *qwen3.8* || "$MODEL_BASE_LOWER" == *qwen38* ]]; then
        MODEL_ALIAS="qwen3.8-flash-next"
    elif [[ "$MODEL_BASE_LOWER" == *deepseek* || "$MODEL_BASE_LOWER" == *dsv4* ]]; then
        MODEL_ALIAS="deepseek-v4-flash"
    else
        MODEL_ALIAS="nemotron-3.5-30b-a3b"
    fi
fi

args=(
    --model "$MODEL"
    --alias "$MODEL_ALIAS"
    --host 127.0.0.1
    --port "$PORT"
    # The default is '*' with credentials. Then any web page in the browser can call the API and read the replies.
    --cors-origins localhost
    --ctx-size "$CTX"
    --parallel 1
    --flash-attn on
    --cache-type-k "$KV_TYPE"
    --cache-type-v "$KV_TYPE"
    --batch-size 2048
    --ubatch-size "$UBATCH"
    --cache-ram "$CACHE_RAM"
    --ctx-checkpoints "$CHECKPOINTS"
    --jinja
    --reasoning-format deepseek
)

if [[ -n "$N_CPU_MOE" ]]; then
    args+=(--n-cpu-moe "$N_CPU_MOE")
fi

if [[ "$GATEWAY" == 1 ]]; then
    args+=(--skip-chat-parsing)
fi

case "$SPEC" in
    none)
        ;;
    routecrack2)
        if "$LLAMA_SERVER" --help 2>&1 | grep -q "routecrack2"; then
            args+=(--spec-type routecrack2)
        else
            echo "[run-server] binary lacks routecrack2 (unpatched llama-server); falling back to --spec-type ngram-mod" >&2
            args+=(--spec-type ngram-mod)
        fi
        ;;
    ngram|ngram-mod)
        args+=(--spec-type ngram-mod)
        ;;
    ngram-map-k)
        args+=(--spec-type ngram-map-k)
        ;;
    mtp|draft-mtp)
        if [[ -f "$MTP_MODEL" ]]; then
            args+=(--spec-type draft-mtp --spec-draft-model "$MTP_MODEL")
        else
            args+=(--spec-type draft-mtp)
        fi
        ;;
    *)
        echo "Unknown SPEC=$SPEC (use routecrack2, none, ngram, ngram-map-k or mtp)" >&2
        exit 1
        ;;
esac

echo "[run-server] ctx=$CTX kv=$KV_TYPE spec=$SPEC ubatch=$UBATCH n_cpu_moe=${N_CPU_MOE:-fit} cache_ram=${CACHE_RAM}MiB checkpoints=$CHECKPOINTS gateway=$GATEWAY port=$PORT" >&2
exec "$LLAMA_SERVER" "${args[@]}" "$@"
