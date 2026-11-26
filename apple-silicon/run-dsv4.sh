#!/usr/bin/env bash
# run-dsv4.sh — DeepSeek-V4-Flash-0731-MaxQ-v4 (284B, 57.99 GiB, 100% std ggml) on Apple Silicon Mac (64 GB).
#
# Build the engine first with ./build-dsv4-engine.sh (unified llama-modchip v3.2.0:
# llama.cpp 46eb1742d + patches/llama-modchip-v3.2.patch). Required for ConeCollapse
# ffn_down_exps.bias and RouteCrack 2.0 support.
#
# PROFILE=fast    (default) experts of 5 middle layers on the CPU (~51.8 GB wired on GPU), needs
#                 `sudo sysctl iogpu.wired_limit_mb=57344` (resets at reboot).
#                 Note: the 27.0 tok/s tg128 / 169.4 pp512 M5 Max measurement in results/dsv4-m5max.md
#                 was recorded on the earlier withdrawn DeepSeek-V4-Flash-0731-Fractal (56.56 GiB) build;
#                 MaxQ-v4 (57.99 GiB) uses the same hybrid placement profile and awaits fresh M5 Max numbers.
# PROFILE=nosudo  experts of 8 pure-IQ1_S layers on the CPU. ~49 GB on the GPU, which fits the default
#                 Metal limit. llama-bench tg128 23.7 tok/s.
#
# Do not use --n-cpu-moe: it takes layers 0..N-1, and with those on the CPU decode is 13.6 tok/s
# (--n-cpu-moe 4) instead of 27.0. Why is not established; see results/dsv4-m5max.md.
# Default SPEC=routecrack2 uses the RouteCrack 2.0 L1 Micro-Arbiter (set SPEC=none if measuring pure n=1 decode).
#
# Environment: LLAMA_DIR, MODEL, PORT (8789), CTX (32768), THREADS (12; measured best, 18 drops to 4.1 tok/s).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${LLAMA_DIR:-}" ]]; then
    if [[ -x "$HERE/../workspace/llama.cpp/build/bin/llama-server" ]]; then
        LLAMA_DIR="$HERE/../workspace/llama.cpp/build"
    elif [[ -x "$HERE/../workspace/llama.cpp-v3/build/bin/llama-server" ]]; then
        LLAMA_DIR="$HERE/../workspace/llama.cpp-v3/build"
    elif [[ -x "$HERE/../llama.cpp/build/bin/llama-server" ]]; then
        LLAMA_DIR="$HERE/../llama.cpp/build"
    else
        LLAMA_DIR="$HERE/../workspace/llama.cpp-dsv4/build"
    fi
fi
if [[ -z "${MODEL:-}" ]]; then
    if [[ -f "$HOME/models/dsv4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf" ]]; then
        MODEL="$HOME/models/dsv4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
    elif [[ -f "$HERE/../workspace/models/dsv4-maxq-v4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf" ]]; then
        MODEL="$HERE/../workspace/models/dsv4-maxq-v4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
    else
        MODEL="$HOME/models/dsv4/DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
    fi
fi
PORT="${PORT:-8789}"
CTX="${CTX:-32768}"
THREADS="${THREADS:-12}"
PROFILE="${PROFILE:-fast}"
SPEC="${SPEC:-routecrack2}"

case "$PROFILE" in
    fast)   cpu_layers='10|12|14|16|18'; need_mb=57344 ;;
    nosudo) cpu_layers='10|12|14|16|18|20|22|24'; need_mb=0 ;;
    *) echo "Unknown PROFILE=$PROFILE (use fast or nosudo)" >&2; exit 1 ;;
esac

if [[ ! -f "$MODEL" ]]; then echo "Model not found: $MODEL" >&2; exit 1; fi
if [[ ! -x "$LLAMA_DIR/bin/llama-server" ]]; then echo "llama-server not found in $LLAMA_DIR/bin" >&2; exit 1; fi

limit="$(sysctl -n iogpu.wired_limit_mb)"
if (( need_mb > 0 && limit < need_mb )); then
    echo "PROFILE=fast needs iogpu.wired_limit_mb >= $need_mb (now $limit)." >&2
    echo "Run: sudo sysctl iogpu.wired_limit_mb=$need_mb   or use PROFILE=nosudo" >&2
    exit 1
fi

echo "[run-dsv4] profile=$PROFILE cpu_expert_layers=$cpu_layers ctx=$CTX threads=$THREADS port=$PORT" >&2
echo "[run-dsv4] if the log shows kIOGPUCommandBufferCallbackErrorOutOfMemory, output is invalid: stop and use PROFILE=nosudo" >&2
SPEC_ARGS=()
[[ "$SPEC" != "none" ]] && SPEC_ARGS=(--spec-type "$SPEC")
export RC_TABLES_DIR="/nonexistent"
exec "$LLAMA_DIR/bin/llama-server" \
    --model "$MODEL" \
    --alias deepseek-v4-flash \
    --host 127.0.0.1 --port "$PORT" \
    --cors-origins localhost \
    --ctx-size "$CTX" \
    --flash-attn on \
    -ctk q8_0 -ctv q8_0 \
    --n-gpu-layers 99 \
    --override-tensor "blk\.(${cpu_layers})\.ffn_.*_exps\.=CPU" \
    --fit off \
    --parallel 1 \
    --jinja \
    --threads "$THREADS" \
    "${SPEC_ARGS[@]}" \
    "$@"
