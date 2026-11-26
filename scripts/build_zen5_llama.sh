#!/usr/bin/env bash
# build_zen5_llama.sh — Build Unified llama-modchip v3.2 Engine (RouteCrack 2.0 + Quantized Split-KV + HARP-Cache v2)
# Base commit : ggml-org/llama.cpp 46eb1742db6d2caec1bfa848c5702b1bdcc815e5
# Patch       : patches/llama-modchip-v3.2.patch
# Architectures: Qwen3.8-Flash-Next (qwen4exp), DeepSeek-V4 (deepseek4), Nemotron-3.5 (nemotron_h_moe), Qwen3.5-MoE, Inkling-2-Mini (inkling2)
# Speculative  : RouteCrack 2.0 (--spec-type routecrack2: MTP / RCPredictor + Two-Zone Suffix + Stockfish-PVS + L1 Micro-Arbiter)
# Custom Types : GGML_TYPE_FRACTAL_64 = 44 (30 B/64, CPU + Metal GPU), GGML_TYPE_VQ3_64 = 45 (24 B/64, CPU)
#                (Upstream GGML_TYPE_Q2_0 = 42 and GGML_TYPE_F8_E4M3 = 43 remain 100% untouched)
set -euo pipefail

LLAMA_DIR="${1:-./llama.cpp}"
BUILD_DIR="${BUILD_DIR:-${LLAMA_DIR}/build}"
LLAMA_COMMIT="46eb1742db6d2caec1bfa848c5702b1bdcc815e5"
LLAMA_BUILD_NUM="12604"
export HF_UI_VERSION="${HF_UI_VERSION:-b${LLAMA_BUILD_NUM}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/llama-modchip-v3.2.patch"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
NPROC="$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"

if [[ "$OS_NAME" == "Darwin" ]]; then
    if [[ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" == "1" ]]; then
        echo "[!] WARNING: Running inside a Rosetta x86_64 shell on Apple Silicon! Re-run in a native arm64 terminal ('arch -arm64 bash') for NEON SDOT + Metal performance." >&2
    fi
fi

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "[!] ERROR: Unified patch missing at $PATCH_FILE" >&2
    exit 1
fi

reset_tree() {
    git -C "$LLAMA_DIR" checkout HEAD -- .
    rm -f \
        "$LLAMA_DIR/ggml/src/vq3_codebook.h" \
        "$LLAMA_DIR/common/rc_predictor.h" \
        "$LLAMA_DIR/common/rc_stockfish.h" \
        "$LLAMA_DIR/common/rc_trie_v2.h" \
        "$LLAMA_DIR/common/rc_arbiter_l1.h" \
        "$LLAMA_DIR/common/rc_routecrack2.h" \
        "$LLAMA_DIR/tools/server/rc_harp_radix.h"
    while read -r created_file; do
        [[ -n "$created_file" ]] && rm -f "$LLAMA_DIR/$created_file"
    done < <(git -C "$REPO_ROOT" apply --summary "$PATCH_FILE" 2>/dev/null | awk '/create mode/{print $4}')
}

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "[*] Cloning llama.cpp into $LLAMA_DIR..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
else
    CUR_COMMIT="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null || true)"
    if [[ "$CUR_COMMIT" != "$LLAMA_COMMIT" ]]; then
        echo "[*] Existing $LLAMA_DIR is at ${CUR_COMMIT:-unknown}; checking out pinned commit $LLAMA_COMMIT..."
        reset_tree || true
        git -C "$LLAMA_DIR" fetch --depth=1 origin "$LLAMA_COMMIT"
        git -C "$LLAMA_DIR" checkout "$LLAMA_COMMIT"
    fi
fi

if [[ "${RESET_LLAMA_TREE:-0}" == "1" ]]; then
    echo "[*] RESET_LLAMA_TREE=1: resetting $LLAMA_DIR working tree to clean $LLAMA_COMMIT..."
    reset_tree
fi

echo "[*] Applying unified llama-modchip v3.2 patch ($(basename "$PATCH_FILE"))..."
if git -C "$LLAMA_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "[i] llama-modchip-v3.2.patch already applied cleanly in $LLAMA_DIR."
elif git -C "$LLAMA_DIR" apply --check "$PATCH_FILE" 2>/dev/null; then
    git -C "$LLAMA_DIR" apply "$PATCH_FILE"
    echo "[+] llama-modchip-v3.2.patch applied cleanly."
else
    echo "[*] Resetting $LLAMA_DIR working tree to clean $LLAMA_COMMIT before applying llama-modchip-v3.2.patch..."
    reset_tree
    if git -C "$LLAMA_DIR" apply --check "$PATCH_FILE"; then
        git -C "$LLAMA_DIR" apply "$PATCH_FILE"
        echo "[+] llama-modchip-v3.2.patch applied cleanly."
    else
        echo "[!] ERROR: Failed to apply $PATCH_FILE onto $LLAMA_DIR (expected commit $LLAMA_COMMIT)." >&2
        exit 1
    fi
fi

SHARED_LIBS_DEFAULT="ON"
if [[ "${PORTABLE_STATIC:-0}" == "1" ]]; then
    SHARED_LIBS_DEFAULT="OFF"
fi
CMAKE_ISA_ARGS=(-DGGML_NATIVE="${GGML_NATIVE:-ON}" -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-${SHARED_LIBS_DEFAULT}}")

if [[ "$OS_NAME" == "Darwin" ]]; then
    METAL_OPT="${GGML_METAL:-ON}"
    OPENMP_OPT="${GGML_OPENMP:-OFF}"
    BLAS_OPT="${GGML_BLAS:-ON}"
    if [[ "$ARCH_NAME" == "arm64" ]]; then
        echo "[*] Detected macOS Apple Silicon (arm64) — enabling ARM64 NEON SDOT 1x{8,4,3,2} (FRACTAL_64 + VQ3_64) + Metal=${METAL_OPT} (FRACTAL_64 native GPU) + BLAS=${BLAS_OPT}..."
    else
        echo "[*] Detected macOS ($ARCH_NAME) — Metal=${METAL_OPT}, BLAS=${BLAS_OPT}..."
    fi
    CMAKE_ISA_ARGS+=(
        -DGGML_OPENMP="${OPENMP_OPT}"
        -DGGML_METAL="${METAL_OPT}"
        -DGGML_METAL_EMBED_LIBRARY="${GGML_METAL_EMBED_LIBRARY:-ON}"
        -DGGML_BLAS="${BLAS_OPT}"
    )
elif [[ "$ARCH_NAME" == "aarch64" || "$ARCH_NAME" == "arm64" ]]; then
    OPENMP_OPT="${GGML_OPENMP:-ON}"
    echo "[*] Detected Linux ARM64 ($ARCH_NAME) — enabling ARMv8 NEON SDOT 1x{8,4,3,2} (FRACTAL_64 + VQ3_64)..."
    CMAKE_ISA_ARGS+=(-DGGML_OPENMP="${OPENMP_OPT}")
else
    CPU_FLAGS=""
    if [[ -r /proc/cpuinfo ]]; then
        CPU_FLAGS="$(grep -m1 -E '^flags' /proc/cpuinfo || true)"
    fi
    OPENMP_OPT="${GGML_OPENMP:-ON}"
    CMAKE_ISA_ARGS+=(-DGGML_OPENMP="${OPENMP_OPT}")

    PORTABLE_LINK_ARGS=()
    if [[ "${PORTABLE_STATIC:-0}" == "1" ]]; then
        STATIC_GOMP="$("${CXX:-g++}" -print-file-name=libgomp.a 2>/dev/null || true)"
        PORTABLE_LINK_ARGS=(
            -DLLAMA_OPENSSL=OFF
            -DLLAMA_CURL=OFF
            -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
        )
        if [[ -n "$STATIC_GOMP" && "$STATIC_GOMP" == /* && -f "$STATIC_GOMP" ]]; then
            PORTABLE_LINK_ARGS+=(-DOpenMP_gomp_LIBRARY="${STATIC_GOMP}")
        fi
    fi

    if [[ "${PORTABLE_X86_ISA:-}" == "avx2" ]]; then
        echo "[*] Building portable x86-64-v3 (Haswell+: AVX2 + FMA + F16C + BMI2, zero AVX-512 instructions)..."
        CMAKE_ISA_ARGS=(
            -DGGML_NATIVE=OFF
            -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-${SHARED_LIBS_DEFAULT}}"
            -DGGML_OPENMP="${OPENMP_OPT}"
            -DGGML_AVX512=OFF
            -DGGML_AVX512_VBMI=OFF
            -DGGML_AVX512_VNNI=OFF
            -DGGML_AVX512_BF16=OFF
            -DGGML_AVX_VNNI=OFF
            -DGGML_AVX2=ON
            -DGGML_FMA=ON
            -DGGML_F16C=ON
            -DGGML_BMI2=ON
            -DCMAKE_C_FLAGS="-march=haswell -mno-avx512f"
            -DCMAKE_CXX_FLAGS="-march=haswell -mno-avx512f"
            "${PORTABLE_LINK_ARGS[@]}"
        )
    elif [[ "${PORTABLE_X86_ISA:-}" == "avx512" ]]; then
        echo "[*] Building portable x86-64-v4 + AVX-512 VBMI/VNNI (Zen 4/5 & Ice Lake+, zero AVX512-BF16)..."
        CMAKE_ISA_ARGS=(
            -DGGML_NATIVE=OFF
            -DBUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-${SHARED_LIBS_DEFAULT}}"
            -DGGML_OPENMP="${OPENMP_OPT}"
            -DGGML_AVX512=ON
            -DGGML_AVX512_VBMI=ON
            -DGGML_AVX512_VNNI=ON
            -DGGML_AVX512_BF16=OFF
            -DGGML_AVX_VNNI=OFF
            -DGGML_AVX2=ON
            -DGGML_FMA=ON
            -DGGML_F16C=ON
            -DGGML_BMI2=ON
            -DCMAKE_C_FLAGS="-march=skylake-avx512 -mavx512vbmi -mavx512vnni"
            -DCMAKE_CXX_FLAGS="-march=skylake-avx512 -mavx512vbmi -mavx512vnni"
            "${PORTABLE_LINK_ARGS[@]}"
        )
    elif echo "$CPU_FLAGS" | grep -q "avx512f" && echo "$CPU_FLAGS" | grep -q "avx512_vnni"; then
        echo "[*] Detected AVX-512 VNNI (AMD Zen 4/5 or Intel Ice Lake+) — enabling AVX-512 tile kernels..."
        CMAKE_ISA_ARGS+=(
            -DGGML_AVX512=ON
            -DGGML_AVX512_VNNI=ON
        )
        if echo "$CPU_FLAGS" | grep -q "avx512_bf16"; then
            CMAKE_ISA_ARGS+=(-DGGML_AVX512_BF16=ON)
        fi
        if echo "$CPU_FLAGS" | grep -qE "avx512_?vbmi"; then
            CMAKE_ISA_ARGS+=(-DGGML_AVX512_VBMI=ON)
        fi
    elif echo "$CPU_FLAGS" | grep -q "avx2"; then
        echo "[*] Detected AVX2 + FMA — enabling AVX2 native kernels (AVX-512 disabled)..."
        CMAKE_ISA_ARGS+=(
            -DGGML_AVX512=OFF
            -DGGML_AVX2=ON
            -DGGML_FMA=ON
            -DGGML_F16C=ON
        )
    else
        echo "[*] Using native architecture auto-detection..."
    fi
fi

VULKAN_OPT="${GGML_VULKAN:-OFF}"
CUDA_OPT="${GGML_CUDA:-OFF}"
case "$VULKAN_OPT" in
    OFF|off|0|FALSE|false|NO|no|"") ;;
    *) echo "[*] Enabling Vulkan GPU backend (-DGGML_VULKAN=${VULKAN_OPT})..." ;;
esac
case "$CUDA_OPT" in
    OFF|off|0|FALSE|false|NO|no|"") ;;
    *) echo "[*] Enabling CUDA GPU backend (-DGGML_CUDA=${CUDA_OPT})..." ;;
esac
CMAKE_ISA_ARGS+=(
    -DGGML_VULKAN="${VULKAN_OPT}"
    -DGGML_CUDA="${CUDA_OPT}"
)

BUILD_TARGETS=(llama-server llama-cli llama-quantize llama-bench llama-perplexity)
if [[ "${BUILD_TESTS:-0}" == "1" ]]; then
    CMAKE_ISA_ARGS+=(-DLLAMA_BUILD_TESTS=ON)
    BUILD_TARGETS+=(test-quantize-fns test-backend-ops test-llama-archs)
fi

echo "[*] Configuring CMake build in ${BUILD_DIR}..."
cmake -S "$LLAMA_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_NUMBER="${LLAMA_BUILD_NUM}" \
    "${CMAKE_ISA_ARGS[@]}"

echo "[*] Building ${BUILD_TARGETS[*]} with ${NPROC} threads..."
cmake --build "$BUILD_DIR" --config Release -j "${NPROC}" --target "${BUILD_TARGETS[@]}"

echo "[+] Build complete: ${BUILD_DIR}/bin/ (${BUILD_TARGETS[*]})"
