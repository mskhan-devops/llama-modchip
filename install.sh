#!/usr/bin/env bash
# ==============================================================================
# install.sh — Interactive Zero-to-Running Installer for llama-modchip (v3.2.0)
# ==============================================================================
# Usage:
#   One-liner (from scratch):
#     curl -sL https://raw.githubusercontent.com/synchopate/llama-modchip/main/install.sh | bash
#
#   From cloned repo:
#     ./install.sh [options]
#
# Options:
#   --workspace <dir>      Target build/model directory (default: ./workspace)
#   --preset <id>          3-Tier RAM Presets:
#                            64 GB Tier: qwen38-64gb | qwen38-64gb-ablit | dsv4-maxq
#                            32 GB Tier: qwen38-32gb | qwen38-32gb-ablit
#                            16 GB Tier: nemotron-vq3 | nemotron-maxq-16gb | nemotron-ablit
#   --model <path>         Use an existing local GGUF file instead of downloading
#   --skip-model           Install engine & scripts without downloading GGUF weights
#   --prebuilt             Download CPU-matched Linux x86_64 v3.2.0 binary (-avx512 / -avx2)
#   --source               Compile unified v3.2.0 engine locally from source (cmake/g++)
#   --yes, -y              Non-interactive mode: accept hardware-matched aligned defaults
#   --accept-abliterated   Confirm opt-in consent when selecting an abliterated preset in --yes mode
#   --with-mmproj          Also download & pass mmproj-Qwen3.8-Flash-Next-BF16.gguf (0.85 GiB vision projector)
#   --corpus <dir>         Optional path to your codebase for Nemotron RouteCrack n-grams
#   --port <port>          llama-server port (default: 8789)
# ==============================================================================
set -euo pipefail

# If piped via `curl ... | bash`, bootstrap the repository first and re-exec
SCRIPT_SOURCE="${BASH_SOURCE[0]:-}"
if [[ -z "$SCRIPT_SOURCE" ]] || [[ ! -f "$(dirname "$SCRIPT_SOURCE")/patches/llama-modchip-v3.2.patch" ]]; then
    echo "=============================================================================="
    echo " [*] Bootstrapping llama-modchip v3.2.0 repository..."
    echo "=============================================================================="
    if [[ ! -d "./llama-modchip" ]]; then
        if command -v git >/dev/null 2>&1; then
            git clone https://github.com/synchopate/llama-modchip.git ./llama-modchip
        else
            mkdir -p ./llama-modchip
            curl -fsSL https://github.com/synchopate/llama-modchip/archive/refs/heads/main.tar.gz \
                | tar -xz --strip-components=1 -C ./llama-modchip
        fi
    fi
    cd ./llama-modchip
    exec bash ./install.sh "$@"
fi

REPO_ROOT="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"

resolve_path() {
    local p="$1"
    if command -v python3 >/dev/null 2>&1; then
        python3 - "$p" <<'PYEOF'
import os, sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PYEOF
    elif [[ "$p" == /* ]]; then
        printf '%s\n' "$p"
    else
        printf '%s/%s\n' "$PWD" "$p"
    fi
}

compute_sha256() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$f" | awk '{print $1}'
    else
        python3 - "$f" <<'PYEOF'
import hashlib, sys
h = hashlib.sha256()
with open(sys.argv[1], "rb") as fp:
    for chunk in iter(lambda: fp.read(1048576), b""):
        h.update(chunk)
print(h.hexdigest())
PYEOF
    fi
}

if [[ -z "${WORKSPACE_DIR:-}" && -f "${REPO_ROOT}/.modchip_workspace" ]]; then
    SAVED_WS="$(head -n 1 "${REPO_ROOT}/.modchip_workspace" 2>/dev/null | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' || true)"
    [[ -n "$SAVED_WS" ]] && WORKSPACE_DIR="$SAVED_WS"
fi
WORKSPACE_DIR="${WORKSPACE_DIR:-${REPO_ROOT}/workspace}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_EXPLICIT=0
[[ -n "$MODEL_PATH" ]] && MODEL_EXPLICIT=1
MODEL_PRESET="${MODEL_PRESET:-}"
SKIP_MODEL="${SKIP_MODEL:-0}"
ENGINE_MODE_EXPLICIT=0
if [[ -n "${USE_PREBUILT:-}" ]]; then
    ENGINE_MODE_EXPLICIT=1
fi
USE_PREBUILT="${USE_PREBUILT:-0}"
NONINTERACTIVE="${NONINTERACTIVE:-0}"
ACCEPT_ABLITERATED="${ACCEPT_ABLITERATED:-0}"
WITH_MMPROJ="${WITH_MMPROJ:-0}"
USER_CORPUS_DIR=""
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8789}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${GATEWAY_PORT:-8787}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --workspace)
            WORKSPACE_DIR="$(resolve_path "$2")"
            shift 2
            ;;
        --preset)
            MODEL_PRESET="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$(resolve_path "$2")"
            MODEL_EXPLICIT=1
            shift 2
            ;;
        --skip-model)
            SKIP_MODEL=1
            shift
            ;;
        --prebuilt)
            USE_PREBUILT=1
            ENGINE_MODE_EXPLICIT=1
            shift
            ;;
        --source)
            USE_PREBUILT=0
            ENGINE_MODE_EXPLICIT=1
            shift
            ;;
        --yes|-y)
            NONINTERACTIVE=1
            shift
            ;;
        --accept-abliterated)
            ACCEPT_ABLITERATED=1
            shift
            ;;
        --with-mmproj)
            WITH_MMPROJ=1
            shift
            ;;
        --corpus)
            USER_CORPUS_DIR="$(resolve_path "$2")"
            shift 2
            ;;
        --port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,27p' "$0"
            exit 0
            ;;
        *)
            echo "[!] Unknown option: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

mkdir -p "$WORKSPACE_DIR/bin" "$WORKSPACE_DIR/models" "$WORKSPACE_DIR/routecrack" "$WORKSPACE_DIR/cache"

check_required_commands() {
    local missing=()
    local cmd
    for cmd in "$@"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            missing+=("$cmd")
        fi
    done
    if [[ ${#missing[@]} -eq 0 ]]; then
        return 0
    fi
    echo "[!] Missing required system tools: ${missing[*]}"
    if [[ "${INSTALL_DEPS:-0}" == "1" ]]; then
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update && sudo apt-get install -y build-essential cmake git curl python3 libomp-dev
        elif command -v dnf >/dev/null 2>&1; then
            sudo dnf install -y gcc-c++ cmake git curl python3 libomp-devel
        elif command -v pacman >/dev/null 2>&1; then
            sudo pacman -Sy --noconfirm base-devel cmake git curl python openmp
        fi
    else
        echo "    Install them with your package manager (or re-run with INSTALL_DEPS=1 ./install.sh):"
        echo "      Debian/Ubuntu : sudo apt-get install -y curl python3 build-essential cmake git libomp-dev"
        echo "      Fedora/RHEL   : sudo dnf install -y curl python3 gcc-c++ cmake git libomp-devel"
        echo "      Arch Linux    : sudo pacman -S curl python base-devel cmake git openmp"
        exit 1
    fi
}

echo "=============================================================================="
echo " STEP 1/5: HARDWARE READINESS AUDIT & 3-TIER RAM CONFIGURATION"
echo "=============================================================================="
check_required_commands curl python3

python3 "${REPO_ROOT}/scripts/check_cpu_readiness.py" || true

PHYS_CORES="$(python3 -c '
import os, platform, subprocess
cores = set()
try:
    with open("/proc/cpuinfo") as f:
        phys_id, core_id = "0", None
        for line in f:
            if line.startswith("physical id"): phys_id = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[1].strip()
                cores.add((phys_id, core_id))
except Exception:
    pass
n = len(cores)
if n == 0 and platform.system() == "Darwin":
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            v = int(subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).strip())
            if v > 0:
                n = v
                break
        except Exception:
            pass
print(n if n > 0 else (os.cpu_count() or 8))
')"

TOTAL_RAM_GB="$(python3 -c '
import glob, platform, subprocess
ram_gb = 0
try:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                ram_gb = int(line.split()[1]) // (1024 * 1024)
                break
except Exception:
    pass
if ram_gb == 0 and platform.system() == "Darwin":
    try:
        ram_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL).strip()) // (1024 ** 3)
    except Exception:
        pass
if ram_gb <= 0:
    ram_gb = 32
if ram_gb > 65536:
    try:
        numa_kb = 0
        for p in glob.glob("/sys/devices/system/node/node*/meminfo"):
            with open(p) as nf:
                for nline in nf:
                    if "MemTotal:" in nline:
                        numa_kb += int(nline.split()[3])
                        break
        if numa_kb > 0:
            ram_gb = numa_kb // (1024 * 1024)
    except Exception:
        pass
print(ram_gb)
')"

detect_x86_prebuilt_asset() {
    local flags=""
    if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
        printf '%s\n' ""
        return
    fi
    if [[ -r /proc/cpuinfo ]]; then
        flags="$(grep -m1 -E '^flags' /proc/cpuinfo || true)"
    fi
    if echo "$flags" | grep -q "avx512f" && echo "$flags" | grep -qE "avx512_?vbmi" && echo "$flags" | grep -qE "avx512_?vnni"; then
        printf '%s\n' "llama-server-modchip-linux-x86_64-avx512"
    elif echo "$flags" | grep -q "avx2"; then
        printf '%s\n' "llama-server-modchip-linux-x86_64-avx2"
    else
        printf '%s\n' ""
    fi
}

X86_PREBUILT_ASSET="$(detect_x86_prebuilt_asset)"
HAS_AVX512_VBMI=0
[[ "$X86_PREBUILT_ASSET" == *-avx512 ]] && HAS_AVX512_VBMI=1

IS_AMD_ZEN12=0
if [[ -r /proc/cpuinfo ]] && grep -qi "AuthenticAMD" /proc/cpuinfo && grep -qE '^cpu family[[:space:]]*:[[:space:]]*23$' /proc/cpuinfo; then
    IS_AMD_ZEN12=1
fi

IS_INTERACTIVE=0
if [[ "$NONINTERACTIVE" != "1" && -t 1 && -r /dev/tty ]]; then
    IS_INTERACTIVE=1
fi

# Hardware-aware safety-aligned default preset across the 3 RAM Tiers (16 GB / 32 GB / 64 GB):
#   >= 60 GB RAM (64 GB Tier) -> Qwen3.8-Flash-Next-MaxQ-64GB (53.99 GiB, RouteCrack 2.0, 15.34–21.38 t/s)
#   30..59 GB RAM (32 GB Tier) -> Qwen3.8-Flash-Next-MaxQ-32GB (24.98 GiB Shard 1 + lazy NVMe PLE, RouteCrack 2.0)
#   21..29 GB RAM (16–24 GB)   -> Nemotron-3.5-30B-A3B-VQ3-64 (AVX-512 VBMI CPU-Turbo) or MaxQ-16GB
#   <= 20 GB RAM (16 GB strict)-> Nemotron-3.5-30B-A3B-MaxQ-16GB (13.39 GiB disk; smallest 16 GB footprint, ~15.0G N=1)
if [[ "$TOTAL_RAM_GB" -ge 60 ]]; then
    DEFAULT_PRESET="qwen38-64gb"
    DEFAULT_MENU_CHOICE="1"
elif [[ "$TOTAL_RAM_GB" -ge 30 ]]; then
    DEFAULT_PRESET="qwen38-32gb"
    DEFAULT_MENU_CHOICE="4"
elif [[ "$TOTAL_RAM_GB" -ge 21 && "$HAS_AVX512_VBMI" == "1" && "$IS_AMD_ZEN12" == "0" ]]; then
    DEFAULT_PRESET="nemotron-vq3"
    DEFAULT_MENU_CHOICE="6"
else
    DEFAULT_PRESET="nemotron-maxq-16gb"
    DEFAULT_MENU_CHOICE="7"
fi

if [[ "$MODEL_EXPLICIT" == "0" && -z "$MODEL_PRESET" ]]; then
    if [[ "$SKIP_MODEL" == "0" && "$IS_INTERACTIVE" == "1" ]]; then
        echo ""
        echo "------------------------------------------------------------------------------"
        echo " MAXQ v3.2.0 3-TIER MODEL MATRIX (Detected RAM: ~${TOTAL_RAM_GB} GiB | Physical Cores: ${PHYS_CORES})"
        echo "------------------------------------------------------------------------------"
        echo "  [64 GB RAM TIER]"
        echo "  1) Qwen3.8-Flash-Next-MaxQ-64GB                 [176.5B (5B Active) · 53.99 GiB · 2 shards]"
        echo "     Default 64GB Flagship (15.34–21.38 t/s RouteCrack 2.0 · PPL 3.43 · 100% std ggml)"
        echo "  2) Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated     [176.5B (5B Active) · 53.99 GiB · 2 shards]"
        echo "     [OPT-IN UNCENSORED] 64GB v3.0-r2 Heretic-2 (0.0818 KLD + blk.48 MTP · RouteCrack 2.0)"
        echo "  3) DeepSeek-V4-Flash-0731-MaxQ-v4               [284B MoE · 57.99 GiB · 3 shards]"
        echo "     284B-A13B Alternative (RouteCrack 2.0 Heavy-MoE Arbiter · 100% std ggml)"
        echo ""
        echo "  [32 GB RAM TIER]"
        echo "  4) Qwen3.8-Flash-Next-MaxQ-32GB                 [176.5B (5B Active) · 24.98 GiB + 10.73 GiB PLE]"
        echo "     Default 32GB Flagship (27.44 GiB peak RSS + lazy NVMe PLE · RouteCrack 2.0)"
        echo "  5) Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated     [176.5B (5B Active) · 24.98 GiB + 10.73 GiB PLE]"
        echo "     [OPT-IN UNCENSORED] 32GB v3.0-r2 Heretic-2 (0.0818 KLD + blk.48 MTP · RouteCrack 2.0)"
        echo ""
        echo "  [16 GB RAM TIER]"
        echo "  6) NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64           [30B MoE · 13.30 GiB · v3.1 CPU-Turbo]"
        echo "     Default 16GB CPU-Turbo (AVX-512 VBMI VQ3_64=45 + RouteCrack 2.0 · 28.2–60.4 t/s)"
        echo "  7) NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB        [30B MoE · 13.39 GiB · 100% std ggml]"
        echo "     Universal 16GB (standard ggml <=42 — upstream llama.cpp & Q2_0-capable runners)"
        echo "  8) NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3  [30B MoE · 13.30 GiB · Opt-In]"
        echo "     [OPT-IN UNCENSORED] 16GB VQ3-64 + RouteCrack 2.0 (refusal removed)"
        echo "------------------------------------------------------------------------------"
        while true; do
            printf " Select model [1-8, Enter=%s]: " "$DEFAULT_MENU_CHOICE"
            read -r USER_MODEL_CHOICE < /dev/tty || USER_MODEL_CHOICE="$DEFAULT_MENU_CHOICE"
            USER_MODEL_CHOICE="${USER_MODEL_CHOICE:-$DEFAULT_MENU_CHOICE}"
            case "$USER_MODEL_CHOICE" in
                1|qwen38-64gb|qwen-64gb)                 MODEL_PRESET="qwen38-64gb"; break ;;
                2|qwen38-64gb-ablit|qwen-64gb-ablit)     MODEL_PRESET="qwen38-64gb-ablit"; break ;;
                3|dsv4-maxq|dsv4-maxq-v4|deepseek)       MODEL_PRESET="dsv4-maxq"; break ;;
                4|qwen38-32gb|qwen-32gb)                 MODEL_PRESET="qwen38-32gb"; break ;;
                5|qwen38-32gb-ablit|qwen-32gb-ablit)     MODEL_PRESET="qwen38-32gb-ablit"; break ;;
                6|nemotron-vq3)                          MODEL_PRESET="nemotron-vq3"; break ;;
                7|nemotron-maxq-16gb|nemotron-16gb)      MODEL_PRESET="nemotron-maxq-16gb"; break ;;
                8|nemotron-ablit|nemotron-abliterated)   MODEL_PRESET="nemotron-ablit"; break ;;
                *)
                    echo "[!] Unrecognized choice '${USER_MODEL_CHOICE}'. Please enter a number from 1 to 8."
                    ;;
            esac
        done
    else
        MODEL_PRESET="$DEFAULT_PRESET"
    fi
fi

MODEL_FAMILY="nemotron"
IS_ABLITERATED=0
MODEL_SHARDS=()
MODEL_SHARD_SIZES=()
MODEL_SHARD_SHA256S=()
MODEL_SUBDIR=""

if [[ "$MODEL_EXPLICIT" == "1" ]]; then
    MODEL_LABEL="$(basename "$MODEL_PATH")"
    if [[ "$MODEL_PATH" == *[Qq]wen3.8* || "$MODEL_PATH" == *[Qq]wen38* ]]; then
        MODEL_FAMILY="qwen38"
    elif [[ "$MODEL_PATH" == *[Dd]eep[Ss]eek* || "$MODEL_PATH" == *dsv4* ]]; then
        MODEL_FAMILY="dsv4"
    fi
    if [[ "$MODEL_PATH" == *[Aa]bliterated* || "$MODEL_PATH" == *-ablit* ]]; then
        IS_ABLITERATED=1
    fi
else
    case "$MODEL_PRESET" in
        1|qwen38-64gb|qwen-64gb|Qwen3.8-Flash-Next-MaxQ-64GB)
            MODEL_PRESET="qwen38-64gb"
            MODEL_FAMILY="qwen38"
            MODEL_LABEL="Qwen3.8-Flash-Next-MaxQ-64GB (176.5B (5B Active) · 53.99 GiB · 2 shards)"
            MODEL_REPO="Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF"
            MODEL_REV="c0e7f6a3df86b2fac578e504b9c8b65ac420f28d"
            MODEL_SUBDIR="qwen38-maxq-64gb"
            MODEL_SHARDS=(
                "Qwen3.8-Flash-Next-MaxQ-64GB-00001-of-00002.gguf"
                "Qwen3.8-Flash-Next-MaxQ-64GB-00002-of-00002.gguf"
            )
            MODEL_SHARD_SIZES=(46449692736 11520055616)
            MODEL_SHARD_SHA256S=(
                "f2b8bc79acb4b4d79033acd91c9c360ba96da92cd5841512befc450e15f4786f"
                "86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e"
            )
            ;;
        2|qwen38-64gb-ablit|qwen-64gb-ablit|Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated)
            MODEL_PRESET="qwen38-64gb-ablit"
            MODEL_FAMILY="qwen38"
            IS_ABLITERATED=1
            MODEL_LABEL="Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated (176.5B (5B Active) · 53.99 GiB · v3.0-r2 Heretic-2 · 0.0818 KLD + blk.48 MTP · 2 shards)"
            MODEL_REPO="Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-GGUF"
            MODEL_REV="0a7cac7fdf7d8c299a5e3979ef79680cfa21b585"
            MODEL_SUBDIR="qwen38-maxq-64gb-ablit"
            MODEL_SHARDS=(
                "Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-00001-of-00002.gguf"
                "Qwen3.8-Flash-Next-MaxQ-64GB-Abliterated-00002-of-00002.gguf"
            )
            MODEL_SHARD_SIZES=(46449692736 11520055616)
            MODEL_SHARD_SHA256S=(
                "1dac27e81a3a73dcec21edea78f224eb48393a726e3f9ee7676c603d3174e82f"
                "86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e"
            )
            ;;
        3|dsv4-maxq|dsv4-maxq-v4|deepseek|DeepSeek-V4-Flash-0731-MaxQ-v4)
            MODEL_PRESET="dsv4-maxq"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-MaxQ-v4 (284B · 57.99 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF"
            MODEL_REV="48356a120e78c01e82f425a74380f46f3925fb9c"
            MODEL_SUBDIR="dsv4-maxq-v4"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-MaxQ-v4-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-MaxQ-v4-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5258336 35543586816 26721192096)
            MODEL_SHARD_SHA256S=(
                "6112e887b21d1fdf7b7196232b528698a6ec2d5a59760a040cb4231f8e8d691b"
                "5d85c740992b70395da5c3e4a856b3fad6ef725271722c8d837b35befe28330d"
                "a8654d4d5fedf748d6f2b6404a8b7f9e19435a8aa8e2bb97ce9a66bc012c0f3a"
            )
            ;;
        4|qwen38-32gb|qwen-32gb|Qwen3.8-Flash-Next-MaxQ-32GB)
            MODEL_PRESET="qwen38-32gb"
            MODEL_FAMILY="qwen38"
            MODEL_LABEL="Qwen3.8-Flash-Next-MaxQ-32GB (176.5B (5B Active) · 24.98 GiB Shard 1 + 10.73 GiB Lazy PLE)"
            MODEL_REPO="Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-GGUF"
            MODEL_REV="ae6abb575d0a09e7b6178258361f9d933cbe3be2"
            MODEL_SUBDIR="qwen38-maxq-32gb"
            MODEL_SHARDS=(
                "Qwen3.8-Flash-Next-MaxQ-32GB-00001-of-00002.gguf"
                "Qwen3.8-Flash-Next-MaxQ-32GB-00002-of-00002.gguf"
            )
            MODEL_SHARD_SIZES=(26821627808 11520055616)
            MODEL_SHARD_SHA256S=(
                "06cb65f9cf6e510cd16500ca1719701a0ffa13dcaa4c4d0cc135e4f2071a8b4d"
                "86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e"
            )
            ;;
        5|qwen38-32gb-ablit|qwen-32gb-ablit|Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated)
            MODEL_PRESET="qwen38-32gb-ablit"
            MODEL_FAMILY="qwen38"
            IS_ABLITERATED=1
            MODEL_LABEL="Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated (176.5B (5B Active) · 24.98 GiB Shard 1 + Lazy PLE · v3.0-r2 Heretic-2 · 0.0818 KLD + blk.48 MTP)"
            MODEL_REPO="Synchopate/Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-GGUF"
            MODEL_REV="a3b3ddaf955e7f11f6d487cbe9bddf0454ce83b9"
            MODEL_SUBDIR="qwen38-maxq-32gb-ablit"
            MODEL_SHARDS=(
                "Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-00001-of-00002.gguf"
                "Qwen3.8-Flash-Next-MaxQ-32GB-Abliterated-00002-of-00002.gguf"
            )
            MODEL_SHARD_SIZES=(26821627808 11520055616)
            MODEL_SHARD_SHA256S=(
                "09b54c335505f9f609415dbc774868ad5ae188f17e873dead3ec743ec84e1009"
                "86b79714867743ddf5911ee7dff2717ae275f870551d86469cbdbf026c6b552e"
            )
            ;;
        6|nemotron-vq3|NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64)
            MODEL_PRESET="nemotron-vq3"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64 (30B · 13.30 GiB · v3.2.0 CPU-Turbo)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64-CPU-GGUF"
            MODEL_REV="d113f815ec2b24d9a1b2cb5f825a318ba37b183b"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-VQ3-64.gguf")
            MODEL_SHARD_SIZES=(14282559040)
            MODEL_SHARD_SHA256S=("04a460cdd831002d80b2f47ec5d70e7a92227abcf0a57f31ad73fe0bd50e711f")
            ;;
        7|nemotron-maxq-16gb|nemotron-16gb|NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB)
            MODEL_PRESET="nemotron-maxq-16gb"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB (30B · 13.39 GiB · 100% standard ggml)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB-GGUF"
            MODEL_REV="c858974ad057526f42165d789cfe9e04865738f8"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-MaxQ-16GB.gguf")
            MODEL_SHARD_SIZES=(14382336640)
            MODEL_SHARD_SHA256S=("26b884d70a52046a7198c4e01cac397af8bf44584b04cdcf59f896145587db67")
            ;;
        8|nemotron-ablit|nemotron-abliterated|NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64)
            MODEL_PRESET="nemotron-ablit"
            MODEL_FAMILY="nemotron"
            IS_ABLITERATED=1
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64 (30B · 13.30 GiB · v3.2.0)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64-CPU-GGUF"
            MODEL_REV="eb42aabff286f7d5b68ac81269b1d670544089cf"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-Abliterated-VQ3-64.gguf")
            MODEL_SHARD_SIZES=(14282559488)
            MODEL_SHARD_SHA256S=("75da8372f3896bfacd103f3cb85bf22b729d39cb1efce82fb3c838ab2af53e57")
            ;;
        # Legacy v2.5.4 DeepSeek-V4 presets used pre-v3.0 custom tensor type IDs (42/43); redirect cleanly to dsv4-maxq (100% standard ggml)
        dsv4-maxq-fractal|dsv4-vq3|dsv4-fractal|dsv4-ablit|dsv4-abliterated)
            LEGACY_DSV4_PRESET="$MODEL_PRESET"
            echo "[!] NOTICE: Legacy v2.5.4 preset '${LEGACY_DSV4_PRESET}' used retired custom tensor type IDs (42/43) that require llama-modchip v2.5.4."
            echo "    Redirecting cleanly to v3.2.0 '--preset dsv4-maxq' (DeepSeek-V4-Flash-0731-MaxQ-v4 · 57.99 GiB · 100% standard ggml)."
            if [[ "$LEGACY_DSV4_PRESET" == *ablit* ]]; then
                echo "    (Tip: For 64 GB opt-in uncensored v3.2.0 deployment, pass '--preset qwen38-64gb-ablit'.)"
            fi
            MODEL_PRESET="dsv4-maxq"
            MODEL_FAMILY="dsv4"
            MODEL_LABEL="DeepSeek-V4-Flash-0731-MaxQ-v4 (284B · 57.99 GiB · 3 shards)"
            MODEL_REPO="Synchopate/DeepSeek-V4-Flash-0731-MaxQ-v4-64GB-CPU-GGUF"
            MODEL_REV="48356a120e78c01e82f425a74380f46f3925fb9c"
            MODEL_SUBDIR="dsv4-maxq-v4"
            MODEL_SHARDS=(
                "DeepSeek-V4-Flash-0731-MaxQ-v4-00001-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-MaxQ-v4-00002-of-00003.gguf"
                "DeepSeek-V4-Flash-0731-MaxQ-v4-00003-of-00003.gguf"
            )
            MODEL_SHARD_SIZES=(5258336 35543586816 26721192096)
            MODEL_SHARD_SHA256S=(
                "6112e887b21d1fdf7b7196232b528698a6ec2d5a59760a040cb4231f8e8d691b"
                "5d85c740992b70395da5c3e4a856b3fad6ef725271722c8d837b35befe28330d"
                "a8654d4d5fedf748d6f2b6404a8b7f9e19435a8aa8e2bb97ce9a66bc012c0f3a"
            )
            ;;
        nemotron-fractal)
            MODEL_PRESET="nemotron-fractal"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64 (30B · 14.36 GiB · v3.2.0)"
            MODEL_REPO="Synchopate/NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64-CPU-GGUF"
            MODEL_REV="d1336d5828ed475d4bd6e6604cadf52250a4c818"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-30B-A3B-Fractal-64.gguf")
            MODEL_SHARD_SIZES=(15420034624)
            MODEL_SHARD_SHA256S=("3a3a09eac29b5906397ca6a93d2ff0f566fca0bdd365294d067aeb5d0ebf4a1b")
            ;;
        nemotron-q4)
            MODEL_PRESET="nemotron-q4"
            MODEL_FAMILY="nemotron"
            MODEL_LABEL="NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0 (30B · 17.60 GiB)"
            MODEL_REPO="ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"
            MODEL_REV="8cd082170ed313484be1cf44f256a5caa2cfdad3"
            MODEL_SHARDS=("NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf")
            MODEL_SHARD_SIZES=(18898091584)
            MODEL_SHARD_SHA256S=("61f87e75974e4b535dcdf9aad056541a9514f1dfa4538b463b081d19b7a00e3c")
            ;;
        *)
            echo "[!] ERROR: Unknown --preset '${MODEL_PRESET}'." >&2
            echo "    Valid v3.2.0 presets: qwen38-64gb, qwen38-64gb-ablit, dsv4-maxq, qwen38-32gb, qwen38-32gb-ablit, nemotron-vq3, nemotron-maxq-16gb, nemotron-ablit" >&2
            exit 1
            ;;
    esac
    DEFAULT_MODEL_NAME="${MODEL_SHARDS[0]}"
    if [[ "$MODEL_FAMILY" == "qwen38" && "$WITH_MMPROJ" == "1" ]]; then
        MODEL_SHARDS+=("mmproj-Qwen3.8-Flash-Next-BF16.gguf")
        MODEL_SHARD_SIZES+=(907542880)
        MODEL_SHARD_SHA256S+=("ef2af1d6aa8a22740c7252bc7ce40c5c2f1249aa8d9c8499a4d4572c3cfdf9df")
    fi
fi

if [[ "$MODEL_PRESET" == "qwen38-64gb"* && "$TOTAL_RAM_GB" -lt 58 ]]; then
    echo "[!] WARNING: Qwen3.8-MaxQ-64GB requires ~45.1 GiB VmRSS (64 GB+ host RAM recommended; use --preset qwen38-32gb on 32 GB hosts)."
elif [[ "$MODEL_PRESET" == "qwen38-32gb"* && "$TOTAL_RAM_GB" -lt 28 ]]; then
    echo "[!] WARNING: Qwen3.8-MaxQ-32GB requires ~27.44 GiB peak VmHWM + NVMe mmap PLE (32 GB+ host RAM recommended; use --preset nemotron-maxq-16gb on 16 GB hosts)."
elif [[ "$MODEL_FAMILY" == "dsv4" && "$TOTAL_RAM_GB" -lt 60 ]]; then
    echo "[!] WARNING: DeepSeek-V4 284B requires ~58.3–59.1 GiB anonymous-heap VmRSS (64 GB+ host RAM required)."
fi

if [[ "$IS_ABLITERATED" == "1" ]]; then
    ALIGNED_ALT="qwen38-64gb"
    [[ "$MODEL_PRESET" == *32gb* ]] && ALIGNED_ALT="qwen38-32gb"
    [[ "$MODEL_FAMILY" == "dsv4" ]] && ALIGNED_ALT="dsv4-maxq"
    [[ "$MODEL_FAMILY" == "nemotron" ]] && ALIGNED_ALT="nemotron-vq3"
    echo ""
    echo "=============================================================================="
    echo " [!] DISCLAIMER — OPT-IN ABLITERATED / UNCENSORED MODEL CHECKPOINT"
    echo "=============================================================================="
    echo " Selected model (${MODEL_LABEL}) has refusal-mediating activation directions"
    echo " attenuated via orthogonal subspace projection. Upstream safety refusal"
    echo " filters cannot be relied upon for content moderation."
    echo "   • Scope: Authorized security research, reverse engineering, red-teaming,"
    echo "     CTF, code auditing, and uncensored legal/policy analysis only."
    echo "   • Deployment: Do NOT expose this model to unmoderated public networks"
    echo "     without external access control and safety filtering."
    echo "   • Responsibility: You assume full legal and operational responsibility for"
    echo "     all generated outputs. For standard safety alignment, use"
    echo "     './install.sh --preset ${ALIGNED_ALT}'."
    echo "=============================================================================="
    if [[ "$ACCEPT_ABLITERATED" != "1" ]]; then
        if [[ "$IS_INTERACTIVE" == "1" ]]; then
            printf " Proceed with uncensored/abliterated checkpoint? [y/N]: "
            read -r ABLIT_CONFIRM < /dev/tty || ABLIT_CONFIRM=""
            if [[ ! "$ABLIT_CONFIRM" =~ ^[Yy]([Ee][Ss])?$ ]]; then
                echo "[i] Aborted. Re-run './install.sh --preset ${ALIGNED_ALT}' for the standard safety-aligned model."
                exit 1
            fi
        else
            echo "[!] ERROR: Abliterated/uncensored presets require explicit consent." >&2
            echo "    Pass '--accept-abliterated' when running with --yes / non-interactive mode," >&2
            echo "    or use the safety-aligned preset '--preset ${ALIGNED_ALT}'." >&2
            exit 1
        fi
    fi
fi

# Unified v3.2.0 Engine Installation Mode (Prebuilt binary vs Local Source Build)
if [[ "$ENGINE_MODE_EXPLICIT" == "0" ]]; then
    if [[ -n "$X86_PREBUILT_ASSET" ]]; then
        if [[ "$IS_INTERACTIVE" == "1" ]]; then
            echo ""
            echo "------------------------------------------------------------------------------"
            echo " ENGINE INSTALLATION MODE (Linux x86_64 — CPUID matched: ${X86_PREBUILT_ASSET})"
            echo "------------------------------------------------------------------------------"
            echo "  1) Download precompiled v3.2.0 binary (${X86_PREBUILT_ASSET}) [Recommended, ~5s]"
            echo "     Unified RouteCrack 2.0 binary serving Qwen3.8, DeepSeek-V4 284B & Nemotron-30B"
            echo "     (static libgomp/libstdc++, GLIBC <= 2.30; zero C++ compiler required)."
            echo "  2) Compile unified v3.2.0 engine from source"
            echo "     Applies patches/llama-modchip-v3.2.patch onto pinned llama.cpp."
            echo "------------------------------------------------------------------------------"
            printf " Select engine mode [1-2, Enter=1]: "
            read -r USER_ENGINE_CHOICE < /dev/tty || USER_ENGINE_CHOICE=""
            USER_ENGINE_CHOICE="${USER_ENGINE_CHOICE:-1}"
            case "$USER_ENGINE_CHOICE" in
                1) USE_PREBUILT=1 ;;
                2) USE_PREBUILT=0 ;;
                *)
                    echo "[!] Unrecognized choice '${USER_ENGINE_CHOICE}'; using precompiled binary."
                    USE_PREBUILT=1
                    ;;
            esac
        else
            USE_PREBUILT=1
        fi
    else
        USE_PREBUILT=0
    fi
fi

if [[ "$USE_PREBUILT" == "0" ]]; then
    check_required_commands git cmake g++
fi

if [[ -f /sys/kernel/mm/transparent_hugepage/enabled ]]; then
    THP_STATE="$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    if [[ "$THP_STATE" == *"[madvise]"* ]] || [[ "$THP_STATE" == *"[always]"* ]]; then
        echo "[+] 2MB Transparent HugePages (THP) active: ${THP_STATE}"
    else
        echo "[i] Tip: Enable madvise Transparent HugePages for lower TLB overhead:"
        echo "    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled"
        echo "    echo defer+madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag"
    fi
fi

echo ""
echo "=============================================================================="
echo " STEP 2/5: INSTALLING UNIFIED LLAMA-MODCHIP v3.2.0 ENGINE"
echo "=============================================================================="
LLAMA_BUILD_DIR="${WORKSPACE_DIR}/llama.cpp"

PREBUILT_OK=0
if [[ "$USE_PREBUILT" == "1" && -n "$X86_PREBUILT_ASSET" ]]; then
    ASSET_URL="https://github.com/synchopate/llama-modchip/releases/download/v3.2.0/${X86_PREBUILT_ASSET}"
    FALLBACK_ASSET_URL="https://huggingface.co/Synchopate/Qwen3.8-Flash-Next-MaxQ-64GB-GGUF/resolve/main/${X86_PREBUILT_ASSET}"
    EXPECTED_BIN_SHA256="d96745036cb44f76bce70237fc1e474058958e4d2eff739b85415eac4c78131d"
    [[ "$X86_PREBUILT_ASSET" == *-avx512 ]] && EXPECTED_BIN_SHA256="0cf2d3bcd31349d98bcef1d23af3545912b70930049437e8aa7f6cf5e2ff2082"
    echo "[*] CPUID selected portable v3.2.0 precompiled binary: ${X86_PREBUILT_ASSET}"
    if curl -fL --progress-bar -o "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "$ASSET_URL" \
        || curl -fL --progress-bar -o "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "$FALLBACK_ASSET_URL"; then
        ACTUAL_BIN_SHA256="$(compute_sha256 "${WORKSPACE_DIR}/bin/llama-server-nemotron.part")"
        chmod +x "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
        if [[ "$ACTUAL_BIN_SHA256" != "$EXPECTED_BIN_SHA256" ]]; then
            rm -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
            echo "[!] SHA-256 mismatch on prebuilt binary (expected ${EXPECTED_BIN_SHA256}, got ${ACTUAL_BIN_SHA256}); falling back to local source build..."
            check_required_commands git cmake g++
        elif ! "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" --version >/dev/null 2>&1; then
            rm -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
            echo "[!] Prebuilt binary failed '--version' host compatibility smoke-test; falling back to local source build..."
            check_required_commands git cmake g++
        else
            mv -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
            ln -sf "llama-server-nemotron" "${WORKSPACE_DIR}/bin/${X86_PREBUILT_ASSET}"
            ln -sf "llama-server-nemotron" "${WORKSPACE_DIR}/bin/llama-server-modchip-linux-x86_64"
            # Purge any stale pre-v3.0 RouteCrack helpers and old v2.x shared libs (libggml/libllama)
            rm -f "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" "${WORKSPACE_DIR}/bin/rc_build_tables"
            rm -f "${LLAMA_BUILD_DIR}/build/bin"/libggml* "${LLAMA_BUILD_DIR}/build/bin"/libllama* "${LLAMA_BUILD_DIR}/build/bin"/libmtmd* 2>/dev/null || true
            echo "[+] Installed & verified prebuilt v3.2.0 binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron (${X86_PREBUILT_ASSET}, SHA-256 + --version OK)"
            PREBUILT_OK=1
        fi
    else
        rm -f "${WORKSPACE_DIR}/bin/llama-server-nemotron.part"
        echo "[!] Prebuilt download failed; checking build toolchain for local source fallback..."
        check_required_commands git cmake g++
    fi
elif [[ "$USE_PREBUILT" == "1" ]]; then
    echo "[i] No precompiled Linux x86_64 AVX2/AVX-512 binary matches $(uname -s)/$(uname -m); compiling natively from source..."
    check_required_commands git cmake g++
fi

if [[ "$PREBUILT_OK" == "0" ]]; then
    bash "${REPO_ROOT}/scripts/build_zen5_llama.sh" "$LLAMA_BUILD_DIR"

    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-server" "${WORKSPACE_DIR}/bin/llama-server-nemotron"
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-cli" "${WORKSPACE_DIR}/bin/llama-cli" 2>/dev/null || true
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-quantize" "${WORKSPACE_DIR}/bin/llama-quantize" 2>/dev/null || true
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-perplexity" "${WORKSPACE_DIR}/bin/llama-perplexity" 2>/dev/null || true
    cp -f "${LLAMA_BUILD_DIR}/build/bin/llama-bench" "${WORKSPACE_DIR}/bin/llama-bench" 2>/dev/null || true
    echo "[+] Installed unified v3.2.0 binary: ${WORKSPACE_DIR}/bin/llama-server-nemotron"
fi

if [[ "$PREBUILT_OK" == "0" && "$MODEL_FAMILY" == "nemotron" && -d "${LLAMA_BUILD_DIR}/include" ]]; then
    echo "[*] Compiling optional Nemotron RouteCrack vocabulary tokenizer & table builder..."
    rm -f "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" "${WORKSPACE_DIR}/bin/rc_build_tables"
    g++ -O3 -std=c++17 "${REPO_ROOT}/speculative/cpp/rc_tokenize_nemotron.cpp" \
        -o "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" \
        -I"${LLAMA_BUILD_DIR}/include" -I"${LLAMA_BUILD_DIR}/ggml/include" \
        -L"${LLAMA_BUILD_DIR}/build/bin" -lllama -lggml -lggml-base \
        -Wl,-rpath,"${LLAMA_BUILD_DIR}/build/bin" || true

    g++ -O3 -std=c++17 "${REPO_ROOT}/speculative/cpp/rc_build_tables.cpp" \
        -o "${WORKSPACE_DIR}/bin/rc_build_tables" || true
fi

echo ""
echo "=============================================================================="
echo " STEP 3/5: RESOLVING / DOWNLOADING MODEL (${MODEL_LABEL})"
echo "=============================================================================="
if [[ -z "$MODEL_PATH" ]]; then
    if [[ -n "$MODEL_SUBDIR" ]]; then
        MODEL_DEST_DIR="${WORKSPACE_DIR}/models/${MODEL_SUBDIR}"
    else
        MODEL_DEST_DIR="${WORKSPACE_DIR}/models"
    fi
    if [[ -f "${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_PATH="${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}"
    elif [[ -f "${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}" ]]; then
        MODEL_DEST_DIR="${WORKSPACE_DIR}/models"
        MODEL_PATH="${WORKSPACE_DIR}/models/${DEFAULT_MODEL_NAME}"
    else
        MODEL_PATH="${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}"
    fi
fi

if [[ "$MODEL_EXPLICIT" == "1" ]]; then
    if [[ -f "$MODEL_PATH" ]]; then
        echo "[+] Using explicit local GGUF model: $MODEL_PATH"
    else
        echo "[!] ERROR: Explicit --model file not found: $MODEL_PATH" >&2
        exit 1
    fi
elif [[ "$SKIP_MODEL" == "1" ]]; then
    echo "[i] --skip-model specified. Skipping model weight download."
    echo "    Expected default GGUF path: $MODEL_PATH"
else
    file_stat_key() {
        python3 - "$1" <<'PYEOF'
import os, sys
st = os.stat(sys.argv[1])
print(f"{st.st_size} {st.st_mtime_ns}")
PYEOF
    }

    mkdir -p "$MODEL_DEST_DIR"
    NUM_SHARDS="${#MODEL_SHARDS[@]}"
    for idx in "${!MODEL_SHARDS[@]}"; do
        SHARD_NAME="${MODEL_SHARDS[$idx]}"
        SHARD_DEST="${MODEL_DEST_DIR}/${SHARD_NAME}"
        SHARD_CACHE="${SHARD_DEST}.sha256.cache"
        SHARD_NUM=$((idx + 1))
        if [[ -s "$SHARD_DEST" ]]; then
            if [[ ${#MODEL_SHARD_SIZES[@]} -gt "$idx" && -n "${MODEL_SHARD_SIZES[$idx]}" ]]; then
                EXISTING_BYTES="$(wc -c < "$SHARD_DEST" | tr -d '[:space:]')"
                if [[ "$EXISTING_BYTES" != "${MODEL_SHARD_SIZES[$idx]}" ]]; then
                    echo "[!] Existing file ${SHARD_DEST} has unexpected byte size (${EXISTING_BYTES} != ${MODEL_SHARD_SIZES[$idx]}); re-downloading..."
                    rm -f "$SHARD_DEST" "$SHARD_CACHE"
                elif [[ ${#MODEL_SHARD_SHA256S[@]} -gt "$idx" && -n "${MODEL_SHARD_SHA256S[$idx]}" ]]; then
                    EXPECTED_SHA256="${MODEL_SHARD_SHA256S[$idx]}"
                    CUR_KEY="$(file_stat_key "$SHARD_DEST" 2>/dev/null || true)"
                    CACHED_LINE=""
                    [[ -f "$SHARD_CACHE" ]] && CACHED_LINE="$(head -n 1 "$SHARD_CACHE" 2>/dev/null || true)"
                    if [[ -n "$CUR_KEY" && "$CACHED_LINE" == "${CUR_KEY} ${EXPECTED_SHA256}" ]]; then
                        echo "[+] Found verified existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}, cached SHA-256): ${SHARD_DEST}"
                        continue
                    fi
                    echo "[*] Verifying SHA-256 of existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_NAME}..."
                    EXISTING_SHA256="$(compute_sha256 "$SHARD_DEST")"
                    if [[ "$EXISTING_SHA256" != "$EXPECTED_SHA256" ]]; then
                        echo "[!] Existing ${SHARD_DEST} failed SHA-256 verification (stale release or corrupted shard; expected ${EXPECTED_SHA256:0:12}..., got ${EXISTING_SHA256:0:12}...). Re-downloading..."
                        rm -f "$SHARD_DEST" "$SHARD_CACHE"
                    else
                        [[ -n "$CUR_KEY" ]] && printf '%s %s\n' "$CUR_KEY" "$EXPECTED_SHA256" > "$SHARD_CACHE" 2>/dev/null || true
                        echo "[+] Found verified existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
                        continue
                    fi
                else
                    echo "[+] Found existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
                    continue
                fi
            else
                echo "[+] Found existing GGUF file (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
                continue
            fi
        fi
        SHARD_URL="https://huggingface.co/${MODEL_REPO}/resolve/${MODEL_REV}/${SHARD_NAME}"
        echo "[*] Downloading (${SHARD_NUM}/${NUM_SHARDS}) ${SHARD_NAME} from ${MODEL_REPO}@${MODEL_REV:0:12}..."
        echo "    URL:  ${SHARD_URL}"
        echo "    Dest: ${SHARD_DEST}"
        curl -fL -C - --progress-bar -o "${SHARD_DEST}.part" "$SHARD_URL"

        MAGIC="$(head -c 4 "${SHARD_DEST}.part" 2>/dev/null || true)"
        if [[ "$MAGIC" != "GGUF" ]]; then
            echo "[!] ERROR: Downloaded file ${SHARD_DEST}.part does not have a valid GGUF header (got '${MAGIC}')." >&2
            rm -f "${SHARD_DEST}.part"
            exit 1
        fi

        if [[ ${#MODEL_SHARD_SIZES[@]} -gt "$idx" && -n "${MODEL_SHARD_SIZES[$idx]}" ]]; then
            EXPECTED_BYTES="${MODEL_SHARD_SIZES[$idx]}"
            ACTUAL_BYTES="$(wc -c < "${SHARD_DEST}.part" | tr -d '[:space:]')"
            if [[ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ]]; then
                echo "[!] ERROR: Size mismatch for ${SHARD_NAME} (expected ${EXPECTED_BYTES} bytes, got ${ACTUAL_BYTES})!" >&2
                rm -f "${SHARD_DEST}.part"
                exit 1
            fi
        fi

        if [[ ${#MODEL_SHARD_SHA256S[@]} -gt "$idx" && -n "${MODEL_SHARD_SHA256S[$idx]}" ]]; then
            EXPECTED_SHA256="${MODEL_SHARD_SHA256S[$idx]}"
            echo "[*] Verifying SHA-256 checksum for ${SHARD_NAME}..."
            ACTUAL_SHA256="$(compute_sha256 "${SHARD_DEST}.part")"
            if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
                echo "[!] ERROR: SHA-256 mismatch for ${SHARD_DEST}.part (removing corrupt partial file)!" >&2
                echo "    Expected: $EXPECTED_SHA256" >&2
                echo "    Got:      $ACTUAL_SHA256" >&2
                rm -f "${SHARD_DEST}.part"
                exit 1
            fi
        fi

        mv "${SHARD_DEST}.part" "$SHARD_DEST"
        if [[ ${#MODEL_SHARD_SHA256S[@]} -gt "$idx" && -n "${MODEL_SHARD_SHA256S[$idx]}" ]]; then
            NEW_KEY="$(file_stat_key "$SHARD_DEST" 2>/dev/null || true)"
            [[ -n "$NEW_KEY" ]] && printf '%s %s\n' "$NEW_KEY" "${MODEL_SHARD_SHA256S[$idx]}" > "$SHARD_CACHE" 2>/dev/null || true
        fi
        echo "[+] Verified (${SHARD_NUM}/${NUM_SHARDS}): ${SHARD_DEST}"
    done
    MODEL_PATH="${MODEL_DEST_DIR}/${DEFAULT_MODEL_NAME}"
fi

echo ""
echo "=============================================================================="
echo " STEP 4/5: SPECULATIVE DECODING ENGINE CONFIGURATION"
echo "=============================================================================="
if [[ "$MODEL_FAMILY" == "nemotron" && -f "$MODEL_PATH" && -x "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" && -x "${WORKSPACE_DIR}/bin/rc_build_tables" ]]; then
    CORPUS_TXT="${WORKSPACE_DIR}/routecrack/bootstrap_corpus.txt"
    CORPUS_BIN="${WORKSPACE_DIR}/routecrack/corpus_nemotron_train.bin"
    echo "[*] Assembling coding & agentic <tool_call> training corpus..."
    {
        for _ in 1 2 3 4 5; do
            find "$REPO_ROOT" -maxdepth 3 -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" -o -name "*.md" -o -name "*.sh" -o -name "*.jinja" \) \
                -not -path "*/.git/*" -not -path "${REPO_ROOT}/workspace/*" -not -path "${WORKSPACE_DIR}/*" -exec cat {} + 2>/dev/null || true
        done
        if [[ -n "$USER_CORPUS_DIR" && -d "$USER_CORPUS_DIR" ]]; then
            echo "[*] Including custom codebase from $USER_CORPUS_DIR..." >&2
            find "$USER_CORPUS_DIR" -maxdepth 4 -type f \( -name "*.py" -o -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.rs" -o -name "*.go" -o -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.md" \) \
                -not -path "*/.git/*" -not -path "*/node_modules/*" -exec head -c 100000 {} + 2>/dev/null || true
        fi
    } > "$CORPUS_TXT"

    echo "[*] Tokenizing corpus with native Nemotron 131K vocabulary..."
    if "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" "$MODEL_PATH" "$CORPUS_TXT" "$CORPUS_BIN" \
        && cp -f "$CORPUS_BIN" "${WORKSPACE_DIR}/routecrack/corpus_nemotron_warm.bin" \
        && echo "[*] Building 8-level FNV-64 Hash Cascade, 300 Hot Rules & Macro Codebook..." \
        && "${WORKSPACE_DIR}/bin/rc_build_tables" "$CORPUS_BIN" "${WORKSPACE_DIR}/routecrack"; then
        echo "[+] RouteCrack tables ready in ${WORKSPACE_DIR}/routecrack/"
    else
        rm -f "${WORKSPACE_DIR}/bin/rc_tokenize_nemotron" "${WORKSPACE_DIR}/bin/rc_build_tables"
        echo "[!] Optional offline RouteCrack bootstrap skipped (stale pre-v3.0 helper binary removed)."
        echo "[+] Online Stockfish-PVS v2 (192 KB L2-resident Zobrist TT) active — zero offline vocabulary build required."
    fi
elif [[ "$MODEL_FAMILY" == "qwen38" ]]; then
    echo "[+] RouteCrack 2.0 (--spec-type routecrack2: NextN MTP + Two-Zone Suffix + Stockfish-PVS + L1 Micro-Arbiter) ready (RC_TABLES_DIR=/nonexistent isolated)."
elif [[ "$MODEL_FAMILY" == "dsv4" ]]; then
    echo "[+] RouteCrack 2.0 (--spec-type routecrack2: Heavy-MoE L1 Micro-Arbiter + Two-Zone Suffix + Stockfish-PVS) ready (RC_TABLES_DIR=/nonexistent isolated)."
else
    echo "[+] RouteCrack 2.0 (--spec-type routecrack2: RCPredictor + Two-Zone Suffix + Stockfish-PVS + L1 Micro-Arbiter) ready."
fi

echo ""
echo "=============================================================================="
echo " STEP 5/5: GENERATING AUTO-TUNED v3.2.0 LAUNCH SCRIPTS"
echo "=============================================================================="
if [[ "$MODEL_FAMILY" == "qwen38" ]]; then
    CTX_SIZE=32768
elif [[ "$TOTAL_RAM_GB" -ge 30 ]]; then
    CTX_SIZE=262144
else
    CTX_SIZE=32768
fi

mkdir -p "${WORKSPACE_DIR}/run"
printf '%s\n' "${WORKSPACE_DIR}" > "${REPO_ROOT}/.modchip_workspace"
printf '%s\n' "${SERVER_PORT}" > "${WORKSPACE_DIR}/run/server_port"
if [[ -n "$MODEL_PATH" ]]; then
    case "$MODEL_PATH" in
        /*) MODEL_ABS="$MODEL_PATH" ;;
        *)  MODEL_ABS="${PWD}/${MODEL_PATH}" ;;
    esac
    if [[ -f "$MODEL_ABS" || "$MODEL_EXPLICIT" == "1" ]]; then
        printf '%s\n' "${MODEL_ABS}" > "${WORKSPACE_DIR}/run/default_model"
    fi
fi

if [[ -f "${REPO_ROOT}/start_server.sh" && ! -f "${REPO_ROOT}/start_server.sh.bak" ]]; then
    cp -f "${REPO_ROOT}/start_server.sh" "${REPO_ROOT}/start_server.sh.bak"
    echo "[i] Backed up original start_server.sh -> start_server.sh.bak"
fi

cat > "${REPO_ROOT}/start_server.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh (llama-modchip v3.2.0)
# Usage: ./start_server.sh [/path/to/model.gguf] [extra llama-server flags]
set -euo pipefail

export WORKSPACE_DIR="\${WORKSPACE_DIR:-${WORKSPACE_DIR}}"
if [[ "${PREBUILT_OK}" == "0" ]]; then
    export LD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/build/bin:\${LD_LIBRARY_PATH:-}"
    export DYLD_LIBRARY_PATH="${LLAMA_BUILD_DIR}/build/bin:\${DYLD_LIBRARY_PATH:-}"
fi

MODEL="${MODEL_PATH}"
PASS_ARGS=()
if [[ \$# -gt 0 && "\$1" != -* ]]; then
    MODEL="\$1"
    shift
fi
while [[ \$# -gt 0 ]]; do
    case "\$1" in
        -m|--model)
            if [[ \$# -ge 2 ]]; then
                MODEL="\$2"
                shift 2
                continue
            fi
            ;;
        --model=*)
            MODEL="\${1#--model=}"
            shift
            continue
            ;;
    esac
    PASS_ARGS+=("\$1")
    shift
done
if [[ ! -f "\$MODEL" ]]; then
    echo "[!] Model file not found at: \$MODEL"
    echo "    Run ./install.sh (without --skip-model) or pass /path/to/model.gguf [extra flags]"
    exit 1
fi

# Inspect GGUF header and write per-UID SHM profile
python3 "${REPO_ROOT}/scripts/llama_modchip.py" --model "\$MODEL" --apply-shm

PROFILE_ENV="\$(python3 - "\$MODEL" <<'PYEOF'
import sys
sys.path.insert(0, "${REPO_ROOT}/scripts")
import llama_modchip as mc
hw = mc.probe_hardware()
info = mc.inspect_gguf(sys.argv[1])
prof = mc.compute_modchip_profile(hw, info)
n_cores = max(1, len(hw.physical_core_cpus))
print(f"export MODEL_ARCH=\"{info.arch}\"")
print(f"export MODEL_HAS_MTP=\"{1 if info.has_mtp else 0}\"")
print(f"export IS_QWEN38_32GB=\"{1 if info.is_qwen38_32gb else 0}\"")
print(f"export USE_REPACK=\"{1 if prof.use_repack else 0}\"")
print(f"export N_CORES=\"{n_cores}\"")
print(f"export LLAMA_KV_L3_WINDOW=\${{LLAMA_KV_L3_WINDOW:-{prof.l3_window}}}")
print(f"export LLAMA_ATTN_WINDOW=\${{LLAMA_ATTN_WINDOW:-{prof.attn_window}}}")
print(f"export LLAMA_ATTN_SINKS=\${{LLAMA_ATTN_SINKS:-{prof.attn_sinks}}}")
print(f"export LLAMA_SWA_STRIDE=\${{LLAMA_SWA_STRIDE:-{prof.swa_stride}}}")
print(f"export LLAMA_KV_DISABLE_CHUNKING=\${{LLAMA_KV_DISABLE_CHUNKING:-{prof.disable_chunking}}}")
print(f"export LLAMA_RC_ENABLED=\${{LLAMA_RC_ENABLED:-{prof.rc_enabled}}}")
print(f"export LLAMA_RC_CERT_D0=\${{LLAMA_RC_CERT_D0:-{prof.rc_cert_d0}}}")
print(f"export LLAMA_RC_CERT_D1=\${{LLAMA_RC_CERT_D1:-{prof.rc_cert_d1}}}")
print(f"export LLAMA_RC_CERT_D2=\${{LLAMA_RC_CERT_D2:-{prof.rc_cert_d2}}}")
if prof.moe_top_k > 0 and info.arch != "deepseek4":
    print(f"export LLAMA_MOE_TOP_K=\${{LLAMA_MOE_TOP_K:-{prof.moe_top_k}}}")
print(f"export OMP_PROC_BIND=\${{OMP_PROC_BIND:-{hw.omp_proc_bind}}}")
print(f"export OMP_PLACES=\${{OMP_PLACES:-cores}}")
print(f'export TASKSET_MASK="\${{TASKSET_MASK:-{hw.taskset_mask}}}"')
print(f"export DECODE_THREADS=\${{DECODE_THREADS:-{prof.hw.decode_threads}}}")
print(f"export BATCH_THREADS=\${{BATCH_THREADS:-{prof.hw.batch_threads}}}")
PYEOF
)" || exit 1
eval "\$PROFILE_ENV"

# Mandatory RouteCrack vocabulary isolation: NEVER load Nemotron 131K tables when serving Qwen3.8 or DeepSeek-V4!
if [[ "\${MODEL_ARCH:-}" == "nemotron_h_moe" ]]; then
    export RC_TABLES_DIR="\${RC_TABLES_DIR:-${WORKSPACE_DIR}/routecrack}"
else
    export RC_TABLES_DIR="/nonexistent"
fi

# Non-convoy thread fallback if DECODE_THREADS was not set above: \$((N_CORES - 1)) clamped to >= 1
if [[ -z "\${DECODE_THREADS:-}" ]]; then
    if [[ "\${N_CORES:-1}" -gt 1 ]]; then
        DECODE_THREADS=\$((N_CORES - 1))
    else
        DECODE_THREADS=1
    fi
fi
BATCH_THREADS="\${BATCH_THREADS:-\${DECODE_THREADS}}"

CACHE_V_TYPE="q4_0"
FLASH_ATTN_MODE="auto"
EFFECTIVE_CTX_SIZE="\${MODCHIP_CTX_SIZE:-${CTX_SIZE}}"
UBATCH_SIZE="\${MODCHIP_UBATCH_SIZE:-2048}"
SLOT_SAVE_DIR="\${MODCHIP_SLOT_SAVE_PATH:-/dev/shm}"
DEFAULT_LOAD_MODE="none"
[[ "\$(uname -s)" == "Darwin" ]] && DEFAULT_LOAD_MODE="mmap"

# Isolate DSV4_MOE_CFG_PATH inside a 0700 per-user directory and reset to unpruned Top-6 on startup
MODCHIP_PRIV_RUN="\${XDG_RUNTIME_DIR:-${WORKSPACE_DIR}/run}/modchip-\$(id -u)"
mkdir -p "\$MODCHIP_PRIV_RUN"
chmod 700 "\$MODCHIP_PRIV_RUN" 2>/dev/null || true
export DSV4_MOE_CFG_PATH="\${DSV4_MOE_CFG_PATH:-\${MODCHIP_PRIV_RUN}/dsv4_moe_cfg.bin}"
if [[ "\${MODCHIP_KEEP_DSV4_CFG:-0}" != "1" ]]; then
    python3 - "\$DSV4_MOE_CFG_PATH" <<'CFGEOF'
import os, struct, sys
p = sys.argv[1]
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
try:
    os.write(fd, struct.pack("<8i", 6, 6, 0, 3, 43, 1, 0, 0))
finally:
    os.close(fd)
CFGEOF
fi

MODEL_EXTRA_ARGS=()
SPEC_ARGS=()
CKPT_ARGS=(--ctx-checkpoints 64 --checkpoint-min-step 128)

if [[ "\${MODEL_ARCH:-}" == "qwen4exp" ]]; then
    CACHE_V_TYPE="q8_0"
    FLASH_ATTN_MODE="on"
    DEFAULT_LOAD_MODE="mmap"
    [[ -z "\${MODCHIP_CTX_SIZE:-}" ]] && EFFECTIVE_CTX_SIZE=32768
    if [[ "${TOTAL_RAM_GB}" -lt 48 || "\${IS_QWEN38_32GB:-0}" == "1" ]]; then
        [[ -z "\${MODCHIP_UBATCH_SIZE:-}" ]] && UBATCH_SIZE=512
        CKPT_ARGS=(--ctx-checkpoints "\${MODCHIP_CTX_CHECKPOINTS:-0}")
    else
        CKPT_ARGS=(--ctx-checkpoints "\${MODCHIP_CTX_CHECKPOINTS:-1}")
    fi
    MODEL_EXTRA_ARGS+=(--jinja)
    MMPROJ_CAND="\$(dirname "\$MODEL")/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
    if [[ -f "\$MMPROJ_CAND" ]]; then
        HAS_MMPROJ_FLAG=0
        for _arg in \${PASS_ARGS[@]+"\${PASS_ARGS[@]}"} \${EXTRA_SERVER_ARGS:-}; do
            case "\$_arg" in
                --mmproj|--mmproj=*|--no-mmproj) HAS_MMPROJ_FLAG=1; break ;;
            esac
        done
        [[ "\$HAS_MMPROJ_FLAG" == "0" ]] && MODEL_EXTRA_ARGS+=(--mmproj "\$MMPROJ_CAND")
    fi
    if [[ "\${IS_QWEN38_32GB:-0}" == "1" ]]; then
        MODEL_EXTRA_ARGS+=(--temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.2 --repeat-penalty 1.05)
    fi
elif [[ "\${MODEL_ARCH:-}" == "deepseek4" ]]; then
    CACHE_V_TYPE="q8_0"
    if [[ "${TOTAL_RAM_GB}" -lt 80 ]]; then
        AVAIL_KIB="\$(awk '/MemAvailable:/ {print \$2}' /proc/meminfo 2>/dev/null || echo 62914560)"
        if [[ "\${AVAIL_KIB:-62914560}" -lt 67108864 || "${TOTAL_RAM_GB}" -le 68 ]]; then
            [[ -z "\${MODCHIP_CTX_SIZE:-}" ]] && EFFECTIVE_CTX_SIZE=32768
            [[ -z "\${MODCHIP_UBATCH_SIZE:-}" ]] && UBATCH_SIZE=512
            [[ -z "\${MODCHIP_SLOT_SAVE_PATH:-}" ]] && SLOT_SAVE_DIR="${WORKSPACE_DIR}/cache"
            echo "[i] 64GB OS Headroom Guard: using -c \${EFFECTIVE_CTX_SIZE} -ub \${UBATCH_SIZE} and disk slot cache"
            echo "    so desktop/OS processes retain >=3.4–4.0 GiB free RAM. (Set MODCHIP_CTX_SIZE=262144 on a dedicated headless node.)"
        fi
    fi
else
    if [[ "${TOTAL_RAM_GB}" -le 18 ]]; then
        [[ -z "\${MODCHIP_CTX_SIZE:-}" ]] && EFFECTIVE_CTX_SIZE=32768
        [[ -z "\${MODCHIP_UBATCH_SIZE:-}" ]] && UBATCH_SIZE=512
        [[ -z "\${MODCHIP_SLOT_SAVE_PATH:-}" ]] && SLOT_SAVE_DIR="${WORKSPACE_DIR}/cache"
        echo "[i] 16GB RAM Swap Guard: defaulting to -c \${EFFECTIVE_CTX_SIZE} -ub \${UBATCH_SIZE} (~14.6 GiB VmRSS) to prevent swap thrashing."
    fi
fi

# Unified RouteCrack 2.0 speculative engine across all 3 flagship families (honors LLAMA_RC_ENABLED=0 / LLAMA_RC_DISABLE=1 and MODCHIP_SPEC_TYPE override)
if [[ "\${LLAMA_RC_ENABLED:-1}" == "0" || "\${LLAMA_RC_DISABLE:-0}" == "1" ]]; then
    SPEC_ARGS=(--spec-type none)
elif [[ -n "\${MODCHIP_SPEC_TYPE:-}" ]]; then
    case "\${MODCHIP_SPEC_TYPE}" in
        none)
            SPEC_ARGS=(--spec-type none)
            ;;
        ngram-simple)
            SPEC_ARGS=(--spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 7 --spec-ngram-simple-min-hits 1)
            ;;
        *)
            SPEC_ARGS=(--spec-type "\${MODCHIP_SPEC_TYPE}")
            ;;
    esac
else
    SPEC_ARGS=(--spec-type routecrack2)
fi

# Container / macOS /dev/shm capacity guard
if [[ -z "\${MODCHIP_SLOT_SAVE_PATH:-}" && "\$SLOT_SAVE_DIR" == "/dev/shm" ]]; then
    SHM_FREE_KIB="\$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print \$4}' || echo 0)"
    if [[ ! -d /dev/shm || ! -w /dev/shm || "\${SHM_FREE_KIB:-0}" -lt 1048576 ]]; then
        SLOT_SAVE_DIR="${WORKSPACE_DIR}/cache"
        echo "[i] /dev/shm unavailable or <1 GiB free (\${SHM_FREE_KIB:-0} KiB); using disk slot cache at \${SLOT_SAVE_DIR}."
    fi
fi
mkdir -p "\$SLOT_SAVE_DIR" 2>/dev/null || true

LOAD_MODE="\${MODCHIP_LOAD_MODE:-\${DEFAULT_LOAD_MODE}}"

# Default --repack ON across v3.2.0 (zero-copy 3D MoE mmap preserved on Qwen3.8 + VNNI/AVX2 dense tile repack)
REPACK_ARGS=()
[[ "\${USE_REPACK:-1}" == "1" ]] && REPACK_ARGS=(--repack)

LAUNCH_PREFIX=()
if command -v taskset >/dev/null 2>&1 && [[ -n "\${TASKSET_MASK:-}" ]]; then
    LAUNCH_PREFIX=(taskset -c "\${TASKSET_MASK}")
fi

SERVER_HOST="\${SERVER_HOST:-${SERVER_HOST}}"
SERVER_PORT="\${SERVER_PORT:-${SERVER_PORT}}"
EXTRA_ARGS_ARR=()
if [[ -n "\${EXTRA_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA_ARGS_ARR <<< "\${EXTRA_SERVER_ARGS}"
fi
echo "[*] Starting llama-server v3.2.0 on http://\${SERVER_HOST}:\${SERVER_PORT} (Arch: \${MODEL_ARCH}, Decode: \${DECODE_THREADS}T / Prefill: \${BATCH_THREADS}T [mask: \${TASKSET_MASK}, poll=0], RC_TABLES_DIR=\${RC_TABLES_DIR}, Context: \${EFFECTIVE_CTX_SIZE})..."
exec \${LAUNCH_PREFIX[@]+"\${LAUNCH_PREFIX[@]}"} "${WORKSPACE_DIR}/bin/llama-server-nemotron" \\
  --model "\$MODEL" \\
  --port "\${SERVER_PORT}" \\
  --host "\${SERVER_HOST}" \\
  --no-webui \\
  --offline \\
  -c "\${EFFECTIVE_CTX_SIZE}" \\
  -np 1 \\
  -b 2048 \\
  -ub "\${UBATCH_SIZE}" \\
  -t "\${DECODE_THREADS}" \\
  -tb "\${BATCH_THREADS}" \\
  --poll 0 \\
  --poll-batch 0 \\
  --cache-type-k q8_0 \\
  --cache-type-v "\${CACHE_V_TYPE}" \\
  --flash-attn "\${FLASH_ATTN_MODE}" \\
  --load-mode "\${LOAD_MODE}" \\
  \${REPACK_ARGS[@]+"\${REPACK_ARGS[@]}"} \\
  --cache-reuse 256 \\
  --slot-save-path "\${SLOT_SAVE_DIR}" \\
  \${CKPT_ARGS[@]+"\${CKPT_ARGS[@]}"} \\
  \${SPEC_ARGS[@]+"\${SPEC_ARGS[@]}"} \\
  \${MODEL_EXTRA_ARGS[@]+"\${MODEL_EXTRA_ARGS[@]}"} \\
  --reasoning-format "\${REASONING_FORMAT:-deepseek}" \\
  --reasoning-budget "\${REASONING_BUDGET:-0}" \\
  \${EXTRA_ARGS_ARR[@]+"\${EXTRA_ARGS_ARR[@]}"} \\
  \${PASS_ARGS[@]+"\${PASS_ARGS[@]}"}
EOF
chmod +x "${REPO_ROOT}/start_server.sh"

cat > "${REPO_ROOT}/start_gateway.sh" <<EOF
#!/usr/bin/env bash
# Auto-generated by install.sh — Starts the Agentic SSE Gateway + Live Web UI
set -euo pipefail

export SERVER_HOST="\${SERVER_HOST:-${SERVER_HOST}}"
export LLAMA_BACKEND_URL="\${LLAMA_BACKEND_URL:-http://\${SERVER_HOST}:${SERVER_PORT}}"
export GATEWAY_HOST="\${GATEWAY_HOST:-${GATEWAY_HOST}}"
export GATEWAY_PORT="\${GATEWAY_PORT:-${GATEWAY_PORT}}"
echo "[*] Starting Agentic SSE Gateway & Live Web UI on http://\${GATEWAY_HOST}:\${GATEWAY_PORT} -> Backend \${LLAMA_BACKEND_URL}..."
echo "[i] 🌐 Open Web Chat & Telemetry HUD : http://\${GATEWAY_HOST}:\${GATEWAY_PORT}"
echo "[i] 🤖 Connect Claude Code CLI      : ANTHROPIC_BASE_URL=http://\${GATEWAY_HOST}:\${GATEWAY_PORT} claude"
exec python3 "${REPO_ROOT}/gateway/sse_gateway.py" "\${GATEWAY_PORT}"
EOF
chmod +x "${REPO_ROOT}/start_gateway.sh"

echo "=============================================================================="
echo " [+] LLAMA-MODCHIP v3.2.0 INSTALLATION COMPLETE!"
echo "=============================================================================="
echo " Quickest way to start & chat (Unified ./modchip CLI):"
echo "      ./modchip start        # Starts server + gateway in background"
echo "      ./modchip chat         # Interactive terminal chat with live tok/s"
echo "      ./modchip bench        # Run 30s benchmark & print shareable scorecard"
echo "      ./modchip claude       # Launch Claude Code connected to local CPU engine"
echo ""
echo " Or open the Live Web Chat & Telemetry HUD in your browser:"
echo "      http://${GATEWAY_HOST}:${GATEWAY_PORT}"
echo "=============================================================================="
