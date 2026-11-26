#!/usr/bin/env bash
# ==============================================================================
# build_trunk_quant.sh — requantize only the always-active Nemotron trunk
# ==============================================================================
# The official Nemotron-3.5-30B-A3B Q4_0 keeps its 46 routed-expert tensors at
# Q4_0 and its always-active trunk at Q8_0. Decode is dominated by that trunk,
# so lowering only the trunk is a cheap speedup; lowering the Mamba-2 and
# shared-expert weights too far costs tool-call accuracy (issue #39).
#
# Profiles:
#   trunk-q5  token embedding, ssm_in, ssm_out and shared experts -> Q5_0
#   trunk-q4  the same tensors -> Q4_0
#
# Both profiles:
#   - leave the 6 GQA attention projections, the router and output.head at Q8_0;
#   - copy the routed experts at the type already in the input GGUF. That type
#     is read from the GGUF header and passed as llama-quantize's base type;
#     llama-quantize copies a tensor whose target type already matches, so the
#     experts are not requantized.
#
# A stock llama-quantize handles all of these types; no patched binary is
# required. Measured on the official Q4_0 with 600 paired BFCL v3 prompts:
#   trunk-q5  +11% decode, KLD 0.016, BFCL 80.5% (unchanged)
#   trunk-q4  +18% decode, KLD 0.034, BFCL 71.8% (-8.7 points; the Mamba-2 and
#             shared-expert weights are what the tool-call drop comes from)
# ==============================================================================
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./scripts/build_trunk_quant.sh \
    --profile trunk-q5|trunk-q4 \
    --input /path/to/official-q4_0.gguf \
    --output /path/to/Trunk.gguf \
    [--quantize-bin /path/to/llama-quantize] \
    [--imatrix /path/to/imatrix.dat] \
    [--threads 16] \
    [--dry-run]

Profiles:
  trunk-q5  Trunk -> Q5_0. Attention, router and output head -> Q8_0.
            Experts copied at their input type.
  trunk-q4  Trunk -> Q4_0. Same attention/router/head and expert policy.
EOF
    exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_GGUF=""
OUTPUT_GGUF=""
PROFILE=""
QUANTIZE_BIN=""
IMATRIX_FILE=""
THREADS="$(nproc 2>/dev/null || echo 8)"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input|-i)        INPUT_GGUF="$2"; shift 2 ;;
        --output|-o)       OUTPUT_GGUF="$2"; shift 2 ;;
        --profile|-p)      PROFILE="$2"; shift 2 ;;
        --quantize-bin|-b) QUANTIZE_BIN="$2"; shift 2 ;;
        --imatrix|-m)      IMATRIX_FILE="$2"; shift 2 ;;
        --threads|-t)      THREADS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=1; shift 1 ;;
        --help|-h)         usage ;;
        *) echo "Unknown argument: $1" >&2; usage ;;
    esac
done

case "$PROFILE" in
    trunk-q5) TRUNK_TYPE="q5_0" ;;
    trunk-q4) TRUNK_TYPE="q4_0" ;;
    "") echo "ERROR: --profile is required (trunk-q5 or trunk-q4)" >&2; usage ;;
    *) echo "ERROR: Unsupported --profile '$PROFILE' (expected trunk-q5 or trunk-q4)" >&2; exit 1 ;;
esac

[[ -z "$INPUT_GGUF" || -z "$OUTPUT_GGUF" ]] && usage
[[ ! -f "$INPUT_GGUF" ]] && { echo "ERROR: Input GGUF not found: $INPUT_GGUF" >&2; exit 1; }

if [[ -z "$QUANTIZE_BIN" ]]; then
    for cand in \
        "${SCRIPT_DIR}/../workspace/bin/llama-quantize" \
        "${SCRIPT_DIR}/../workspace/bin/llama-quantize-modchip-linux-x86_64" \
        "$(command -v llama-quantize 2>/dev/null || true)"; do
        if [[ -n "$cand" && -x "$cand" ]]; then
            QUANTIZE_BIN="$cand"
            break
        fi
    done
fi
[[ -z "$QUANTIZE_BIN" || ! -x "$QUANTIZE_BIN" ]] && {
    echo "ERROR: llama-quantize not found. Run './modchip download-bin --all-tools' or pass --quantize-bin." >&2
    exit 1
}

EXPERT_TYPE="$(python3 "${SCRIPT_DIR}/gguf_tensor_type.py" "$INPUT_GGUF" \
    --match ffn_up_exps --match ffn_down_exps --match ffn_gate_exps)" || {
    echo "ERROR: could not read the routed-expert type from $INPUT_GGUF." >&2
    exit 1
}

ARGS=(
    "--allow-requantize"
    "--output-tensor-type" "q8_0"
    "--token-embedding-type" "$TRUNK_TYPE"
    "--tensor-type" "attn_k=q8_0"
    "--tensor-type" "attn_v=q8_0"
    "--tensor-type" "attn_q=q8_0"
    "--tensor-type" "attn_output=q8_0"
    "--tensor-type" "ffn_gate_inp=q8_0"
    "--tensor-type" "shexp=${TRUNK_TYPE}"
    "--tensor-type" "ssm_in=${TRUNK_TYPE}"
    "--tensor-type" "ssm_out=${TRUNK_TYPE}"
)
if [[ -n "$IMATRIX_FILE" ]]; then
    [[ ! -f "$IMATRIX_FILE" ]] && { echo "ERROR: imatrix file not found: $IMATRIX_FILE" >&2; exit 1; }
    ARGS=("--imatrix" "$IMATRIX_FILE" "${ARGS[@]}")
fi
if [[ "$DRY_RUN" -eq 1 ]]; then
    ARGS=("--dry-run" "${ARGS[@]}")
fi

BASE_TYPE="$(printf '%s' "$EXPERT_TYPE" | tr '[:lower:]' '[:upper:]')"

echo "======================================================================"
echo " Nemotron trunk-only quantizer"
echo "======================================================================"
echo " Input GGUF       : $INPUT_GGUF"
echo " Output GGUF      : $OUTPUT_GGUF"
echo " Profile          : $PROFILE"
echo " Trunk            : $TRUNK_TYPE (token embedding, ssm_in/out, shared experts)"
echo " Routed Experts   : unchanged ($EXPERT_TYPE from the GGUF header)"
echo " GQA Attn & Head  : q8_0"
echo " Threads          : $THREADS"
echo "======================================================================"

"$QUANTIZE_BIN" "${ARGS[@]}" "$INPUT_GGUF" "$OUTPUT_GGUF" "$BASE_TYPE" "$THREADS"

if [[ "$DRY_RUN" -eq 0 && -f "$OUTPUT_GGUF" ]]; then
    echo ""
    echo "[+] Successfully built the trunk-quantized GGUF:"
    ls -lh "$OUTPUT_GGUF"
fi
