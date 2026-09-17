#!/usr/bin/env bash
# Regenerate sensitivity CSV via PPL-probe (canonical). Auto-dispatches to
# the Conv1D-aware variant for GPT-2 models. Needs a base checkpoint from
# scripts/01_quantize_base.sh. Env vars: MODEL, NSIZE, BASE_DIR, OUT_DIR, CHANNEL_WINDOW.

set -euo pipefail

MODEL="${1:-${MODEL:-phi-2}}"
NSIZE="${NSIZE:-4}"
BASE_DIR="${BASE_DIR:-out/${MODEL}_posit${NSIZE}}"
OUT_DIR="${OUT_DIR:-out/sens_${MODEL}}"
CHANNEL_WINDOW="${CHANNEL_WINDOW:-256}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/_env.sh
source "$REPO_ROOT/scripts/_env.sh"

if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: base checkpoint not found: $BASE_DIR (run scripts/01_quantize_base.sh)" >&2
    exit 1
fi

case "$MODEL" in
    gpt2*) MOD=flexposit.sensitivity.ppl_probe_conv1d ;;
    *)     MOD=flexposit.sensitivity.ppl_probe ;;
esac

echo "[regen-ppl] $MODEL via $MOD -> $OUT_DIR"
python -m "$MOD" \
    --model "$MODEL" --model_dir "$BASE_DIR" \
    --channel_window "$CHANNEL_WINDOW" --es_candidates 1 \
    --out_dir "$OUT_DIR"
