#!/usr/bin/env bash
# Quantize weights to Posit(NSIZE, es=1) base. Env vars: MODEL, NSIZE, OUT_DIR.

set -euo pipefail

MODEL="${1:-${MODEL:-phi-2}}"
NSIZE="${NSIZE:-4}"
OUT_DIR="${OUT_DIR:-out/${MODEL}_posit${NSIZE}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/_env.sh
source "$REPO_ROOT/scripts/_env.sh"

echo "[01] Quantize $MODEL -> Posit($NSIZE,1) -> $OUT_DIR"
python -m flexposit.quantizers.posit \
    --model "$MODEL" --nsize "$NSIZE" --es_candidates 1 \
    --save_dir "$OUT_DIR"
