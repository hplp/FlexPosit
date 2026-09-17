#!/usr/bin/env bash
# Regenerate sensitivity CSV via Fisher-diagonal (faster proxy for PPL-probe).
# Runs on the FP reference model — no base checkpoint needed.
# Env vars: MODEL, OUT_CSV, CHANNEL_WINDOW, B_LOW, B_HIGH.

set -euo pipefail

MODEL="${1:-${MODEL:-phi-2}}"
OUT_CSV="${OUT_CSV:-out/fisher_${MODEL}.csv}"
CHANNEL_WINDOW="${CHANNEL_WINDOW:-256}"
B_LOW="${B_LOW:-4}"
B_HIGH="${B_HIGH:-5}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/_env.sh
source "$REPO_ROOT/scripts/_env.sh"

mkdir -p "$(dirname "$OUT_CSV")"

echo "[regen-fisher] $MODEL -> $OUT_CSV  (bits ${B_LOW}->${B_HIGH}, cw=${CHANNEL_WINDOW})"
python -m flexposit.sensitivity.fisher \
    --model "$MODEL" --channel_window "$CHANNEL_WINDOW" \
    --out_csv "$OUT_CSV" \
    --b_low "$B_LOW" --b_high "$B_HIGH"
