#!/usr/bin/env bash
# MPQ sweep on the Posit base checkpoint, guided by a sensitivity CSV.
# Env vars: MODEL, NSIZE, BASE_DIR, SENS_CSV, OUT_DIR, MPQ_ARGS.

set -euo pipefail

MODEL="${1:-${MODEL:-phi-2}}"
NSIZE="${NSIZE:-4}"
BASE_DIR="${BASE_DIR:-out/${MODEL}_posit${NSIZE}}"
SENS_CSV="${SENS_CSV:-src/flexposit/data/sensitivity/${MODEL}.csv}"
OUT_DIR="${OUT_DIR:-out/mpq_${MODEL}_sweep}"
MPQ_ARGS="${MPQ_ARGS:---sweep_bits_start 4.0 --sweep_bits_end 5.0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/_env.sh
source "$REPO_ROOT/scripts/_env.sh"

if [[ ! -f "$SENS_CSV" ]]; then
    echo "ERROR: sensitivity CSV not found: $SENS_CSV" >&2
    echo "       Shipped: $(ls src/flexposit/data/sensitivity/ 2>/dev/null | tr '\n' ' ')" >&2
    exit 1
fi
if [[ ! -d "$BASE_DIR" ]]; then
    echo "ERROR: base checkpoint not found: $BASE_DIR (run scripts/01_quantize_base.sh)" >&2
    exit 1
fi

echo "[02] MPQ $MODEL  base=$BASE_DIR  sens=$SENS_CSV  args=$MPQ_ARGS  out=$OUT_DIR"

# shellcheck disable=SC2086
python -m flexposit.mpq.channel_window \
    --model "$MODEL" \
    --base_dir "$BASE_DIR" \
    --sensitivity_csv "$SENS_CSV" \
    $MPQ_ARGS \
    --es_candidates 1 \
    --out_dir "$OUT_DIR"
