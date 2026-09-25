#!/usr/bin/env bash
# Sourced by the other scripts. If flexposit is already importable (e.g. after
# `pip install -e .`), use the current Python as is. Otherwise activate the
# conda env from install.sh (named `flexposit`, or FLEXPOSIT_ENV).

FLEXPOSIT_ENV="${FLEXPOSIT_ENV:-flexposit}"

if ! python -c "import flexposit" >/dev/null 2>&1; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: flexposit is not installed in this Python and conda is not on PATH." >&2
        echo "       Run 'pip install -e .' from the repo root, or 'bash install.sh'." >&2
        return 1 2>/dev/null || exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda activate "$FLEXPOSIT_ENV" 2>/dev/null; then
        echo "ERROR: flexposit is not installed in this Python, and conda env '$FLEXPOSIT_ENV' does not exist." >&2
        echo "       Run 'pip install -e .' from the repo root, or 'bash install.sh'." >&2
        return 1 2>/dev/null || exit 1
    fi
fi
