#!/usr/bin/env bash
# Sourced by the other scripts. Activates a conda env for FlexPosit and, if the
# default nvcc is < 11.8, points PATH at the newest /usr/local/cuda-* so
# qtorch_plus's JIT compile can target modern GPU archs.
#
# Env activation policy: if the caller already has any conda env active, we
# respect it (so a pre-activated / custom-named env just works). Only when no
# env is active do we auto-activate $FLEXPOSIT_ENV (default: flexposit).

FLEXPOSIT_ENV="${FLEXPOSIT_ENV:-flexposit}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not on PATH; cannot activate env '$FLEXPOSIT_ENV'." >&2
        return 1 2>/dev/null || exit 1
    fi
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$FLEXPOSIT_ENV"
fi

_flexposit_nvcc_ok() {
    command -v nvcc >/dev/null 2>&1 || return 1
    local ver maj min
    ver=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')
    [[ -n "$ver" ]] || return 1
    maj=${ver%%.*}; min=${ver##*.}
    if (( maj > 11 )); then return 0; fi
    if (( maj == 11 )) && (( min >= 8 )); then return 0; fi
    return 1
}

if ! _flexposit_nvcc_ok; then
    latest=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "${latest:-}" && -x "$latest/bin/nvcc" ]]; then
        export CUDA_HOME="$latest"
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
    fi
fi

if ! _flexposit_nvcc_ok; then
    echo "ERROR: no CUDA toolkit >= 11.8 found on PATH or under /usr/local/cuda-*." >&2
    echo "       See install.sh header for how to set CUDA_HOME/PATH manually." >&2
    return 1 2>/dev/null || exit 1
fi

if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    [[ -n "${cc:-}" ]] && export TORCH_CUDA_ARCH_LIST="$cc"
fi

unset -f _flexposit_nvcc_ok
