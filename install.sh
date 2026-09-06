#!/usr/bin/env bash
# FlexPosit — convenience install script.
#
# This is the recipe we use to set up a working env. It is NOT the only way
# to install: `requirements.txt` is the source of truth. If you'd rather
# manage your own env, install torch from the CUDA 11.8 index manually,
# then `pip install -r requirements.txt && pip install -e .`.
#
# What this script does:
#   1. Creates a conda env `flexposit` (override with ENV_NAME=...).
#   2. Installs the CUDA 11.8 torch wheels + the rest of requirements.txt.
#   3. Installs the flexposit package in editable mode.
#   4. Runs a sanity check (nvidia-smi, torch, qtorch_plus CUDA JIT).
#
# Prerequisites:
#   - conda on PATH
#   - CUDA 11.8-compatible NVIDIA driver (nvidia-smi driver >= 520)
#   - CUDA toolkit with nvcc >= 11.8 on PATH (11.8, 12.x — anything above the
#     floor works). qtorch_plus JIT-compiles its CUDA extension on first
#     import; without a modern-enough nvcc the build fails, and on newer
#     GPUs (RTX 4090 sm_89, H100 sm_90) a pre-11.8 nvcc simply cannot target
#     the arch. If you only have an older /usr/bin/nvcc, point PATH at a
#     newer toolkit first:
#       export CUDA_HOME=/usr/local/cuda-11.8   # or /usr/local/cuda-12.x
#       export PATH="$CUDA_HOME/bin:$PATH"
#       export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

set -euo pipefail

export PYTHONNOUSERSITE=1
export PIP_USER=0

ENV_NAME="${ENV_NAME:-flexposit}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==========================================================="
echo "FlexPosit — Install"
echo "==========================================================="
echo "Conda env: $ENV_NAME"

command -v conda >/dev/null || { echo "ERROR: conda not on PATH"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
nvidia-smi -L

# --- nvcc preflight ---
# qtorch_plus JIT-compiles CUDA code, so nvcc must be present AND match torch's
# CUDA (11.8) closely enough to support the local GPU arch.
if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not on PATH. qtorch_plus needs the CUDA 11.8 toolkit to JIT-compile."
  echo "       Install it (e.g. /usr/local/cuda-11.8) and set CUDA_HOME/PATH — see"
  echo "       the header of this script for the exact env vars."
  exit 1
fi
NVCC_VER="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
NVCC_MAJ="${NVCC_VER%%.*}"
NVCC_MIN="${NVCC_VER##*.}"
if (( NVCC_MAJ < 11 )) || { (( NVCC_MAJ == 11 )) && (( NVCC_MIN < 8 )); }; then
  echo "ERROR: nvcc $NVCC_VER is too old — torch cu118 wheels + modern GPU archs"
  echo "       (RTX 4090 = sm_89, H100 = sm_90) require nvcc >= 11.8. Install the"
  echo "       CUDA 11.8 toolkit and prepend it to PATH — see the header of this script."
  exit 1
fi
echo "[nvcc] $NVCC_VER (>= 11.8) OK"

# Optional: auto-set TORCH_CUDA_ARCH_LIST from the local GPU's compute capability
# so JIT only compiles for the arch you'll actually run on. We also persist it
# to the env below via `conda env config vars set` so future shells inherit.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  GPU_CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)"
  if [[ -n "$GPU_CC" ]]; then
    export TORCH_CUDA_ARCH_LIST="$GPU_CC"
    echo "[gpu] compute capability $GPU_CC -> TORCH_CUDA_ARCH_LIST=$GPU_CC"
  fi
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[skip] Conda env '$ENV_NAME' already exists"
else
  echo "[create] Conda env '$ENV_NAME' (Python 3.10)"
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"

PY_PREFIX="$(python -c 'import sys; print(sys.prefix)')"
if [[ "$PY_PREFIX" != "$CONDA_PREFIX" ]]; then
  echo "ERROR: python's sys.prefix ($PY_PREFIX) != CONDA_PREFIX ($CONDA_PREFIX)"
  exit 1
fi

echo "[pip] Installing torch stack (CUDA 11.8)"
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

echo "[pip] Installing remaining deps"
python -m pip install -r "$REPO_ROOT/requirements.txt"

echo "[pip] Installing flexposit package (editable)"
python -m pip install -e "$REPO_ROOT"

echo
echo "==========================================================="
echo "Sanity checks"
echo "==========================================================="
python - <<'PY'
import sys, torch, transformers, qtorch_plus, datasets
print(f"python           : {sys.version.split()[0]}")
print(f"torch            : {torch.__version__}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"cuda device      : {name}")
    print(f"cuda mem (GB)    : {mem_gb:.1f}")
print(f"transformers     : {transformers.__version__}")
print(f"datasets         : {datasets.__version__}")

# qtorch_plus CUDA JIT smoke test — fail loudly here rather than at first sweep.
from qtorch_plus.quant import posit_quantize
x = torch.randn(4, device='cuda' if torch.cuda.is_available() else 'cpu')
posit_quantize(x, nsize=4, es=1)
print("qtorch_plus posit kernel OK")

import flexposit
print(f"flexposit        : importable from {flexposit.__file__ or '(namespace)'}")
PY

# Persist env vars into the conda env so future shells inherit them.
# PYTHONNOUSERSITE=1 prevents ~/.local/site-packages from shadowing the env's
# torch (silent footgun that lets the wrong torch load if the user has one).
# TORCH_CUDA_ARCH_LIST avoids re-JIT-compiling for all archs on every fresh shell.
echo
echo "[env] Persisting PYTHONNOUSERSITE + TORCH_CUDA_ARCH_LIST into '$ENV_NAME'"
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 >/dev/null
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  conda env config vars set -n "$ENV_NAME" TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" >/dev/null
fi

echo
echo "==========================================================="
echo "Install complete. Activate with:  conda activate $ENV_NAME"
echo "==========================================================="
