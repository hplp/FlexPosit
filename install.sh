#!/usr/bin/env bash
# FlexPosit — convenience install script.
#
# `pip install -e .` is enough for most uses. This script reproduces the exact
# environment behind the MICRO 2026 results (pinned versions in
# requirements.txt):
#   1. Creates a conda env `flexposit` (override with ENV_NAME=...).
#   2. Installs torch 2.4.1 (CUDA 11.8 wheels if an NVIDIA GPU is present,
#      CPU wheels otherwise) and the rest of requirements.txt.
#   3. Installs the flexposit package in editable mode and runs a sanity check.
#
# No CUDA toolkit or compiler is needed: the Posit kernels are pure PyTorch.

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
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi -L
  TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
  echo "[gpu] No NVIDIA GPU found — installing CPU torch (fine for small models and tests)."
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
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

echo "[pip] Installing torch 2.4.1 from $TORCH_INDEX"
python -m pip install --index-url "$TORCH_INDEX" torch==2.4.1

echo "[pip] Installing remaining deps"
python -m pip install -r "$REPO_ROOT/requirements.txt"

echo "[pip] Installing flexposit package (editable)"
python -m pip install -e "$REPO_ROOT"

echo
echo "==========================================================="
echo "Sanity checks"
echo "==========================================================="
python - <<'PY'
import sys, torch, transformers, datasets
print(f"python           : {sys.version.split()[0]}")
print(f"torch            : {torch.__version__}")
print(f"cuda available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device      : {torch.cuda.get_device_name(0)}")
    print(f"cuda mem (GB)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
print(f"transformers     : {transformers.__version__}")
print(f"datasets         : {datasets.__version__}")

import flexposit
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(flexposit.posit_quantize(torch.tensor([0.3, -1.7], device=dev), nsize=4, es=1).tolist())
print(f"flexposit        : {flexposit.__version__} OK")
PY

# PYTHONNOUSERSITE=1 prevents ~/.local/site-packages from shadowing the env's torch.
conda env config vars set -n "$ENV_NAME" PYTHONNOUSERSITE=1 >/dev/null

echo
echo "==========================================================="
echo "Install complete. Activate with:  conda activate $ENV_NAME"
echo "==========================================================="
