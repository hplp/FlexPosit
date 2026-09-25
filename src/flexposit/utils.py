"""Model-walking helpers shared by the quantizers, MPQ drivers and API."""

from __future__ import annotations

import os
from typing import Iterator

import torch
import torch.nn as nn
import transformers
from transformers.pytorch_utils import Conv1D

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

# from_pretrained's dtype argument was renamed torch_dtype -> dtype in transformers 4.56.
_DTYPE_KW = "dtype" if tuple(int(v) for v in transformers.__version__.split(".")[:2]) >= (4, 56) else "torch_dtype"


def default_device() -> str:
    """"cuda" if an NVIDIA GPU is available, else "mps" on Apple silicon, else "cpu"."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_conv1d(mod: nn.Module) -> bool:
    """HF Conv1D (GPT-2 family) stores weights as (Cin, Cout)."""
    return isinstance(mod, Conv1D)


def is_quant_linear(mod: nn.Module, include_named_linear: bool = False) -> bool:
    """nn.Linear or HF Conv1D.

    ``include_named_linear`` also accepts any module whose class name contains
    "linear" and that has a tensor ``weight`` (custom Linear subclasses in
    trust_remote_code models). flexposit.quantizers.posit uses it.
    """
    if isinstance(mod, nn.Linear) or is_conv1d(mod):
        return True
    if include_named_linear:
        name = mod.__class__.__name__.lower()
        return "linear" in name and isinstance(getattr(mod, "weight", None), torch.Tensor)
    return False


def should_skip_layer(name: str, mod: nn.Module, skip_lm_head: bool = True,
                      quantize_embeddings: bool = False) -> bool:
    if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
        return True
    if isinstance(mod, nn.Embedding) and not quantize_embeddings:
        return True
    return False


def quantizable_layers(model: nn.Module, skip_lm_head: bool = True,
                       quantize_embeddings: bool = False,
                       include_named_linear: bool = False) -> Iterator[tuple[str, nn.Module]]:
    """(name, module) for every 2-D Linear/Conv1D weight that gets quantized."""
    for name, mod in model.named_modules():
        if should_skip_layer(name, mod, skip_lm_head, quantize_embeddings):
            continue
        if not is_quant_linear(mod, include_named_linear):
            continue
        w = getattr(mod, "weight", None)
        if isinstance(w, torch.Tensor) and w.dim() == 2:
            yield name, mod


def to_cout_first(mod: nn.Module, w: torch.Tensor) -> torch.Tensor:
    """Arrange ``w`` so dim 0 is the output channel (Conv1D is transposed)."""
    return w.transpose(0, 1).contiguous() if is_conv1d(mod) else w


def from_cout_first(mod: nn.Module, w: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_cout_first`."""
    return w.transpose(0, 1).contiguous() if is_conv1d(mod) else w


def resolve_model(name_or_path: str) -> tuple[str, dict]:
    """Map a MODEL_PRESETS short name to (hf_id, preset); pass other ids through."""
    from flexposit.models import MODEL_PRESETS
    preset = MODEL_PRESETS.get(name_or_path)
    if preset is None:
        return name_or_path, {}
    return preset["hf_id"], preset


def load_model(name_or_path: str, dtype: str | torch.dtype = "fp16", device: str | None = None,
               hf_token: str | None = None):
    """Load a causal LM and its tokenizer by preset short name, HF id or local path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_id, preset = resolve_model(name_or_path)
    torch_dtype = DTYPES[dtype] if isinstance(dtype, str) else dtype
    trust = preset.get("trust_remote_code", False)
    token = hf_token or os.environ.get("HF_TOKEN")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, **{_DTYPE_KW: torch_dtype}, trust_remote_code=trust, token=token,
        low_cpu_mem_usage=True)
    if device is None:
        device = default_device()
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained(
        hf_id, use_fast=preset.get("use_fast_tokenizer", True), trust_remote_code=trust, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
