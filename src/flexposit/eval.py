"""WikiText-2 perplexity and FP8 activation quantization.

``perplexity`` is the single evaluator behind every CLI: chunked,
non-overlapping windows of ``seqlen`` tokens, mean token NLL accumulated in
fp32. The paper's numbers use it with ``seqlen=2048`` (capped at the model's
context length).
"""

from __future__ import annotations

import contextlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flexposit.formats import float_quantize
from flexposit.utils import quantizable_layers

FP8_E4M3_MAX = 240.0  # largest finite E4M3 value without NaN/Inf encodings (IEEE-style)

# Same data as the legacy "wikitext" name, which newer `datasets` no longer resolve.
WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def wikitext2_ids(tokenizer, split: str = "test") -> torch.Tensor:
    """Token ids [1, T] for WikiText-2 (raw), documents joined by blank lines."""
    from datasets import load_dataset
    data = load_dataset(*WIKITEXT2, split=split)
    return tokenizer("\n\n".join(data["text"]), return_tensors="pt",
                     add_special_tokens=False).input_ids


def model_device(model: nn.Module) -> torch.device:
    try:
        return next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def max_seqlen(model: nn.Module, seqlen: int) -> int:
    """Cap ``seqlen`` at the model's context window."""
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_ctx, int) and 0 < max_ctx < seqlen:
        return max_ctx
    return seqlen


@torch.no_grad()
def perplexity(model: nn.Module, ids: torch.Tensor, seqlen: int = 2048,
               autocast_dtype: torch.dtype | None = None, batch_size: int = 1,
               progress: bool = False) -> float:
    """Perplexity of ``model`` on ``ids`` [1, T] over non-overlapping ``seqlen`` chunks.

    ``autocast_dtype`` enables CUDA autocast (e.g. torch.float16); it is
    ignored on other devices.
    """
    model.eval()
    dev = model_device(model)
    nsamples = ids.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for seqlen={seqlen}")
    if dev.type == "cuda" and autocast_dtype is not None:
        autocast = lambda: torch.autocast("cuda", dtype=autocast_dtype)  # noqa: E731
    else:
        autocast = contextlib.nullcontext
    batch_size = max(1, int(batch_size))

    starts = range(0, nsamples, batch_size)
    if progress:
        from tqdm.auto import tqdm
        starts = tqdm(starts, desc=f"PPL (seqlen={seqlen})", unit="batch")

    nll_sum = 0.0
    for i in starts:
        j = min(i + batch_size, nsamples)
        batch = torch.cat([ids[:, k * seqlen:(k + 1) * seqlen] for k in range(i, j)], dim=0).to(dev)
        with autocast():
            logits = model(batch).logits
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nll_sum += loss.item() * (j - i) * seqlen
        if dev.type == "cuda":
            del batch, logits, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()
    return math.exp(nll_sum / (nsamples * seqlen))


def wikitext2_perplexity(model: nn.Module, tokenizer, seqlen: int = 2048,
                         autocast_dtype: torch.dtype | None = None, **kwargs) -> float:
    """Tokenize WikiText-2 test and return :func:`perplexity` at ``seqlen``."""
    return perplexity(model, wikitext2_ids(tokenizer), max_seqlen(model, seqlen),
                      autocast_dtype=autocast_dtype, **kwargs)


def fp8_activation_hook(exp_bits: int = 4, man_bits: int = 3):
    """Forward pre-hook: dynamic per-token FP(exp, man) activation quantization.

      amax  = |x|.max(dim=-1)            per token
      scale = amax / FP8_MAX
      q     = float_quantize((x / scale).clamp(-FP8_MAX, FP8_MAX)) * scale
    """
    fmax = FP8_E4M3_MAX if (exp_bits, man_bits) == (4, 3) else float("inf")

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            return inputs
        x = inputs[0]
        xf = x.contiguous().float()
        amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = amax / fmax
        q = float_quantize((xf / scale).clamp(-fmax, fmax), exp=exp_bits, man=man_bits) * scale
        return (q.to(x.dtype),) + tuple(inputs[1:])

    return hook


def add_fp8_activation_quant(model: nn.Module, exp_bits: int = 4, man_bits: int = 3) -> list:
    """Register :func:`fp8_activation_hook` on every quantizable Linear/Conv1D.

    Returns the hook handles (call ``.remove()`` on each to undo).
    """
    hook = fp8_activation_hook(exp_bits, man_bits)
    return [mod.register_forward_pre_hook(hook) for _, mod in quantizable_layers(model)]


def lm_eval(model: nn.Module, tokenizer, tasks, batch_size: int = 8, limit=None, **kwargs) -> dict:
    """Run lm-evaluation-harness ``tasks`` on an in-memory (e.g. quantized) model.

    Needs ``pip install 'flexposit[eval]'``. Returns lm-eval's per-task results;
    extra keyword arguments go to ``lm_eval.simple_evaluate``.

        flexposit.eval.lm_eval(model, tok, ["arc_easy", "hellaswag"])
    """
    try:
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM
    except ImportError as e:
        raise ImportError("lm_eval() needs lm-evaluation-harness: pip install 'flexposit[eval]'") from e
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    tasks = [tasks] if isinstance(tasks, str) else list(tasks)
    return simple_evaluate(model=lm, tasks=tasks, limit=limit, **kwargs)["results"]
