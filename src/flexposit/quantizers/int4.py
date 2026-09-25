#!/usr/bin/env python3
# quantize_int4.py — per-channel INT4 (range) weight quantization
# + WikiText-2 PPL evaluation.
#
# Per-channel range scaling: scale = qmax / max(|w_row|), qmax=7 for signed INT4.

import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from transformers.pytorch_utils import Conv1D

from flexposit.eval import perplexity, wikitext2_ids
from flexposit.utils import is_quant_linear, should_skip_layer

transformers.logging.set_verbosity_error()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12


def is_lin(m):
    return is_quant_linear(m)


def skip(name, m):
    return should_skip_layer(name, m)


@torch.no_grad()
def int4_per_channel_range(W: torch.Tensor, nbits: int = 4, is_conv1d: bool = False):
    """Per-output-channel signed INT quantize: scale = qmax / amax(row_over_Cin).

    nn.Linear weight is (Cout, Cin) -> reduce dim=-1 (last).
    HF Conv1D weight is (Cin, Cout) -> reduce dim=0 so the scale is still per-Cout.
    """
    qmin = -(1 << (nbits - 1))
    qmax = (1 << (nbits - 1)) - 1
    dtype, dev = W.dtype, W.device
    W_f = W.detach().float()
    reduce_dim = 0 if is_conv1d else -1
    amax = W_f.abs().amax(dim=reduce_dim, keepdim=True).clamp_min(EPS)
    scale = qmax / amax
    q = torch.clamp(torch.round(W_f * scale), qmin, qmax) / scale
    return q.to(dtype=dtype, device=dev)


def eval_wikitext2_ppl(model, tok, seqlen, forward_dtype):
    amp = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[forward_dtype]
    return perplexity(model, wikitext2_ids(tok), seqlen, autocast_dtype=amp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local dir")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--save_dir", required=True, help="Output dir (writes metrics.json)")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=False,
                   help="Run custom model code from the Hub (only needed for non-native architectures).")
    a = p.parse_args()

    os.makedirs(a.save_dir, exist_ok=True)
    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]

    print(f"[Load] {a.model}  dtype={a.dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=td, trust_remote_code=a.trust_remote_code, low_cpu_mem_usage=True
    ).to(DEV)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=a.trust_remote_code)

    n_quantized = 0
    for name, mod in model.named_modules():
        if skip(name, mod) or not is_lin(mod):
            continue
        if mod.weight.dim() != 2:
            continue
        mod.weight.data = int4_per_channel_range(
            mod.weight.data,
            is_conv1d=isinstance(mod, Conv1D),
        )
        n_quantized += 1
    print(f"[Quantize] INT4 (per-output-channel range) on {n_quantized} Linear/Conv1D layers", flush=True)

    ppl = eval_wikitext2_ppl(model, tok, a.seqlen, a.dtype)
    print(f"[Result] wikitext2_ppl = {ppl:.4f}", flush=True)

    with open(os.path.join(a.save_dir, "metrics.json"), "w") as f:
        json.dump({
            "scheme": "int4_per_channel_range",
            "bits": 4,
            "granularity": "per_channel",
            "wikitext2_ppl": float(ppl),
            "seqlen": a.seqlen,
            "dtype_forward": a.dtype,
            "n_quantized_layers": int(n_quantized),
        }, f, indent=2)
    print(f"[Saved] {a.save_dir}/metrics.json", flush=True)


if __name__ == "__main__":
    main()
