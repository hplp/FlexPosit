#!/usr/bin/env python3
# ppl.py — Load a saved checkpoint (or HF model) and eval WikiText-2 PPL,
# with optional per-Linear FP8 (E4M3) activation quantization.

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from flexposit.eval import add_fp8_activation_quant, perplexity, wikitext2_ids

transformers.logging.set_verbosity_error()
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local checkpoint dir")
    p.add_argument("--act_quant", choices=["none", "fp8_e4m3"], default="none")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args()

    td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    print(f"[Load] {a.model}  dtype={a.dtype}  act_quant={a.act_quant}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=td, trust_remote_code=a.trust_remote_code, low_cpu_mem_usage=True).to(DEV)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True, trust_remote_code=a.trust_remote_code)

    # Auto-cap seqlen by model context window (gpt2 = 1024, most others 2048+).
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_ctx, int) and max_ctx > 0 and a.seqlen > max_ctx:
        print(f"[Info] --seqlen {a.seqlen} > model.max_position_embeddings {max_ctx}; "
              f"capping to {max_ctx}", flush=True)
        a.seqlen = max_ctx

    if a.act_quant == "fp8_e4m3":
        k = len(add_fp8_activation_quant(model, exp_bits=4, man_bits=3))
        print(f"[Act quant] FP8-E4M3 hook on {k} Linear/Conv1D modules", flush=True)

    ppl = perplexity(model, wikitext2_ids(tok), a.seqlen)
    print(f"[Result] wikitext2_ppl = {ppl:.4f}", flush=True)


if __name__ == "__main__":
    main()
