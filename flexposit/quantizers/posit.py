#!/usr/bin/env python3
# posit.py — base per-channel Posit quantizer
# Fixed-nsize per-layer; per-channel search for (scale, es).
# GPU-friendly: batched per-channel search (no float(tensor) syncs inside loops).
# Saves model + log + metrics.json (chunked non-overlapping PPL).

import argparse, json, os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import transformers.modeling_utils as modeling_utils

from qtorch_plus.quant import posit_quantize, float_quantize

transformers.logging.set_verbosity_error()
EPS = 1e-8

# -------------------- Model presets --------------------
MODEL_PRESETS = {
    
    "gpt2":         {"hf_id": "gpt2"},
    "gpt2-medium":  {"hf_id": "gpt2-medium"},
    "gpt2-large":   {"hf_id": "gpt2-large"},
    "gpt2-xl":      {"hf_id": "gpt2-xl"},

    "opt-125m":     {"hf_id": "facebook/opt-125m"},
    "opt-350m":     {"hf_id": "facebook/opt-350m"},
    "opt-1.3b":     {"hf_id": "facebook/opt-1.3b"},
    "opt-2.7b":     {"hf_id": "facebook/opt-2.7b"},
    "opt-6.7b":     {"hf_id": "facebook/opt-6.7b"},

    "bloom-7b1":    {"hf_id": "bigscience/bloom-7b1"},

    "phi-2":        {"hf_id": "microsoft/phi-2"},
    "yi-6b":        {"hf_id": "01-ai/Yi-6B", "trust_remote_code": True},
    "llama-2-7b":   {"hf_id": "meta-llama/Llama-2-7b-hf"},
    "llama-3-8b":   {"hf_id": "meta-llama/Meta-Llama-3-8B"},
    
    "qwen2.5-14b": {
    "hf_id": "Qwen/Qwen2.5-14B",
    "trust_remote_code": True,
    "use_fast_tokenizer": False,
    "requires_auth": False,
},
    "llama-2-7b":   {"hf_id": "meta-llama/Llama-2-7b-hf"},
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "qwen2-7b": {
        "hf_id": "Qwen/Qwen2-7B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    "deepseek-llm-7b": {
        "hf_id": "deepseek-ai/deepseek-llm-7b-base",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    "phi-3-small": {
        "hf_id": "microsoft/Phi-3-small-8k-instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    # GATED
    "llama-2-13b": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "requires_auth": True,
    },
}

# -------------------- Args --------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default="mistral-7b")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # HF token (for gated models); can also use env HF_TOKEN
    p.add_argument("--hf_token", default=None)

    # fixed nsize (per layer), per-channel search for (scale, es)
    p.add_argument("--nsize", type=int, default=4)
    p.add_argument(
        "--weight_format",
        choices=["posit4", "posit5"],
        default=None,
        help="Weight format alias (posit4 → nsize=4, posit5 → nsize=5). "
             "Optional — if omitted, derived from --nsize."
    )
    p.add_argument("--es_candidates", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--log2_min", type=int, default=-8)
    p.add_argument("--log2_max", type=int, default=9)

    # GPU batching for per-channel search (important!)
    p.add_argument("--ch_batch", type=int, default=64,
                   help="How many output channels to search at once per layer. Increase if you have headroom.")

    # activation quant (optional; off by default)
    p.add_argument("--use_act_quant", action="store_true", default=False)
    p.add_argument("--act_exp", type=int, default=4)
    p.add_argument("--act_man", type=int, default=3)

    # quantization scope
    p.add_argument("--skip_lm_head", action="store_true", default=True)
    p.add_argument("--quantize_embeddings", action="store_true", default=False)

    # Chunked non-overlapping WikiText-2 PPL
    p.add_argument("--ppl_seqlen", type=int, default=2048)

    # saving
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_log_name", default="quant_log.json")
    return p.parse_args()

def get_torch_dtype(tag: str):
    if tag == "fp16":
        return torch.float16
    if tag == "bf16":
        return torch.bfloat16
    return torch.float32

# -------------------- Act quant hook --------------------
def make_act_hook(use_act: bool, exp_bits: int, man_bits: int):
    if not use_act:
        def passthrough(_m, inputs): return inputs
        return passthrough

    def linear_activation(x: torch.Tensor):
        x_fp32 = x.float()  # float_quantize requires fp32 input
        q_fp32 = float_quantize(x_fp32, exp=exp_bits, man=man_bits, rounding="nearest")
        return q_fp32.to(x.dtype)  # cast back to module's operating dtype

    def hook(_m, inputs):
        return (linear_activation(inputs[0]),)

    return hook

# -------------------- Layer filters --------------------
def is_quant_linear(mod: nn.Module):
    if isinstance(mod, nn.Linear):
        return True
    if isinstance(mod, modeling_utils.Conv1D):
        return True
    name = mod.__class__.__name__.lower()
    if "linear" in name and hasattr(mod, "weight") and isinstance(getattr(mod, "weight"), torch.Tensor):
        return True
    return False


def _to_cout_first(mod: nn.Module, W: torch.Tensor) -> torch.Tensor:
    """Return W arranged so dim 0 is Cout.

    nn.Linear stores (Cout, Cin) — passed through.
    HF Conv1D stores (Cin, Cout) — transposed so per-row iteration is per-Cout.
    """
    if isinstance(mod, modeling_utils.Conv1D):
        return W.transpose(0, 1).contiguous()
    return W


def _from_cout_first(mod: nn.Module, q_w: torch.Tensor) -> torch.Tensor:
    """Undo _to_cout_first: transpose back to storage layout for Conv1D."""
    if isinstance(mod, modeling_utils.Conv1D):
        return q_w.transpose(0, 1).contiguous()
    return q_w

def should_skip_layer(name: str, mod: nn.Module, skip_lm_head: bool, quantize_embeddings: bool):
    if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
        return True
    if isinstance(mod, nn.Embedding) and not quantize_embeddings:
        return True
    return False

def summarize_format_usage(ch_meta):
    """
    Summarize per-channel selected format statistics.
    Supports:
      - mixed meta: {"selected_format": "..."}
      - single-format meta: {"format": "..."}
    """
    total = len(ch_meta)
    counts = {}
    for item in ch_meta:
        fmt = item.get("selected_format", item.get("format", "unknown"))
        counts[fmt] = counts.get(fmt, 0) + 1
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return {"total_channels": total, "counts": counts, "ratios": ratios}

# -------------------- GPU-friendly per-channel search --------------------
@torch.no_grad()
def _per_channel_sqnr_search_batched(
    flat: torch.Tensor,
    nsize: int,
    es_cands,
    sweep_scales,
    ch_batch: int,
    return_quantized: bool,
):
    """
    flat: [Cout, K] fp32 on device.
    If return_quantized: returns (q_out [Cout,K], per_ch_meta list).
    Else: returns (scale_vec [Cout], es_vec [Cout] int64, per_ch_meta list).
    """
    dev = flat.device
    Cout, K = flat.shape
    max_es = max(0, nsize - 1)
    es_list = [int(e) for e in es_cands if int(e) <= max_es] or [0]
    scales = torch.tensor([float(s) for s in sweep_scales], device=dev, dtype=torch.float32)

    q_out = torch.empty_like(flat) if return_quantized else None
    per_ch_meta = []
    scale_out = torch.empty((Cout,), device=dev, dtype=torch.float32)
    es_out = torch.empty((Cout,), device=dev, dtype=torch.int64)

    for c0 in range(0, Cout, ch_batch):
        c1 = min(Cout, c0 + ch_batch)
        X = flat[c0:c1]
        B = X.size(0)
        sp = torch.sum(X * X, dim=1) + EPS

        best_sqnr = torch.full((B,), -1e30, device=dev, dtype=torch.float32)
        best_q = torch.zeros((B, K), device=dev, dtype=torch.float32)
        best_es = torch.zeros((B,), device=dev, dtype=torch.int32)
        best_l2 = torch.zeros((B,), device=dev, dtype=torch.int32)

        for s in scales:
            Xs = X * s
            for es in es_list:
                q = posit_quantize(Xs, nsize=nsize, es=int(es), scale=1.0) / s
                noise = torch.sum((X - q) ** 2, dim=1) + EPS
                sqnr = 10.0 * torch.log10(sp / noise)
                mask = sqnr > best_sqnr
                if mask.any():
                    best_sqnr = torch.where(mask, sqnr, best_sqnr)
                    best_q = torch.where(mask[:, None], q, best_q)
                    best_es = torch.where(mask, torch.tensor(es, device=dev, dtype=torch.int32), best_es)
                    l2 = int(round(math.log2(float(s.item()))))
                    best_l2 = torch.where(mask, torch.tensor(l2, device=dev, dtype=torch.int32), best_l2)

        if return_quantized:
            q_out[c0:c1] = best_q

        sch = torch.pow(2.0, best_l2.to(dtype=torch.float32))
        scale_out[c0:c1] = sch
        es_out[c0:c1] = best_es.to(torch.int64)

        best_sqnr_cpu = best_sqnr.detach().cpu().tolist()
        best_es_cpu = best_es.detach().cpu().tolist()
        best_l2_cpu = best_l2.detach().cpu().tolist()
        for i in range(B):
            per_ch_meta.append({
                "channel": int(c0 + i),
                "format": f"posit{int(nsize)}",
                "sqnr": float(best_sqnr_cpu[i]),
                "log2_scale": int(best_l2_cpu[i]),
                "es": int(best_es_cpu[i]),
                "nsize": int(nsize),
            })

    if return_quantized:
        return q_out, per_ch_meta
    return scale_out, es_out, per_ch_meta


@torch.no_grad()
def per_channel_scales_es_batched(
    layer_weight: torch.Tensor,
    nsize: int,
    es_cands,
    sweep_scales,
    ch_batch: int,
):
    """SQNR-optimal per-output-channel scale (float) and es (int) for GPTQ+Posit path."""
    dev = layer_weight.device
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    flat = W.view(W.size(0), -1)
    scale_vec, es_vec, meta = _per_channel_sqnr_search_batched(
        flat, nsize, es_cands, sweep_scales, ch_batch, return_quantized=False
    )
    return scale_vec, es_vec, meta


@torch.no_grad()
def per_channel_quantize_fixed_nsize_batched(layer_weight: torch.Tensor,
                                             nsize: int,
                                             es_cands,
                                             sweep_scales,
                                             ch_batch: int):
    """
    layer_weight: [Cout, K] tensor on model device.
    Returns:
      q_w: quantized weight, same shape/device/dtype as layer_weight
      per_ch_meta: list of dicts (channel, log2_scale, es, sqnr)  (small)
    """
    dev = layer_weight.device
    out_dtype = layer_weight.dtype
    W = layer_weight.detach().to(dtype=torch.float32, device=dev)
    flat = W.view(W.size(0), -1)
    q_out, per_ch_meta = _per_channel_sqnr_search_batched(
        flat, nsize, es_cands, sweep_scales, ch_batch, return_quantized=True
    )
    q_w = q_out.view_as(W).to(dtype=out_dtype, device=dev)
    return q_w, per_ch_meta

@torch.no_grad()
def eval_wikitext_ppl(model, tokenizer, seqlen: int, forward_dtype: str):
    model.eval()

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False)
    ids = enc.input_ids  # CPU

    nsamples = ids.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for seqlen={seqlen}")

    try:
        dev = next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_amp = (forward_dtype in ["fp16", "bf16"]) and (dev.type == "cuda")
    amp_dtype = torch.float16 if forward_dtype == "fp16" else torch.bfloat16

    if dev.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        autocast_ctx = _NoOp()

    nll_sum = 0.0
    for i in tqdm(range(nsamples), desc="PPL", unit="chunk"):
        batch = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        with autocast_ctx:
            logits = model(batch).logits  # (B, T, V)

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                               shift_labels.view(-1))
        nll_sum += (loss.item() * seqlen)

        if dev.type == "cuda":
            del batch, logits, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()

    ppl = math.exp(nll_sum / (nsamples * seqlen))
    return float(ppl)
def make_fwd_autocast(device: torch.device, forward_dtype: str):
    use_amp = (forward_dtype in ["fp16", "bf16"]) and (device.type == "cuda")
    amp_dtype = torch.float16 if forward_dtype == "fp16" else torch.bfloat16
    if device.type == "cuda":
        return lambda: torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)

    class _NoOp:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    return lambda: _NoOp()


# -------------------- Main --------------------
def main():
    args = get_args()
    device = torch.device(args.device)
    torch_dtype = get_torch_dtype(args.dtype)

    preset = MODEL_PRESETS[args.model]
    hf_id = preset["hf_id"]
    trust_remote = preset.get("trust_remote_code", False)
    use_fast = preset.get("use_fast_tokenizer", True)
    requires_auth = preset.get("requires_auth", False)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", None)
    if requires_auth and not hf_token:
        raise RuntimeError(
            f"Model '{args.model}' is gated and requires an HF token.\n"
            f"Set env HF_TOKEN or pass --hf_token <token>."
        )

    print(f"[Load] {args.model} -> {hf_id}  (dtype={args.dtype}, device={device})")
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote,
        token=hf_token
    ).to(device)

    tok = AutoTokenizer.from_pretrained(
        hf_id,
        use_fast=use_fast,
        trust_remote_code=trust_remote,
        token=hf_token
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    # Auto-cap ppl_seqlen by model context if available
    max_ctx = getattr(model.config, "max_position_embeddings", None)
    if isinstance(max_ctx, int) and max_ctx > 0 and args.ppl_seqlen > max_ctx:
        print(f"[Info] ppl_seqlen {args.ppl_seqlen} > max_position_embeddings {max_ctx}; capping to {max_ctx}")
        args.ppl_seqlen = max_ctx

    # Candidate scales (powers of two)
    sweep_scales = [2.0 ** k for k in range(args.log2_min, args.log2_max + 1)]
    act_hook = make_act_hook(args.use_act_quant, args.act_exp, args.act_man)

    # Collect target layers
    target_layers = []
    for name, mod in model.named_modules():
        if should_skip_layer(name, mod, skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings):
            continue
        if is_quant_linear(mod) and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            if mod.weight.dim() == 2:
                target_layers.append((name, mod))

    # weight_format is a human-readable alias; nsize is the source of truth.
    # If --weight_format was passed, it overrides --nsize (with a warning on mismatch).
    if args.weight_format is not None:
        target_nsize = {"posit4": 4, "posit5": 5}[args.weight_format]
        if args.nsize != target_nsize:
            print(f"[Warn] --weight_format={args.weight_format} overrides --nsize {args.nsize} -> {target_nsize}")
            args.nsize = target_nsize
    else:
        args.weight_format = f"posit{args.nsize}"

    print(f"\n[Quantize] weight_format={args.weight_format} | "
          f"nsize={args.nsize} | es∈{args.es_candidates}, log2∈[{args.log2_min},{args.log2_max}], ch_batch={args.ch_batch}")
    print(f"Layers to quantize: {len(target_layers)}")

    quant_log = {}
    global_usage_counts = {}
    global_total_channels = 0
    with torch.no_grad():
        for name, mod in tqdm(target_layers, desc="Quantizing layers"):
            # Per-channel search assumes dim 0 = Cout. HF Conv1D stores (Cin, Cout);
            # transpose in and back out so scale/es search is per-output-channel.
            W_in = _to_cout_first(mod, mod.weight)
            q_w, per_ch = per_channel_quantize_fixed_nsize_batched(
                W_in, nsize=args.nsize, es_cands=args.es_candidates,
                sweep_scales=sweep_scales, ch_batch=args.ch_batch
            )
            mod.weight.data = _from_cout_first(mod, q_w)

            if args.use_act_quant and not getattr(mod, "_act_quant_hooked", False):
                mod.register_forward_pre_hook(act_hook)
                setattr(mod, "_act_quant_hooked", True)

            layer_usage = summarize_format_usage(per_ch)
            for k, v in layer_usage["counts"].items():
                global_usage_counts[k] = global_usage_counts.get(k, 0) + int(v)
            global_total_channels += int(layer_usage["total_channels"])
            quant_log[name] = {
                "mode": f"weight_only_{args.weight_format}",
                "shape": list(q_w.shape),
                "nsize": int(args.nsize),
                "format_usage": layer_usage,
                "channels": per_ch,
            }

    if global_total_channels > 0:
        global_usage_ratios = {
            k: (v / global_total_channels if global_total_channels > 0 else 0.0)
            for k, v in global_usage_counts.items()
        }
        quant_log["_summary"] = {
            "weight_format": args.weight_format,
            "format_usage": {
                "total_channels": global_total_channels,
                "counts": global_usage_counts,
                "ratios": global_usage_ratios,
            },
        }
    # Save model + tokenizer + log
    os.makedirs(args.save_dir, exist_ok=True)
    model.save_pretrained(args.save_dir, safe_serialization=True)
    tok.save_pretrained(args.save_dir)
    with open(os.path.join(args.save_dir, args.save_log_name), "w") as f:
        json.dump(quant_log, f, indent=2)

    print(f"[Saved] model+tokenizer -> {args.save_dir}")
    print(f"[Saved] log -> {os.path.join(args.save_dir, args.save_log_name)}")
    format_usage_summary = quant_log.get("_summary", {}).get(
        "format_usage", {"total_channels": 0, "counts": {}, "ratios": {}}
    )
    print(f"[Format Usage] total={format_usage_summary['total_channels']} "
          f"counts={format_usage_summary['counts']} ratios={format_usage_summary['ratios']}")

    # Chunked non-overlapping PPL
    ppl = eval_wikitext_ppl(model, tok, seqlen=args.ppl_seqlen, forward_dtype=args.dtype)
    print(f"\nPerplexity ({args.model}, {args.weight_format}, nsize={args.nsize}) [seqlen={args.ppl_seqlen}, fwd={args.dtype}]: {ppl:.4f}")

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump({
            "ppl": ppl,
            "model": args.model,
            "nsize": int(args.nsize),
            "weight_format": args.weight_format,
            "format_usage": format_usage_summary,
            "ppl_seqlen": args.ppl_seqlen,
            "dtype_forward": args.dtype,
        }, f, indent=2)

    print(f"[Saved] metrics -> {os.path.join(args.save_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
