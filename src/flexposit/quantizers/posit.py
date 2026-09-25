#!/usr/bin/env python3
# posit.py — base per-channel Posit quantizer (CLI).
# Every Linear/Conv1D weight is quantized to Posit(nsize, es) with a per-output-
# channel power-of-two scale chosen for best SQNR (flexposit.core).
# Saves model + quant_log.json + metrics.json (chunked non-overlapping PPL).

import argparse, json, math, os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

from flexposit.core import search_channels
from flexposit.eval import perplexity, wikitext2_ids
from flexposit.formats import float_quantize
from flexposit.models import MODEL_PRESETS
from flexposit.utils import DTYPES, from_cout_first, quantizable_layers, to_cout_first

transformers.logging.set_verbosity_error()

# Kept for tests and external callers of the pre-refactor names.
_to_cout_first = to_cout_first
_from_cout_first = from_cout_first


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default="mistral-7b")
    p.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # HF token (for gated models); can also use env HF_TOKEN
    p.add_argument("--hf_token", default=None)

    # fixed nsize (per layer), per-channel search for the power-of-two scale
    p.add_argument("--nsize", type=int, default=4)
    p.add_argument(
        "--weight_format",
        choices=["posit4", "posit5"],
        default=None,
        help="Weight format alias (posit4 → nsize=4, posit5 → nsize=5). "
             "Optional — if omitted, derived from --nsize."
    )
    p.add_argument("--es_candidates", type=int, nargs="+", default=[1],
                   help="Posit es. The paper fixes es=1; several values search es per channel.")
    p.add_argument("--log2_min", type=int, default=-8)
    p.add_argument("--log2_max", type=int, default=9)

    # batching for the per-channel search
    p.add_argument("--ch_batch", type=int, default=64,
                   help="How many output channels to search at once per layer. Increase if you have headroom.")

    # activation quant (optional; off by default). Per-element, no scaling;
    # flexposit.ppl --act_quant fp8_e4m3 is the per-token variant used in the paper.
    p.add_argument("--use_act_quant", action="store_true", default=False)
    p.add_argument("--act_exp", type=int, default=4)
    p.add_argument("--act_man", type=int, default=3)

    # quantization scope
    p.add_argument("--skip_lm_head", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--quantize_embeddings", action="store_true", default=False)

    # Chunked non-overlapping WikiText-2 PPL
    p.add_argument("--ppl_seqlen", type=int, default=2048)

    # saving
    p.add_argument("--save_dir", required=True)
    p.add_argument("--save_log_name", default="quant_log.json")
    return p.parse_args()


def make_act_hook(exp_bits: int, man_bits: int):
    def hook(_m, inputs):
        x = inputs[0]
        return (float_quantize(x.float(), exp=exp_bits, man=man_bits).to(x.dtype),)
    return hook


def summarize_format_usage(ch_meta):
    """Per-channel format counts and ratios for a list of channel dicts."""
    total = len(ch_meta)
    counts = {}
    for item in ch_meta:
        fmt = item.get("selected_format", item.get("format", "unknown"))
        counts[fmt] = counts.get(fmt, 0) + 1
    ratios = {k: (v / total if total > 0 else 0.0) for k, v in counts.items()}
    return {"total_channels": total, "counts": counts, "ratios": ratios}


@torch.no_grad()
def per_channel_quantize_fixed_nsize_batched(layer_weight: torch.Tensor, nsize: int, es_cands,
                                             sweep_scales, ch_batch: int):
    """Quantize a [Cout, K] weight; return (q_w in the input dtype, per-channel meta)."""
    log2s = [int(round(math.log2(s))) for s in sweep_scales]
    res = search_channels(layer_weight.view(layer_weight.size(0), -1), nsize, es=list(es_cands),
                          log2_min=min(log2s), log2_max=max(log2s), ch_batch=ch_batch)
    meta = [
        {"channel": c, "format": f"posit{nsize}", "sqnr": sq, "log2_scale": l2, "es": e, "nsize": nsize}
        for c, (sq, l2, e) in enumerate(zip(res.sqnr.tolist(), res.log2_scale.tolist(), res.es.tolist()))
    ]
    return res.q.view_as(layer_weight).to(layer_weight.dtype), meta


def main():
    args = get_args()
    device = torch.device(args.device)
    torch_dtype = DTYPES[args.dtype]

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
    act_hook = make_act_hook(args.act_exp, args.act_man)

    target_layers = list(quantizable_layers(model, skip_lm_head=args.skip_lm_head,
                                            quantize_embeddings=args.quantize_embeddings,
                                            include_named_linear=True))

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
            # transpose in and back out so the scale search is per-output-channel.
            W_in = to_cout_first(mod, mod.weight)
            q_w, per_ch = per_channel_quantize_fixed_nsize_batched(
                W_in, nsize=args.nsize, es_cands=args.es_candidates,
                sweep_scales=sweep_scales, ch_batch=args.ch_batch
            )
            mod.weight.data = from_cout_first(mod, q_w)

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

    # Chunked non-overlapping PPL (CUDA autocast in the forward dtype)
    amp = torch_dtype if args.dtype in ("fp16", "bf16") else None
    ppl = perplexity(model, wikitext2_ids(tok), seqlen=args.ppl_seqlen, autocast_dtype=amp, progress=True)
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
