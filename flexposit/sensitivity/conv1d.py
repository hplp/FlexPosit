#!/usr/bin/env python3
# conv1d.py — Conv1D per-Cout wrapper around sensitivity.window.
#
# Wrapper around sensitivity/window.py that transposes HF Conv1D
# weights before applying the window scan, so that windows walk along Cout
# (not Cin). nn.Linear layers are unchanged.
#
# Requirements:
#   * The base checkpoint (--model_dir) must already be quantized on the
#     per-Cout axis (see gate0_partA/per_cout/quantize_posit41_per_cout.py).
#   * FP reference (--ref_model_id) is loaded from HF and consulted layer-wise;
#     for Conv1D we transpose it before window search and transpose back at
#     the write step so mod.weight retains its native (Cin, Cout) storage.
#
# CSV format is identical to the upstream script; win_start/win_end now index
# the Cout axis on Conv1D layers.

import argparse, os, json, math, time, csv, sys
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import transformers.modeling_utils as modeling_utils

# Reuse the sibling channel-window PPL-probe generator.
from . import window as sens_up  # noqa: E402
from flexposit.quantizers.posit import MODEL_PRESETS


def load_completed_windows(csv_path: str):
    """Return a set of (layer, win_start, win_end) already present in an
    existing sensitivity.csv, so a rerun can resume where it left off.
    Empty set if the file doesn't exist.
    """
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add((row["layer"], int(row["win_start"]), int(row["win_end"])))
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass
    return done

transformers.logging.set_verbosity_error()
EPS = 1e-8


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default=None,
                   help="Short model name; used to auto-derive --ref_model_id from MODEL_PRESETS.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--ref_model_id", default=None,
                   help="HF id for FP reference. Optional if --model is given.")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device_map", choices=["none", "auto"], default="none")
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--baseline_ppl", type=float, default=None)
    p.add_argument("--override_nsize", type=int, default=5)
    p.add_argument("--channel_window", type=int, default=256)
    p.add_argument("--es_candidates", type=int, nargs="+", default=[1])
    p.add_argument("--log2_min", type=int, default=-8)
    p.add_argument("--log2_max", type=int, default=9)
    p.add_argument("--max_scales_per_pass", type=int, default=8)
    p.add_argument("--max_es_per_pass", type=int, default=2)
    p.add_argument("--skip_lm_head", action="store_true", default=True)
    p.add_argument("--quantize_embeddings", action="store_true", default=False)
    p.add_argument("--batch_chunks", type=int, default=8)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--attn", choices=["auto", "fa2", "sdpa"], default="sdpa")
    args = p.parse_args()
    if args.ref_model_id is None:
        if args.model is None:
            p.error("Either --model or --ref_model_id must be provided.")
        args.ref_model_id = MODEL_PRESETS[args.model]["hf_id"]
    return args


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    q_dtype = sens_up.get_torch_dtype(args.dtype)
    device_map = None if args.device_map == "none" else "auto"
    search_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sweep_scales = [2.0 ** k for k in range(args.log2_min, args.log2_max + 1)]

    print(f"[Load-Eval] {args.model_dir}  dtype={args.dtype}  attn={args.attn}", flush=True)
    try:
        tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    except Exception as e:
        print(f"[Warn] tokenizer not in model_dir; loading from ref: {e}")
        tok = AutoTokenizer.from_pretrained(args.ref_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(torch_dtype=q_dtype, device_map=device_map, low_cpu_mem_usage=True)
    if args.attn == "sdpa":
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **model_kwargs)
    if device_map is None and torch.cuda.is_available():
        model = model.to(search_device)
    model.eval()

    print(f"[Load-Ref] {args.ref_model_id} (FP32 on CPU)", flush=True)
    ref_cfg = AutoConfig.from_pretrained(args.ref_model_id, trust_remote_code=False)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_id, config=ref_cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    ref_state = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
    del ref_model

    print(f"[Data] Encoding wikitext2 seqlen={args.seqlen}", flush=True)
    ids_cpu = sens_up.encode_wikitext2_cpu(tok).pin_memory()

    if args.baseline_ppl is None:
        print("[Info] Computing baseline PPL (chunked batched)...", flush=True)
        baseline_ppl = sens_up.eval_wikitext_batched(
            model, ids_cpu, seqlen=args.seqlen,
            use_fp16_fwd=(args.dtype == "fp16"),
            batch_chunks=args.batch_chunks,
        )
    else:
        baseline_ppl = float(args.baseline_ppl)
    print(f"[Baseline] PPL = {baseline_ppl:.4f}", flush=True)

    target_layers: List[Tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if sens_up.should_skip_layer(name, mod, args.skip_lm_head, args.quantize_embeddings):
            continue
        if sens_up.is_quant_linear(mod):
            if getattr(mod, "weight", None) is not None and mod.weight.dim() == 2:
                target_layers.append((name, mod))
    n_conv1d = sum(1 for _, m in target_layers if isinstance(m, modeling_utils.Conv1D))
    n_lin = len(target_layers) - n_conv1d
    print(f"Layers to test: {len(target_layers)} (Conv1D={n_conv1d}, Linear={n_lin}) | "
          f"nsize={args.override_nsize} | cw={args.channel_window} | axis=per-Cout", flush=True)

    csv_path = os.path.join(args.out_dir, "sensitivity.csv")
    header = ["layer", "win_start", "win_end", "ppl_new", "delta_ppl", "seconds",
              "override_nsize", "channel_window", "eval_batch_size", "l2_diff_vs_baseline"]
    resume_set = load_completed_windows(csv_path)
    if os.path.exists(csv_path) and len(resume_set) > 0:
        cf = open(csv_path, "a", newline="")
        cw_writer = csv.writer(cf)
        print(f"[Resume] existing CSV with {len(resume_set)} done windows -> {csv_path}", flush=True)
    else:
        cf = open(csv_path, "w", newline="")
        cw_writer = csv.writer(cf)
        cw_writer.writerow(header); cf.flush()

    results = []
    pbar = tqdm(target_layers, desc="Sensitivity", unit="layer", dynamic_ncols=True, file=sys.stdout)
    for name, mod in pbar:
        pbar.set_postfix_str(name[-48:])
        if not hasattr(mod, "weight") or mod.weight is None or mod.weight.dim() != 2:
            continue

        w_backup = mod.weight.detach().clone()   # stored layout (Cin, Cout) for Conv1D
        ref_key = name + ".weight"
        if ref_key not in ref_state:
            tqdm.write(f"[Warn] missing FP key {ref_key}, skipping")
            continue
        W_src_raw = ref_state[ref_key].float().cpu()
        if W_src_raw.dim() != 2:
            continue

        # Per-Cout orientation for the window search:
        is_conv1d = isinstance(mod, modeling_utils.Conv1D)
        if is_conv1d:
            W_src_effective = W_src_raw.transpose(0, 1).contiguous()   # (Cout, Cin)
        else:
            W_src_effective = W_src_raw                                # (Cout, Cin) already

        Cout = int(W_src_effective.size(0))
        cw = max(1, int(args.channel_window))
        start_idx = list(range(0, Cout, cw))

        for start in start_idx:
            end = min(start + cw, Cout)
            if (name, start, end) in resume_set:
                continue

            try:
                Q_win_cpu = sens_up.quantize_window_gpu(
                    W_src_effective,
                    nsize=args.override_nsize, start=start, end=end,
                    es_cands=args.es_candidates, sweep_scales=sweep_scales,
                    device=search_device if search_device.type == "cuda" else torch.device("cpu"),
                    max_scales_per_pass=args.max_scales_per_pass,
                    max_es_per_pass=args.max_es_per_pass,
                )  # (Cout, Cin); rows [start:end) are quantized, other rows == W_src_effective

                # Extract only the quantized rows and write them back into mod.weight
                # in the correct orientation.
                Q_rows_cout = Q_win_cpu[start:end]   # (window, Cin)

                if is_conv1d:
                    # Stored weight is (Cin, Cout); Q_rows_cout corresponds to
                    # Cout indices [start:end). Transpose to (Cin, window) and
                    # write into columns [start:end) of the stored weight.
                    Q_cols_cin = Q_rows_cout.transpose(0, 1).contiguous()  # (Cin, window)
                    Q_dev = w_backup.clone()
                    Q_dev[:, start:end] = Q_cols_cin.to(device=Q_dev.device, dtype=Q_dev.dtype,
                                                        non_blocking=True)
                else:
                    Q_dev = w_backup.clone()
                    Q_dev[start:end] = Q_rows_cout.to(device=Q_dev.device, dtype=Q_dev.dtype,
                                                      non_blocking=True)

                with torch.no_grad():
                    mod.weight.data = Q_dev
                if search_device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.time()
                ppl_new = sens_up.eval_wikitext_batched(
                    model, ids_cpu, seqlen=args.seqlen,
                    use_fp16_fwd=(args.dtype == "fp16"),
                    batch_chunks=args.batch_chunks,
                )
                dt = time.time() - t0
                delta = ppl_new - baseline_ppl
                l2diff = torch.norm((Q_dev - w_backup.to(Q_dev.dtype)).float()).item()

                tqdm.write(f"[Win] {name} [{start:4d}:{end:4d}) ΔPPL={delta:+.4f} "
                           f"ppl={ppl_new:.4f} ({dt:.1f}s)")
                row = [name, start, end, float(ppl_new), float(delta), float(dt),
                       int(args.override_nsize), int(args.channel_window),
                       int(args.batch_chunks), float(l2diff)]
                cw_writer.writerow(row); cf.flush()
                results.append({"layer": name, "win_start": start, "win_end": end,
                                "ppl_new": float(ppl_new), "delta_ppl": float(delta)})
            except Exception as e:
                tqdm.write(f"[Warn] {name} win[{start}:{end}) failed: {e}")
            finally:
                with torch.no_grad():
                    mod.weight.data = w_backup
                if search_device.type == "cuda":
                    torch.cuda.empty_cache()

    cf.close()

    json_path = os.path.join(args.out_dir, "sensitivity.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "ref_model_id": args.ref_model_id,
            "baseline_ppl": baseline_ppl,
            "seqlen": args.seqlen,
            "override_nsize": args.override_nsize,
            "channel_window": args.channel_window,
            "batch_chunks": args.batch_chunks,
            "log2_range": [args.log2_min, args.log2_max],
            "es_candidates": args.es_candidates,
            "results_len": len(results),
            "dtype": args.dtype,
            "axis": "per_out_channel",
            "notes": "Conv1D FP32 reference weight transposed before window search; "
                     "quantized rows transposed back before writing to mod.weight.",
        }, f, indent=2)
    print(f"[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
