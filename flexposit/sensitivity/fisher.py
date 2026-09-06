#!/usr/bin/env python3
"""Generate Fisher-style per-channel-window sensitivity CSV in FlexPosit format.

For each channel-window (row range of a linear's weight), compute aggregate
Fisher-weighted quantization perturbation reduction:

    score(window) = Σ_{i ∈ window-rows} F_ii · ((w_i - Q_{b_low}(w_i))^2
                                                - (w_i - Q_{b_high}(w_i))^2)

F_ii = (1/N) Σ_n (∂L_n/∂w_i)^2 estimated on the FP reference model with
WikiText-2 train calibration.

Two ways to specify the window layout:
  * --input_sens_csv <path> : reuse the (layer, win_start, win_end) layout
    from an existing FlexPosit sensitivity CSV. Guarantees apples-to-apples
    comparison with a PPL-probe run on the same layout.
  * --channel_window <N>    : enumerate the layout from the model itself
    (nn.Linear layers only — Conv1D is skipped with a warning). Standalone.

The `delta_ppl` column of the output CSV is `-score(window)` so ASCENDING
sort (which the MPQ apply script uses) picks the highest-Fisher windows first.
"""
import argparse, csv, gc, math, os, sys, time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import transformers.modeling_utils as modeling_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse Fisher helpers from the layer-MPQ driver.
from flexposit.mpq.layer import (
    is_quant_linear, calib_chunks, compute_fisher_diagonal,
    quantize_pc_posit,
)
from flexposit.quantizers.posit import MODEL_PRESETS


DEFAULT_HEADER = ["layer", "win_start", "win_end", "ppl_new", "delta_ppl",
                  "seconds", "override_nsize", "channel_window",
                  "eval_batch_size", "l2_diff_vs_baseline"]


def read_windows(sens_csv):
    """Return list of dict rows (preserving original order)."""
    rows = []
    with open(sens_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def enumerate_windows(model, channel_window, override_nsize, skip_lm_head=True):
    """Auto-generate (layer, win_start, win_end) layout by walking the model.

    nn.Linear layers only — Conv1D is skipped with a warning (the per-row
    Fisher-score aggregation below assumes weights are stored as (Cout, Cin);
    Conv1D is (Cin, Cout) and would need the same transpose treatment as
    posit.py's _to_cout_first).
    """
    windows = []
    n_conv1d_skipped = 0
    for name, mod in model.named_modules():
        if isinstance(mod, modeling_utils.Conv1D):
            n_conv1d_skipped += 1
            continue
        if not is_quant_linear(mod):
            continue
        if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
            continue
        if not hasattr(mod, "weight") or mod.weight is None or mod.weight.dim() != 2:
            continue
        Cout = mod.weight.shape[0]     # nn.Linear stores (Cout, Cin)
        for start in range(0, Cout, channel_window):
            end = min(start + channel_window, Cout)
            windows.append({
                "layer": name,
                "win_start": str(start),
                "win_end": str(end),
                "ppl_new": "",
                "delta_ppl": "0",
                "seconds": "",
                "override_nsize": str(override_nsize),
                "channel_window": str(channel_window),
                "eval_batch_size": "",
                "l2_diff_vs_baseline": "",
            })
    if n_conv1d_skipped > 0:
        print(f"[warn] skipped {n_conv1d_skipped} Conv1D layers — Fisher standalone "
              f"path is nn.Linear-only. For GPT-2, pass --input_sens_csv from a "
              f"prior flexposit.sensitivity.conv1d run.", flush=True)
    return windows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default=None,
                   help="Short model name; used to auto-derive --ref_id from MODEL_PRESETS.")
    p.add_argument("--ref_id", default=None,
                   help="HF id for FP reference. Optional if --model is given.")
    p.add_argument("--input_sens_csv", default=None,
                   help="Existing FlexPosit sensitivity CSV to reuse the window "
                        "layout from. If omitted, --channel_window is required "
                        "and the layout is enumerated from the model.")
    p.add_argument("--channel_window", type=int, default=None,
                   help="Window size for standalone layout enumeration "
                        "(used when --input_sens_csv is not given).")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--b_low", type=int, default=4)
    p.add_argument("--b_high", type=int, default=5)
    p.add_argument("--es", type=int, default=1)
    p.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    p.add_argument("--fisher_seqlen", type=int, default=512)
    p.add_argument("--n_calib", type=int, default=128)
    p.add_argument("--skip_lm_head", action="store_true", default=True)
    args = p.parse_args()

    if args.ref_id is None:
        if args.model is None:
            p.error("Either --model or --ref_id must be provided.")
        args.ref_id = MODEL_PRESETS[args.model]["hf_id"]

    if args.input_sens_csv is None and args.channel_window is None:
        p.error("Either --input_sens_csv or --channel_window must be provided")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    print(f"[args] {vars(args)}", flush=True)

    td = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.ref_id, use_fast=True)

    print(f"[Phase 1] Load FP reference {args.ref_id}", flush=True)
    model_fp = AutoModelForCausalLM.from_pretrained(
        args.ref_id, torch_dtype=td, device_map="auto", low_cpu_mem_usage=True,
    )

    # ---- Window layout ---------------------------------------------------
    if args.input_sens_csv is not None:
        rows = read_windows(args.input_sens_csv)
        print(f"[Layout] {len(rows)} windows from {args.input_sens_csv}", flush=True)
        with open(args.input_sens_csv, "r", newline="") as f:
            header = next(csv.reader(f))
    else:
        # override_nsize used only for the metadata column in the output CSV.
        rows = enumerate_windows(model_fp, args.channel_window,
                                 override_nsize=args.b_high,
                                 skip_lm_head=args.skip_lm_head)
        print(f"[Layout] {len(rows)} windows enumerated from model "
              f"(cw={args.channel_window})", flush=True)
        header = DEFAULT_HEADER

    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r["layer"]].append(r)
    print(f"[Layout] {len(by_layer)} unique linears with windows", flush=True)

    # ---- Fisher diagonal -------------------------------------------------
    print(f"[Fisher] n_calib={args.n_calib} fisher_seqlen={args.fisher_seqlen}", flush=True)
    t0 = time.time()
    fisher_diag = compute_fisher_diagonal(model_fp, tok, args.fisher_seqlen, args.n_calib)
    print(f"[Fisher] done ({time.time()-t0:.1f}s)", flush=True)

    print("[Snapshot ref weights -> CPU FP16]", flush=True)
    ref_sd = {n: p.detach().cpu().to(torch.float16)
              for n, p in model_fp.named_parameters() if n.endswith(".weight")}
    del model_fp
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Per-window Fisher score aggregation -----------------------------
    print(f"[Score] aggregating per-window Fisher · Δw² ...", flush=True)
    t0 = time.time()
    out_rows = []
    for li, layer_key in enumerate(sorted(by_layer.keys())):
        wkey = layer_key + ".weight"
        if wkey not in ref_sd or wkey not in fisher_diag:
            print(f"  [warn] missing {wkey}; emitting score=0 for its windows", flush=True)
            for row in by_layer[layer_key]:
                row2 = dict(row)
                row2["delta_ppl"] = "0.0"
                out_rows.append(row2)
            continue
        w_fp = ref_sd[wkey].float()
        w_low = quantize_pc_posit(w_fp, nsize=args.b_low, es=args.es, device="cuda")
        w_high = quantize_pc_posit(w_fp, nsize=args.b_high, es=args.es, device="cuda")
        per_row_score = (
            fisher_diag[wkey] * ((w_fp - w_low) ** 2 - (w_fp - w_high) ** 2)
        ).sum(dim=-1)  # [Cout]  (assumes nn.Linear storage)
        del w_low, w_high
        for row in by_layer[layer_key]:
            ws = int(row["win_start"])
            we = int(row["win_end"])
            score = float(per_row_score[ws:we].sum().item())
            row2 = dict(row)
            # mpq.channel_window sorts ascending by delta_ppl → most negative first.
            # High Fisher score = beneficial -> store delta_ppl = -score so it
            # ranks first.
            row2["delta_ppl"] = f"{-score:.6e}"
            row2["ppl_new"] = ""
            row2["seconds"] = ""
            row2["l2_diff_vs_baseline"] = ""
            out_rows.append(row2)
        if (li + 1) % 16 == 0 or li + 1 == len(by_layer):
            print(f"  [{li+1}/{len(by_layer)}] {layer_key}", flush=True)
    print(f"[Score] done ({time.time()-t0:.1f}s)", flush=True)

    # ---- Write CSV -------------------------------------------------------
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in header})
    print(f"[Saved] {args.out_csv}  ({len(out_rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
