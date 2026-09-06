#!/usr/bin/env python3
# window.py — accelerated per-channel-window PPL-probe sensitivity scan
# Accelerated sensitivity scan with per-channel-window logging:
# - For each layer, iterate channel windows [start:end)
# - OOM-safe GPU search over (scale × es) per window (per channel inside the window)
# - After applying ONLY that window, run chunked non-overlapping PPL (batched) and log ΔPPL
# - Restore layer, move to next window
#
# CSV has one row per window:
#   layer, win_start, win_end, ppl_new, delta_ppl, seconds,
#   override_nsize, channel_window, eval_batch_size, l2_diff_vs_baseline

import argparse, os, json, math, time, csv, sys
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
import transformers.modeling_utils as modeling_utils

# posit quantize kernel
from qtorch_plus.quant import posit_quantize

from flexposit.quantizers.posit import MODEL_PRESETS

transformers.logging.set_verbosity_error()
EPS = 1e-8


def load_completed_windows(csv_path: str):
    """Return {(layer, win_start, win_end)} already present in an existing
    sensitivity.csv so a rerun can resume without redoing them. Empty set
    if the file doesn't exist or has no data rows.
    """
    done = set()
    if not os.path.exists(csv_path):
        return done
    try:
        with open(csv_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    done.add((row["layer"], int(row["win_start"]), int(row["win_end"])))
                except (KeyError, ValueError):
                    continue
    except Exception:
        pass
    return done


# -------------------- Args --------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(MODEL_PRESETS.keys()), default=None,
                   help="Short model name; used to auto-derive --ref_model_id from MODEL_PRESETS.")
    p.add_argument("--model_dir", required=True,
                   help="folder saved by save_pretrained() (your quantized model with dequantized weights)")
    p.add_argument("--ref_model_id", default=None,
                   help="HF id to load FP reference weights from. Optional if --model is given.")
    p.add_argument("--dtype", choices=["fp16","fp32"], default="fp16",
                   help="forward precision (loss computed in fp32)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="compute device for PPL eval and GPU search (e.g., cuda, cuda:0, cpu)")
    p.add_argument("--device_map", choices=["none","auto"], default="none",
                   help="device_map=auto for huge models; default keeps one device for reliability")
    p.add_argument("--seqlen", type=int, default=1024,
                   help="fixed chunk length (non-overlapping)")
    p.add_argument("--baseline_ppl", type=float, default=None,
                   help="If provided, skip baseline recompute.")

    # re-quantization knobs
    p.add_argument("--override_nsize", type=int, default=5,
                   help="nsize used when testing each layer")
    p.add_argument("--channel_window", type=int, default=64,
                   help="group size of channels per evaluation (per-channel if 1)")
    p.add_argument("--es_candidates", type=int, nargs="+", default=[0,1,2],
                   help="posit 'es' candidates per channel")
    p.add_argument("--log2_min", type=int, default=-8, help="min log2(scale) included")
    p.add_argument("--log2_max", type=int, default=9,  help="max log2(scale) included")

    # OOM-safe GPU tiling (new)
    p.add_argument("--max_scales_per_pass", type=int, default=8,
                   help="tile scales; smaller = lower peak memory")
    p.add_argument("--max_es_per_pass", type=int, default=2,
                   help="tile es candidates; smaller = lower peak memory")

    # scope
    p.add_argument("--skip_lm_head", action="store_true", default=True,
                   help="skip tied/output head")
    p.add_argument("--quantize_embeddings", action="store_true", default=False,
                   help="include embeddings (usually False)")

    # eval acceleration
    p.add_argument("--eval_batch_size", type=int, default=8,
                   help="# of chunks per forward for batched non-overlapping PPL")
    p.add_argument("--compile_model", action="store_true", default=False,
                   help="torch.compile(model) for extra inference speed (PyTorch 2.x)")

    # output
    p.add_argument("--out_dir", default=None,
                   help="output directory (default: <model_dir>/sensitivity_n{override}_cw{channel_window})")
    args = p.parse_args()
    if args.ref_model_id is None:
        if args.model is None:
            p.error("Either --model or --ref_model_id must be provided.")
        args.ref_model_id = MODEL_PRESETS[args.model]["hf_id"]
    return args


def get_torch_dtype(tag: str):
    return torch.float16 if tag == "fp16" else torch.float32


# -------------------- Layer selectors --------------------
def is_quant_linear(mod):
    # GPT-2 uses modeling_utils.Conv1D for linears; OPT/LLaMA use nn.Linear
    return isinstance(mod, (nn.Linear, modeling_utils.Conv1D))


def should_skip_layer(name: str, mod: nn.Module, skip_lm_head: bool, quantize_embeddings: bool):
    if skip_lm_head and (name == "lm_head" or name.endswith(".lm_head")):
        return True
    if isinstance(mod, nn.Embedding) and not quantize_embeddings:
        return True
    return False


# -------------------- OOM-safe vectorized GPU search (per window) --------------------
@torch.no_grad()
def _vectorized_search_quantize_block_tiled(
    W_block_cpu_f32: torch.Tensor,    # [Cw, K] CPU float32
    nsize: int,
    es_cands: List[int],
    sweep_scales: List[float],
    device: torch.device,             # CUDA device
    max_scales_per_pass: int,
    max_es_per_pass: int,
) -> torch.Tensor:
    """
    Memory-efficient (scale × es) search per channel on GPU:
      - No giant [Cw,S,E,K] tensor; we stream candidates in tiles.
      - Tracks best (scale, es) per channel by SQNR.
      - Returns dequantized quantized weights with same shape (CPU float32).
    """
    W = W_block_cpu_f32.to(device=device, dtype=torch.float32)   # [Cw, K]
    if W.ndim != 2:
        raise RuntimeError(f"Expect 2D weight block, got shape {tuple(W.shape)}")
    Cw, K = W.shape

    # scales / es lists
    scales = torch.tensor(sweep_scales, device=device, dtype=torch.float32)  # [S]
    S = int(scales.numel())

    max_es = max(0, nsize - 1)
    es_list = [int(e) for e in es_cands if e <= max_es] or [0]
    E = len(es_list)
    es_torch = torch.tensor(es_list, device=device, dtype=torch.int32)

    # Per-channel accumulators for best candidate
    signal_pow = (W**2).sum(dim=-1) + 1e-8               # [Cw]
    best_sqnr = torch.full((Cw,), -float("inf"), device=device)
    best_s_idx = torch.zeros((Cw,), dtype=torch.long, device=device)
    best_e_idx = torch.zeros((Cw,), dtype=torch.long, device=device)

    # Tile loops
    s_tile = max(1, int(max_scales_per_pass))
    e_tile = max(1, int(max_es_per_pass))

    for s0 in range(0, S, s_tile):
        s1 = min(s0 + s_tile, S)
        scales_tile = scales[s0:s1]                              # [St]
        # Pre-scale current tile once: [Cw, St, K]
        W_scaled = W[:, None, :] * scales_tile[None, :, None]   # correct broadcast
        W_scaled_2d = W_scaled.reshape(Cw * (s1 - s0), K)       # [Cw*St, K]

        for e0 in range(0, E, e_tile):
            e1 = min(e0 + e_tile, E)
            es_sub = es_torch[e0:e1]                            # [Et]

            # Quantize for each es in this tile
            # We reuse the pre-scaled 2D buffer to keep memory flat.
            for idx_e, e_val in enumerate(es_sub):
                q_e_2d = posit_quantize(W_scaled_2d, nsize=nsize, es=int(e_val.item()), scale=1.0)
                if q_e_2d.shape != (Cw * (s1 - s0), K):
                    if q_e_2d.numel() != Cw * (s1 - s0) * K:
                        raise RuntimeError(
                            f"posit_quantize returned shape {tuple(q_e_2d.shape)}; "
                            f"expected {(Cw*(s1-s0), K)} for flattened input"
                        )
                    q_e_2d = q_e_2d.reshape(Cw * (s1 - s0), K)

                # de-scale back to weight domain
                q_e = q_e_2d.reshape(Cw, (s1 - s0), K) / scales_tile[None, :, None]  # [Cw, St, K]

                # Compute per-channel noise for each scale in tile, pick the local best
                # noise_pow_tile: [Cw, St]
                noise_pow_tile = ((W[:, None, :] - q_e)**2).sum(dim=-1) + 1e-8
                sqnr_tile = 10.0 * torch.log10(signal_pow[:, None] / noise_pow_tile)  # [Cw, St]

                # Get argmax over St for this es
                local_best_sqnr, local_best_s_rel = torch.max(sqnr_tile, dim=1)  # [Cw], [Cw]
                # Compare to global best
                better = local_best_sqnr > best_sqnr
                if better.any():
                    best_sqnr[better] = local_best_sqnr[better]
                    # absolute scale index (within global list)
                    best_s_idx[better] = (s0 + local_best_s_rel[better])
                    best_e_idx[better] = (e0 + idx_e)

                # free temp ASAP
                del q_e, noise_pow_tile, sqnr_tile, local_best_sqnr, local_best_s_rel, better
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        # free tile buffers
        del W_scaled, W_scaled_2d, scales_tile
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Re-quantize once with chosen (scale, es) per channel
    out = torch.empty_like(W)
    for c in range(Cw):
        sc = float(scales[int(best_s_idx[c])].item())
        es = int(es_list[int(best_e_idx[c])])
        out[c] = posit_quantize(W[c] * sc, nsize=nsize, es=es, scale=1.0) / sc

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return out.detach().cpu()  # [Cw, K]


@torch.no_grad()
def quantize_window_gpu(
    layer_weight_cpu_f32: torch.Tensor,
    nsize: int,
    start: int,
    end: int,
    es_cands: List[int],
    sweep_scales: List[float],
    device: torch.device,
    max_scales_per_pass: int,
    max_es_per_pass: int,
) -> torch.Tensor:
    """
    Quantize ONLY rows [start:end) FROM FP32 reference and return a CPU float32 tensor
    where [start:end) are quantized and the rest identical to source.
    """
    W = layer_weight_cpu_f32.detach().float().cpu()
    if W.ndim != 2:
        raise RuntimeError(f"skip: non-matrix weight with shape {tuple(W.shape)}")
    Cout = W.size(0)
    flat = W.reshape(Cout, -1)

    start = max(0, min(start, Cout))
    end   = max(start, min(end, Cout))
    if start >= end:
        return W  # no-op

    block = flat[start:end]  # CPU [Cw, K]
    q_block = _vectorized_search_quantize_block_tiled(
        block, nsize=nsize, es_cands=es_cands,
        sweep_scales=sweep_scales, device=device,
        max_scales_per_pass=max_scales_per_pass, max_es_per_pass=max_es_per_pass
    )  # CPU [Cw, K]

    out = flat.clone()
    out[start:end] = q_block
    return out.view(W.shape)  # CPU float32


# -------------------- chunked non-overlapping PPL --------------------
@torch.no_grad()
def encode_wikitext2_cpu(tokenizer) -> torch.Tensor:
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt", add_special_tokens=False)
    return enc.input_ids  # CPU [1, T]

@torch.no_grad()
def eval_wikitext_batched(model, ids_cpu: torch.Tensor, seqlen: int,
                                 use_fp16_fwd: bool, batch_chunks: int,
                                 show_progress: bool = False) -> float:
    """
    Exact chunked non-overlapping PPL evaluator (batched).
    Exact non-overlapping chunked evaluator, fp32 loss accumulation.
    """
    model.eval()

    # Find device where the model lives
    try:
        dev = next(p.device for p in model.parameters() if p.device.type != "meta")
    except StopIteration:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Autocast policy
    if dev.type == "cuda":
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_fp16_fwd, dtype=torch.float16)
    else:
        class _NoOp:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        autocast_ctx = _NoOp()

    # Prepare iteration
    nsamples = ids_cpu.numel() // seqlen
    if nsamples == 0:
        raise ValueError(f"Not enough tokens for seqlen={seqlen}")
    batch_chunks = max(1, int(batch_chunks))

    iterator = range(0, nsamples, batch_chunks)
    if show_progress:
        try:
            from tqdm.auto import tqdm as _tqdm
            iterator = _tqdm(iterator, total=(nsamples + batch_chunks - 1)//batch_chunks,
                             desc=f"PPL (B={batch_chunks}, T={seqlen})", dynamic_ncols=True)
        except Exception:
            pass

    nll_sum = 0.0
    processed_tokens = 0

    for i in iterator:
        j = min(i + batch_chunks, nsamples)
        # Stack chunks -> [B, seqlen]
        batch = torch.cat(
            [ids_cpu[:, k*seqlen:(k+1)*seqlen] for k in range(i, j)],
            dim=0
        ).to(dev)

        with autocast_ctx:
            logits = model(batch).logits  # [B, T, V]

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Denominator: total-token count (not batches)
        nll_sum += loss.item() * (j - i) * seqlen
        processed_tokens += (j - i) * seqlen

        if dev.type == "cuda":
            del batch, logits, shift_logits, shift_labels, loss
            torch.cuda.empty_cache()

    ppl = math.exp(nll_sum / processed_tokens)
    return float(ppl)


# -------------------- Main --------------------
def main():
    args = get_args()
    out_dir = args.out_dir or os.path.join(
        args.model_dir, f"sensitivity_n{args.override_nsize}_cw{args.channel_window}"
    )
    os.makedirs(out_dir, exist_ok=True)

    q_dtype = get_torch_dtype(args.dtype)
    device_map = None if args.device_map == "none" else "auto"
    search_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sweep_scales = [2.0 ** k for k in range(args.log2_min, args.log2_max + 1)]

    # ----- Load eval model -----
    print(f"[Load-Eval] {args.model_dir}  (forward={args.dtype}, device_map={args.device_map})")
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=q_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True
    )
    if device_map is None and torch.cuda.is_available():
        model = model.to(search_device)
    model.eval()

    if args.compile_model:
        try:
            model = torch.compile(model)  # PyTorch 2.x
            print("[Info] torch.compile enabled.")
        except Exception as e:
            print(f"[Warn] torch.compile failed: {e}")

    # ----- Load FP reference weights -----
    print(f"[Load-Ref] {args.ref_model_id} (FP reference on CPU)")
    ref_cfg = AutoConfig.from_pretrained(args.ref_model_id, trust_remote_code=False)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model_id, config=ref_cfg, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    ref_state = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
    del ref_model

    # ----- Tokenize once -----
    ids_cpu = encode_wikitext2_cpu(tok)

    # ----- Baseline ppl (batched non-overlapping PPL) -----
    if args.baseline_ppl is None:
        print("[Info] Computing baseline PPL (batched non-overlapping PPL)...")
        baseline_ppl = eval_wikitext_batched(
            model, ids_cpu, seqlen=args.seqlen,
            use_fp16_fwd=(args.dtype == "fp16"),
            batch_chunks=args.eval_batch_size,
            show_progress=False
        )
    else:
        baseline_ppl = float(args.baseline_ppl)
    print(f"[Baseline] PPL = {baseline_ppl:.4f}")

    # ----- Collect target layers -----
    target_layers: List[Tuple[str, nn.Module]] = []
    for name, mod in model.named_modules():
        if should_skip_layer(name, mod, skip_lm_head=args.skip_lm_head, quantize_embeddings=args.quantize_embeddings):
            continue
        if is_quant_linear(mod):
            if getattr(mod, "weight", None) is not None and mod.weight.dim() == 2:
                target_layers.append((name, mod))

    print(f"Layers to test: {len(target_layers)} | override nsize={args.override_nsize} | channel_window={args.channel_window}")
    print(f"[Search Tiling] max_scales_per_pass={args.max_scales_per_pass}  max_es_per_pass={args.max_es_per_pass}  device={search_device}")

    # ----- CSV setup (resume-aware) -----
    csv_path = os.path.join(out_dir, "sensitivity.csv")
    resume_set = load_completed_windows(csv_path)
    if os.path.exists(csv_path) and len(resume_set) > 0:
        cf = open(csv_path, "a", newline="")
        cw_writer = csv.writer(cf)
        print(f"[Resume] existing CSV with {len(resume_set)} done windows -> {csv_path}",
              flush=True)
    else:
        cf = open(csv_path, "w", newline="")
        cw_writer = csv.writer(cf)
        cw_writer.writerow(["layer", "win_start", "win_end", "ppl_new", "delta_ppl", "seconds",
                            "override_nsize", "channel_window", "eval_batch_size",
                            "l2_diff_vs_baseline"])
        cf.flush()

    # ----- Iterate layers / windows -----
    results = []
    pbar = tqdm(target_layers, desc="Sensitivity", unit="layer", dynamic_ncols=True, file=sys.stdout)
    for name, mod in pbar:
        pbar.set_postfix_str(name[-48:])
        if not hasattr(mod, "weight") or mod.weight is None or mod.weight.dim() != 2:
            continue

        # Backup current (quantized) weight
        w_backup = mod.weight.detach().clone()

        # FP reference for this layer
        ref_key = name + ".weight"
        if ref_key not in ref_state:
            tqdm.write(f"[Warn] Missing FP key: {ref_key}; skipping")
            continue
        W_src_cpu = ref_state[ref_key].float().cpu()
        if W_src_cpu.dim() != 2:
            tqdm.write(f"[Warn] {ref_key} is not 2D (shape={tuple(W_src_cpu.shape)}); skipping")
            continue

        # Iterate channel windows for this layer
        Cout = int(W_src_cpu.size(0))
        cw = max(1, int(args.channel_window))
        start_idx = list(range(0, Cout, cw))
        for start in start_idx:
            end = min(start + cw, Cout)
            if (name, start, end) in resume_set:
                continue

            try:
                # Quantize only this window (from FP32 ref) using OOM-safe GPU tiling
                Q_win_cpu = quantize_window_gpu(
                    W_src_cpu,
                    nsize=args.override_nsize,
                    start=start,
                    end=end,
                    es_cands=args.es_candidates,
                    sweep_scales=sweep_scales,
                    device=search_device if search_device.type == "cuda" else torch.device("cpu"),
                    max_scales_per_pass=args.max_scales_per_pass,
                    max_es_per_pass=args.max_es_per_pass,
                )

                # Stitch: take current layer on device, replace rows [start:end)
                Q_full = w_backup.detach().float().cpu()
                Q_full[start:end] = Q_win_cpu[start:end]
                Q_dev = Q_full.to(device=mod.weight.device, dtype=mod.weight.dtype)

                # Assign temporary weight
                with torch.no_grad():
                    mod.weight.data = Q_dev
                if search_device.type == "cuda":
                    torch.cuda.synchronize()

                t0 = time.time()
                ppl_new = eval_wikitext_batched(
                    model, ids_cpu, seqlen=args.seqlen,
                    use_fp16_fwd=(args.dtype == "fp16"),
                    batch_chunks=args.eval_batch_size,
                    show_progress=False
                )
                dt = time.time() - t0

                delta = ppl_new - baseline_ppl  # negative is good (improvement)
                l2diff = torch.norm((Q_dev - w_backup.to(Q_dev.dtype)).float()).item()

                # Log to console per window
                tqdm.write(f"[Win] {name} [{start:4d}:{end:4d}): ΔPPL={delta:+.4f} ppl={ppl_new:.4f} ({dt:.1f}s)")

                # Save row
                row = [name, start, end, float(ppl_new), float(delta), float(dt),
                       int(args.override_nsize), int(args.channel_window), int(args.eval_batch_size),
                       float(l2diff)]
                cw_writer.writerow(row); cf.flush()
                results.append({
                    "layer": name, "win_start": start, "win_end": end,
                    "ppl_new": float(ppl_new), "delta_ppl": float(delta),
                    "seconds": float(dt),
                    "override_nsize": int(args.override_nsize),
                    "channel_window": int(args.channel_window),
                    "eval_batch_size": int(args.eval_batch_size),
                    "l2_diff_vs_baseline": float(l2diff),
                })

            except Exception as e:
                tqdm.write(f"[Warn] {name} win[{start}:{end}] failed: {e}")
            finally:
                # Restore original layer before next window
                with torch.no_grad():
                    mod.weight.data = w_backup
                if search_device.type == "cuda":
                    torch.cuda.empty_cache()

    cf.close()

    # ----- Save JSON summary -----
    json_path = os.path.join(out_dir, "sensitivity.json")
    with open(json_path, "w") as f:
        json.dump({
            "model_dir": args.model_dir,
            "ref_model_id": args.ref_model_id,
            "baseline_ppl": baseline_ppl,
            "seqlen": args.seqlen,
            "override_nsize": args.override_nsize,
            "channel_window": args.channel_window,
            "eval_batch_size": args.eval_batch_size,
            "log2_range": [args.log2_min, args.log2_max],
            "es_candidates": args.es_candidates,
            "max_scales_per_pass": args.max_scales_per_pass,
            "max_es_per_pass": args.max_es_per_pass,
            "results_len": len(results)
        }, f, indent=2)

    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
