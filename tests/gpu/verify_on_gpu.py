#!/usr/bin/env python3
"""GPU release checks for flexposit (run on a CUDA machine; not part of pytest).

  1. formats: flexposit.formats vs qtorch_plus's CUDA kernels on ALL 2**32
     float32 bit patterns (needs `pip install qtorch_plus==0.2.0` and nvcc).
  2. ppl: WikiText-2 perplexity of one model through
       (a) the refactored CLI quantizer at Posit(4,1), and
       (b) the new API, flexposit.quantize(bits=...),
     compared with the MICRO 2026 Table 2 numbers.

    python tests/gpu/verify_on_gpu.py formats
    python tests/gpu/verify_on_gpu.py ppl --model mistral-7b
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]

# From FlexPosit_artifact/expected/table2_headline_ppl.csv (WikiText-2, seqlen 2048).
TABLE2 = {
    "llama-2-7b": {4.0: 6.51, 4.4: 5.80, 5.0: 5.69},
    "mistral-7b": {4.0: 6.12, 4.4: 5.48, 5.0: 5.43},
    "phi-2": {4.0: 12.19, 4.4: 10.51, 5.0: 10.36},
}


def check_formats(chunk_log2: int = 26) -> bool:
    from qtorch_plus.quant import float_quantize as q_float, posit_quantize as q_posit
    from flexposit.formats import _float_quantize_torch, _posit_quantize_torch

    configs = [("posit", n, es) for n in (4, 5, 6, 7, 8) for es in (0, 1, 2)] + [("float", 4, 3), ("float", 5, 2)]
    chunk = 1 << chunk_log2
    ok = True
    for kind, a, b in configs:
        t0, bad = time.time(), 0
        for start in range(0, 1 << 32, chunk):
            x = torch.arange(start, start + chunk, dtype=torch.int64, device="cuda")
            x = torch.where(x >= 1 << 31, x - (1 << 32), x).to(torch.int32).view(torch.float32)
            if kind == "posit":
                ours, ref = _posit_quantize_torch(x, a, b, 1.0), q_posit(x, nsize=a, es=b)
            else:
                ours, ref = _float_quantize_torch(x, a, b), q_float(x, exp=a, man=b, rounding="nearest")
            same = (ours.view(torch.int32) == ref.view(torch.int32)) | (ours.isnan() & ref.isnan())
            bad += int((~same).sum())
        ok &= bad == 0
        print(f"{kind}({a},{b}): {'OK' if bad == 0 else f'{bad} MISMATCHES'} over 2^32 inputs "
              f"[{time.time() - t0:.0f}s]", flush=True)
    return ok


def check_ppl(model_name: str, bits: float, tol: float) -> bool:
    import flexposit

    expected = TABLE2.get(model_name, {})
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        print(f"[cli] flexposit.quantizers.posit --model {model_name} --nsize 4", flush=True)
        out = subprocess.run(
            [sys.executable, "-m", "flexposit.quantizers.posit", "--model", model_name, "--nsize", "4",
             "--dtype", "fp16", "--device", "cuda", "--save_dir", tmp],
            check=True, capture_output=True, text=True, cwd=REPO).stdout
        line = next(l for l in out.splitlines() if l.startswith("Perplexity"))
        ppl_cli = float(line.rsplit(":", 1)[1])
        ok &= _report("CLI Posit(4,1)", ppl_cli, expected.get(4.0), tol)

    model, tok = flexposit.load_model(model_name, dtype="fp16", device="cuda")
    sens = REPO / "data" / "sensitivity" / f"{model_name}.csv"
    state = flexposit.quantize(model, flexposit.FlexPositConfig(bits=bits), sensitivity=str(sens))
    print(f"[api] {state.summary()}", flush=True)
    ppl_api = flexposit.wikitext2_perplexity(model, tok, seqlen=2048, autocast_dtype=torch.float16)
    ok &= _report(f"API bits={bits}", ppl_api, expected.get(bits), tol)
    return ok


def _report(what: str, got: float, want, tol: float) -> bool:
    if want is None:
        print(f"{what}: ppl {got:.4f} (no paper number to compare)")
        return True
    good = abs(got - want) <= tol
    print(f"{what}: ppl {got:.4f} vs paper {want:.2f} -> {'OK' if good else 'DIFFERS'} (tol {tol})")
    return good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("formats")
    p = sub.add_parser("ppl")
    p.add_argument("--model", default="mistral-7b")
    p.add_argument("--bits", type=float, default=4.4)
    p.add_argument("--tol", type=float, default=0.02, help="paper numbers are rounded to 0.01")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "run this on a CUDA machine"
    ok = check_formats() if args.cmd == "formats" else check_ppl(args.model, args.bits, args.tol)
    print("ALL OK" if ok else "SOME CHECKS DIFFER")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
