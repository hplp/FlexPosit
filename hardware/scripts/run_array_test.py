#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yimin Gao
# FlexPosit: https://github.com/hplp/FlexPosit
# If you use this in academic work, please cite:
#   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
#   Inference Accelerators," MICRO 2026 (to appear).

"""Random GEMM tiles through the FlexPosit array, checked bit-exactly.

For each (N, K, P, es) configuration this script draws random FP8 E4M3
activations, random Posit(P, es) weights and random per-column power-of-two
scales, runs tb/array_tb.v under Icarus Verilog, and compares every FP16
output lane against model/flexposit_rtl_exact.py.

    python3 scripts/run_array_test.py                 # default sweep
    python3 scripts/run_array_test.py --n 8 --k 32 --p 6 --es 2
"""

from __future__ import annotations

import argparse
import random
import re
import subprocess
import sys
from pathlib import Path

HW = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW / "model"))

import flexposit_rtl_exact as ref  # noqa: E402

RTL = [
    "mm_efficient.v",
    "systolic_efficient.v",
    "posit_decode_and_queue.v",
    "fp8_pe_mac_stream.v",
    "bfp_normalize.v",
    "bfp_pow2_rescale.v",
    "bfp_to_fp16.v",
    "fifo.v",
    "act_fifo.v",
]

# Every precision/es pair on a 4x4 array, plus two larger tiles.
DEFAULT_CONFIGS = [(4, 16, p, es) for p in range(4, 9) for es in range(3)] + [
    (8, 32, 5, 1),
    (16, 64, 4, 1),
]

LANE_RE = re.compile(r"LANE (\d+) (\d+) ([0-9a-fA-F]{4})")


def write_mem(path: Path, values: list[int], digits: int) -> None:
    path.write_text("".join(f"{v:0{digits}x}\n" for v in values), encoding="ascii")


def run_config(n: int, k: int, p: int, es: int, rng: random.Random) -> int:
    """Run one tile; return the number of mismatching lanes."""
    nar = 1 << (p - 1)
    posit_codes = [c for c in range(1 << p) if c != nar]
    acts = [[rng.randrange(256) for _ in range(k)] for _ in range(n)]
    weights = [[rng.choice(posit_codes) for _ in range(k)] for _ in range(n)]
    scales = [rng.randint(-8, 7) for _ in range(n)]

    build = HW / "build"
    build.mkdir(exist_ok=True)
    write_mem(build / "act.mem", [a for row in acts for a in row], 2)
    w_bits = []
    for col in weights:
        for code in col:
            stream = ref.native_to_stream_code(code, p)
            w_bits += [(stream >> b) & 1 for b in range(p - 1, -1, -1)]
    write_mem(build / "w.mem", w_bits, 1)
    write_mem(build / "scale.mem", [s & 0x1F for s in scales], 2)

    sim = build / f"array_tb_n{n}_p{p}"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(sim), "-s", "array_tb",
         f"-Parray_tb.N={n}", f"-Parray_tb.K={k}",
         f"-Parray_tb.P={p}", f"-Parray_tb.ES={es}",
         str(HW / "tb" / "array_tb.v"), *[str(HW / "rtl" / f) for f in RTL]],
        check=True, cwd=HW, capture_output=True, text=True,
    )  # iverilog warns about unused out-of-range branches of j==0 ? ... : [j-1]
    out = subprocess.run(["vvp", "-n", str(sim)], check=True, cwd=HW,
                         capture_output=True, text=True).stdout

    got = {(int(i), int(j)): int(h, 16) for i, j, h in LANE_RE.findall(out)}
    if len(got) != n * n:
        print(f"  expected {n * n} lanes, got {len(got)}")
        return n * n

    bad = 0
    for i in range(n):
        for j in range(n):
            products = [ref.product(acts[i][kk], weights[j][kk], p, es) for kk in range(k)]
            want = ref.pipeline_lane_fp16(products, scales[j])
            if got[(i, j)] != want:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH out[{i}][{j}]: rtl={got[(i, j)]:04x} model={want:04x}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, help="array size N (N x N PEs)")
    ap.add_argument("--k", type=int, help="reduction length K (<= 128)")
    ap.add_argument("--p", type=int, choices=range(4, 9), help="posit precision")
    ap.add_argument("--es", type=int, choices=range(3), help="posit es")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    single = (args.n, args.k, args.p, args.es)
    if any(v is not None for v in single):
        if any(v is None for v in single):
            ap.error("--n, --k, --p and --es must be given together")
        configs = [single]
    else:
        configs = DEFAULT_CONFIGS

    rng = random.Random(args.seed)
    failed = 0
    lanes = 0
    for n, k, p, es in configs:
        bad = run_config(n, k, p, es, rng)
        lanes += n * n
        status = "PASS" if bad == 0 else f"FAIL ({bad} lanes)"
        print(f"N={n:<3} K={k:<4} P={p} es={es}: {status}")
        failed += bad != 0

    if failed:
        print(f"{failed}/{len(configs)} configurations FAILED")
        return 1
    print(f"PASS: {len(configs)} configurations, {lanes} FP16 lanes bit-exact against the model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
