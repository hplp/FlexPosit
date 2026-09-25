# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yimin Gao
# FlexPosit: https://github.com/hplp/FlexPosit
# If you use this in academic work, please cite:
#   Y. Gao et al., "FlexPosit: Tunable Fractional Precision for LLM
#   Inference Accelerators," MICRO 2026 (to appear).

"""Bit-exact software mirror of the FlexPosit RTL numerical datapath.

Standalone (numpy only). Integer arithmetic throughout — no float shortcuts.
Every function cites the RTL module it mirrors (see ../rtl/).

Layers, in commit order:

  fp8_e4m3_fields       -> act_sign, act_zero, eff_exp, sig
  decode_native_posit   -> posit_sign, posit_zero, scale, fraction_bits[]
  product               -> zero, sign, exp, mag           (per weight)
  accumulate            -> sign, exp, m_ext (signed 16b)  (per K reduction)
  normalize             -> sign, exp (11b), man (15b)     (per ship-out lane)
  rescale               -> sign, exp (11b), man (15b)     (per output lane)
  to_fp16               -> IEEE 754 binary16 bits         (per ship-out lane)

The operator does NOT model FIFOs, streamers, controllers, output serializer,
or any control-plane behavior — only the numerical commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


ACT_MBITS = 3
MAX_POSIT_FRACTION_BITS = 5           # MAN_WIDTH in RTL (posit_decode_and_queue.v)
MAX_POSIT_ES = 2
POSIT_EXP_BIAS = 24                    # fp8_pe_mac_stream.v POSIT_EXP_BIAS
MUL_GUARD = 3                          # sub-integer guard bits below FP8 significand LSB
EXP_SUM_WIDTH = 6                      # fp8_pe_mac_stream.v EXP_SUM_WIDTH
OUT_VAL_WIDTH = ACT_MBITS + 2 + MUL_GUARD  # widened product magnitude (8 bits)
ACC_EXPW = 7
ACC_MANW = 12
GRS = 3
EXT_W = ACC_MANW + GRS                 # bfp_normalize.v: MAN_TOT
NORM_EXPW = ACC_EXPW + 4               # bfp_normalize.v: 11
NORM_MANW = ACC_MANW + GRS             # bfp_normalize.v: 15
SCALE_WIDTH = 5

# FP16 ship-out (bfp_to_fp16.v)
FP16_EXPW = 5
FP16_MANW = 10
FP16_MAX_BIASED_EXP = 30                   # 31 reserved for Inf/NaN; upstream never emits
FP16_EXP_BIAS = 15
# BIAS_DELTA = ACT_BIAS + ACT_MBITS + GRS + POSIT_EXP_BIAS - FP16_EXP_BIAS
ACT_BIAS = 7                               # 2^(ACT_EBITS-1) - 1 for FP8 E4M3
FP16_BIAS_DELTA = ACT_BIAS + ACT_MBITS + GRS + POSIT_EXP_BIAS - FP16_EXP_BIAS


# ---------------------------------------------------------------------------
# native <-> stream posit transform
# ---------------------------------------------------------------------------
def native_to_stream_code(code: int, precision: int) -> int:
    """Two's-complement the low P-1 bits when the sign bit is set."""
    if not 0 <= code < (1 << precision):
        raise ValueError(f"code {code} does not fit in {precision} bits")
    sign_mask = 1 << (precision - 1)
    if not code & sign_mask:
        return code
    low_mask = sign_mask - 1
    return sign_mask | ((-(code & low_mask)) & low_mask)


def stream_to_native_code(code: int, precision: int) -> int:
    return native_to_stream_code(code, precision)


# ---------------------------------------------------------------------------
# FP8 E4M3 activation decode (fp8_pe_mac_stream.v)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fp8Fields:
    sign: int
    zero: bool
    eff_exp: int    # 1..14 for finite normals; 1 for subnormals; 14 for reserved
    sig: int        # 4-bit unsigned (0..15). No hidden bit for subnormals.


def fp8_e4m3_fields(code: int) -> Fp8Fields:
    if not 0 <= code <= 0xFF:
        raise ValueError(f"FP8 code must fit in 8 bits, got {code}")
    sign = (code >> 7) & 1
    exp_field = (code >> ACT_MBITS) & 0xF
    man_field = code & 0x7
    if exp_field == 0 and man_field == 0:
        # Zero (act_temp = 0 anyway per src:325-327)
        return Fp8Fields(sign=sign, zero=True, eff_exp=1, sig=0)
    if exp_field == 0:
        # Subnormal: no hidden 1
        return Fp8Fields(sign=sign, zero=False, eff_exp=1, sig=man_field)
    if exp_field == 0xF:
        # Reserved encoding: saturate to max finite
        return Fp8Fields(sign=sign, zero=False, eff_exp=14, sig=15)
    return Fp8Fields(sign=sign, zero=False, eff_exp=exp_field, sig=8 + man_field)


# ---------------------------------------------------------------------------
# native Posit decode (posit_decode_and_queue.v)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PositFields:
    sign: int                       # native sign bit (code[P-1])
    zero: bool
    scale: int                      # k * 2^es + exponent
    fraction_bits: tuple[int, ...]  # up to MAX_POSIT_FRACTION_BITS, MSB-first


def decode_native_posit(code: int, precision: int, es: int) -> PositFields:
    if not 4 <= precision <= 8:
        raise ValueError(f"precision {precision} out of [4,8]")
    if not 0 <= es <= MAX_POSIT_ES:
        raise ValueError(f"es {es} out of [0,{MAX_POSIT_ES}]")
    if not 0 <= code < (1 << precision):
        raise ValueError(f"code {code} does not fit in {precision} bits")

    if code == 0:
        return PositFields(sign=0, zero=True, scale=0, fraction_bits=())

    stream_code = native_to_stream_code(code, precision)
    sign = (stream_code >> (precision - 1)) & 1

    index = precision - 2
    regime_bit = (stream_code >> index) & 1
    run = 0
    while index >= 0 and ((stream_code >> index) & 1) == regime_bit:
        run += 1
        index -= 1
    k = run - 1 if regime_bit else -run

    if index >= 0:
        index -= 1  # regime terminator consumed

    exponent = 0
    exponent_bits = 0
    while exponent_bits < es and index >= 0:
        exponent = (exponent << 1) | ((stream_code >> index) & 1)
        exponent_bits += 1
        index -= 1
    exponent <<= (es - exponent_bits)  # left-align to fill missing low bits with 0

    fraction: list[int] = []
    while index >= 0 and len(fraction) < MAX_POSIT_FRACTION_BITS:
        fraction.append((stream_code >> index) & 1)
        index -= 1

    return PositFields(
        sign=sign,
        zero=False,
        scale=k * (1 << es) + exponent,
        fraction_bits=tuple(fraction),
    )


# ---------------------------------------------------------------------------
# Serial shift-add product (fp8_pe_mac_stream.v;
#  tb/single_pe_es_tb.v)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Product:
    zero: bool
    sign: int
    exp: int    # RTL commits this as 6-bit unsigned (EXP_SUM_WIDTH)
    mag: int    # RTL commits this as 5-bit unsigned (OUT_VAL_WIDTH)


def product(act_code: int, posit_code: int, precision: int, es: int) -> Product:
    """Bit-exact reproduction of the RTL PE commit for one product."""
    act = fp8_e4m3_fields(act_code)
    p = decode_native_posit(posit_code, precision, es)

    if act.zero or p.zero:
        return Product(zero=True, sign=0, exp=0, mag=0)

    # magnitude with MUL_GUARD guard bits: the internal significand is
    # act.sig << MUL_GUARD, so right shifts by up to (MUL_GUARD + integer
    # width - 1) produce nonzero contributions. This restores per-code
    # monotonicity through P8. MAN_WIDTH cap on posit fraction storage is
    # still enforced upstream (decode_native_posit truncates to 5).
    widened = act.sig << MUL_GUARD
    mag = widened
    for shift_minus_one, bit in enumerate(p.fraction_bits):
        if bit:
            mag += widened >> (shift_minus_one + 1)

    # exponent: RTL truncates the ADDW-bit adder result to EXP_SUM_WIDTH=6
    # bits when captured.
    exp = (act.eff_exp + POSIT_EXP_BIAS + p.scale) & ((1 << EXP_SUM_WIDTH) - 1)

    # magnitude truncated to OUT_VAL_WIDTH bits (8 bits with MUL_GUARD=3).
    mag &= (1 << OUT_VAL_WIDTH) - 1

    return Product(
        zero=False,
        sign=act.sign ^ p.sign,
        exp=exp,
        mag=mag,
    )


# ---------------------------------------------------------------------------
# Block-floating accumulator (pe_bfp_acc, fp8_pe_mac_stream.v)
# ---------------------------------------------------------------------------
@dataclass
class AccumulatorState:
    sign: int = 0
    exp: int = 0                # ACC_EXPW-bit unsigned
    m_ext: int = 0              # (EXT_W+1)-bit signed  (Python signed int OK)
    zero: bool = True

    def widened_magnitude(self) -> int:
        """Return the widened magnitude view (ACC_MANW+GRS bits below sign)."""
        # Match RTL's exposed acc_man = sum_norm[EXT_W-1:0] (the low 15 bits of
        # the 16-bit two's-complement widened mantissa) plus a separate sign bit.
        return self.m_ext & ((1 << EXT_W) - 1)


def _arith_shr_rne(val: int, sh: int, ext_bits: int = EXT_W) -> int:
    """arith_shr_rne (fp8_pe_mac_stream.v).

    Signed magnitude right shift with round-to-nearest-even. `val` is a signed
    (ext_bits+1)-bit integer.
    """
    if sh <= 0:
        return val
    if val < 0:
        magnitude = -val
        val_sign = 1
    else:
        magnitude = val
        val_sign = 0

    if sh > (ext_bits + 1):
        quotient = 0
        guard = 0
        sticky = 0
    else:
        quotient = magnitude >> sh
        guard = (magnitude >> (sh - 1)) & 1
        sticky = 1 if (magnitude & ((1 << (sh - 1)) - 1)) else 0

    if guard and (sticky or (quotient & 1)):
        quotient += 1

    return -quotient if val_sign else quotient


def _accumulator_commit(state: AccumulatorState, term: Product) -> AccumulatorState:
    """one accumulator commit. A zero product leaves state unchanged."""
    if term.zero:
        return state

    # widen input term. With MUL_GUARD sub-integer bits already in mag,
    # insertion shifts by (GRS - MUL_GUARD), which is 0 when MUL_GUARD == GRS.
    in_mag_wide = term.mag << max(GRS - MUL_GUARD, 0)
    in_term_signed = -in_mag_wide if term.sign else in_mag_wide

    # alignment.
    if state.zero:
        work_exp = term.exp
        acc_aligned = 0
        in_aligned = in_term_signed
    else:
        # Signed exponent diff. Both are non-negative ACC_EXPW-bit unsigned.
        exp_diff = state.exp - term.exp
        if exp_diff >= 0:
            work_exp = state.exp
            acc_aligned = state.m_ext
            in_aligned = _arith_shr_rne(in_term_signed, exp_diff)
        else:
            work_exp = term.exp
            acc_aligned = _arith_shr_rne(state.m_ext, -exp_diff)
            in_aligned = in_term_signed

    # add with 1-bit sign extension, detect overflow of the (EXT_W+1)-bit
    # signed range, and one-bit RNE renormalize on overflow.
    sum_ext = acc_aligned + in_aligned
    # Signed (EXT_W+1)-bit representable range:
    lim = 1 << EXT_W
    if sum_ext >= lim or sum_ext < -lim:
        # Overflow: shift right by 1 with RNE, bump exponent.
        if sum_ext < 0:
            mag = -sum_ext
            neg = True
        else:
            mag = sum_ext
            neg = False
        half_q = mag >> 1
        half_round = mag & 1 and half_q & 1  # tie-to-even
        half_q += 1 if half_round else 0
        sum_norm = -half_q if neg else half_q
        work_exp = (work_exp + 1) & ((1 << ACC_EXPW) - 1)
    else:
        sum_norm = sum_ext

    if sum_norm == 0:
        return AccumulatorState(sign=0, exp=0, m_ext=0, zero=True)

    return AccumulatorState(
        sign=1 if sum_norm < 0 else 0,
        exp=work_exp & ((1 << ACC_EXPW) - 1),
        m_ext=sum_norm,
        zero=False,
    )


def accumulate(products: Iterable[Product]) -> AccumulatorState:
    """Run K products through the block-floating accumulator in order."""
    state = AccumulatorState()
    for term in products:
        state = _accumulator_commit(state, term)
    return state


# ---------------------------------------------------------------------------
# bfp_normalize (bfp_normalize.v)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizedLane:
    sign: int
    exp: int    # NORM_EXPW = 11 bits unsigned
    man: int    # NORM_MANW = 15 bits unsigned (implicit leading 1 dropped)


def normalize(state: AccumulatorState) -> NormalizedLane:
    if state.zero or state.m_ext == 0:
        return NormalizedLane(sign=0, exp=0, man=0)

    # Widened magnitude (MAN_TOT+1 = 16-bit two's complement absolute value).
    mag = abs(state.m_ext)  # RTL uses 2-bit sign extension before negate;
                            # abs() in Python handles all values including -2^15.
    if mag > ((1 << (EXT_W + 1)) - 1):
        raise AssertionError(f"magnitude {mag} exceeds normalize input width")

    # Priority encoder: highest set bit in mag[0..MAN_TOT].
    lead_pos = mag.bit_length() - 1
    if lead_pos < 0 or lead_pos > EXT_W:
        # bit_length can equal EXT_W+1 only if mag == 1 << EXT_W, which
        # occurs when acc was exactly -2^EXT_W. Clamp; RTL's priority encoder
        # scans [0..EXT_W] so it saturates at EXT_W.
        lead_pos = EXT_W

    # Left-shift so leading 1 sits at bit EXT_W; drop the implicit-1.
    shifted = mag << (EXT_W - lead_pos)
    man_norm = shifted & ((1 << EXT_W) - 1)  # low EXT_W bits (implicit-1 dropped)

    exp_norm = state.exp + lead_pos
    # exp_norm fits in NORM_EXPW=11 bits since state.exp <= 127 and lead_pos <= 15.
    if exp_norm >= (1 << NORM_EXPW):
        raise AssertionError(f"normalized exponent {exp_norm} exceeds {NORM_EXPW} bits")

    return NormalizedLane(sign=state.sign, exp=exp_norm, man=man_norm)


# ---------------------------------------------------------------------------
# bfp_pow2_rescale (bfp_pow2_rescale.v)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RescaledLane:
    sign: int
    exp: int
    man: int


def rescale(
    lane: NormalizedLane,
    scale: int,
    exp_width: int = NORM_EXPW,
    man_width: int = NORM_MANW,
    scale_width: int = SCALE_WIDTH,
) -> RescaledLane:
    """scale is a signed integer in [-2^(SCALE_WIDTH-1), 2^(SCALE_WIDTH-1)-1]."""
    scale_lim = 1 << (scale_width - 1)
    if not -scale_lim <= scale < scale_lim:
        raise ValueError(f"scale {scale} out of signed {scale_width}-bit range")

    input_zero = (lane.exp == 0 and lane.man == 0)
    scaled_sum = lane.exp + scale  # int arithmetic, matches signed extension
    max_exp = (1 << exp_width) - 1

    if input_zero or scaled_sum < 0:
        # underflow: sign cleared, exp+man zero.
        return RescaledLane(sign=0, exp=0, man=0)
    if scaled_sum > max_exp:
        # overflow: sign preserved, saturate.
        return RescaledLane(sign=lane.sign, exp=max_exp, man=(1 << man_width) - 1)

    return RescaledLane(sign=lane.sign, exp=scaled_sum, man=lane.man)


# ---------------------------------------------------------------------------
# bfp_to_fp16 (bfp_to_fp16.v)
# Convert a normalized+rescaled lane to IEEE 754 binary16 bits.
#   - Saturate to max normal on overflow (never Inf)
#   - Flush to zero on underflow (never subnormal)
#   - RNE rounding of 15-bit fraction to 10-bit
# ---------------------------------------------------------------------------
def to_fp16(lane: "RescaledLane") -> int:
    if lane.exp == 0 and lane.man == 0:
        return 0
    # Rebias the exponent from BIAS_OFFSET to FP16_EXP_BIAS.
    biased = lane.exp - FP16_BIAS_DELTA
    # RNE round the 15-bit man to 10 bits.
    round_shift = NORM_MANW - FP16_MANW  # 5
    top = (lane.man >> round_shift) & ((1 << FP16_MANW) - 1)
    round_bit = (lane.man >> (round_shift - 1)) & 1
    sticky = lane.man & ((1 << (round_shift - 1)) - 1)
    round_up = round_bit & ((1 if sticky else 0) | (top & 1))
    top_rounded = top + round_up
    if top_rounded == (1 << FP16_MANW):
        top_rounded = 0
        biased += 1
    if biased <= 0:  # underflow -> flush to zero
        return 0
    if biased >= FP16_MAX_BIASED_EXP + 1:  # overflow -> saturate to max normal
        return (lane.sign << 15) | (FP16_MAX_BIASED_EXP << FP16_MANW) | ((1 << FP16_MANW) - 1)
    return (lane.sign << 15) | (biased << FP16_MANW) | top_rounded


def decode_fp16(bits: int) -> float:
    """Decode 16-bit IEEE 754 half to Python float (matches main.decode_ship_lane)."""
    bits &= 0xFFFF
    sign = (bits >> 15) & 1
    exp = (bits >> FP16_MANW) & ((1 << FP16_EXPW) - 1)
    man = bits & ((1 << FP16_MANW) - 1)
    if exp == 0 and man == 0:
        return 0.0
    if exp == 0:
        val = (man / (1 << FP16_MANW)) * (2.0 ** (1 - FP16_EXP_BIAS))
    elif exp == 0x1F:
        val = float("inf") if man == 0 else float("nan")
    else:
        val = (1.0 + man / (1 << FP16_MANW)) * (2.0 ** (exp - FP16_EXP_BIAS))
    return -val if sign else val


# ---------------------------------------------------------------------------
# Full-pipeline convenience: one output lane from K products.
# ---------------------------------------------------------------------------
def pipeline_lane(
    products: Sequence[Product],
    scale: int,
) -> RescaledLane:
    """accumulate -> normalize -> rescale (pre-FP16 conversion)."""
    state = accumulate(products)
    lane = normalize(state)
    return rescale(lane, scale)


def pipeline_lane_fp16(
    products: Sequence[Product],
    scale: int,
) -> int:
    """accumulate -> normalize -> rescale -> to_fp16 (16-bit IEEE half bits)."""
    return to_fp16(pipeline_lane(products, scale))


# ---------------------------------------------------------------------------
# Exhaustive product-table builder
# ---------------------------------------------------------------------------
PRODUCT_DTYPE = np.dtype(
    [
        ("zero", np.uint8),
        ("sign", np.uint8),
        ("exp", np.uint8),   # 6-bit
        ("mag", np.uint8),   # 5-bit
    ]
)


def product_table(precision: int, es: int) -> np.ndarray:
    """Build the exhaustive [256, 2^P] product table for (P, es).

    Row axis = FP8 code (0..255). Column axis = native Posit code (0..2^P - 1).
    NaR position (code = 1 << (P-1)) is filled with sentinel zero (all fields 0);
    downstream analysis MUST skip that column.
    """
    if not 4 <= precision <= 8:
        raise ValueError(f"precision {precision} out of [4,8]")
    if not 0 <= es <= MAX_POSIT_ES:
        raise ValueError(f"es {es} out of [0,{MAX_POSIT_ES}]")
    nar = 1 << (precision - 1)
    table = np.zeros((256, 1 << precision), dtype=PRODUCT_DTYPE)
    for a in range(256):
        for c in range(1 << precision):
            if c == nar:
                continue
            pr = product(a, c, precision, es)
            table[a, c]["zero"] = 1 if pr.zero else 0
            table[a, c]["sign"] = pr.sign
            table[a, c]["exp"] = pr.exp
            table[a, c]["mag"] = pr.mag
    return table
