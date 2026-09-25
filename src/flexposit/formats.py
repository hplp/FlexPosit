"""Posit and small-float quantizers in pure PyTorch.

Drop-in replacements for ``qtorch_plus.quant.posit_quantize`` and
``float_quantize(rounding="nearest")``. They run on any device PyTorch runs
on (CPU, CUDA, MPS), need no compiled extension, and are bit-identical to
qtorch_plus 0.2.0, including its edge cases:

* posit: round-to-nearest-even on the regime/exponent/fraction bit string;
  nonzero values at or below minpos snap to minpos and values at or above
  maxpos saturate to maxpos (no underflow to zero); +-0 -> +0; +-inf and NaN
  -> -inf.
* float: round-half-up on the float32 bit pattern, no subnormals (values
  below the smallest normal snap to it or to zero), saturation at the largest
  finite value (240 for E4M3).

Set ``FLEXPOSIT_BACKEND=qtorch`` to route both functions to qtorch_plus.
tests/test_formats.py checks the equivalence.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

__all__ = ["posit_quantize", "float_quantize", "posit_values", "posit_value_table"]

_BACKEND = os.environ.get("FLEXPOSIT_BACKEND", "torch").lower()
if _BACKEND not in ("torch", "qtorch"):
    raise ValueError(f"FLEXPOSIT_BACKEND must be 'torch' or 'qtorch', got {_BACKEND!r}")

_F32_ABS = 0x7FFFFFFF
_F32_INF = 0x7F800000
_F32_FRAC_MASK = 0x007FFFFF


def _check_posit(nsize: int, es: int) -> None:
    if not 2 <= nsize <= 16:
        raise ValueError(f"nsize must be in [2, 16], got {nsize}")
    if not 0 <= es <= nsize - 1:
        raise ValueError(f"es must be in [0, nsize-1], got es={es} for nsize={nsize}")


def _decode_positive_code(code: int, nsize: int, es: int) -> float:
    """Value of a positive posit code (1 <= code < 2**(nsize-1))."""
    bits = [(code >> i) & 1 for i in range(nsize - 2, -1, -1)]
    first = bits[0]
    run = 1
    while run < len(bits) and bits[run] == first:
        run += 1
    k = run - 1 if first else -run
    rest = bits[run + 1:]
    e_bits = rest[:es] + [0] * (es - len(rest[:es]))
    e = int("".join(map(str, e_bits)) or "0", 2)
    f_bits = rest[es:]
    frac = int("".join(map(str, f_bits)) or "0", 2) / (1 << len(f_bits)) if f_bits else 0.0
    return 2.0 ** (k * (1 << es) + e) * (1.0 + frac)


@lru_cache(maxsize=None)
def posit_values(nsize: int, es: int) -> tuple[float, ...]:
    """Values of positive posit codes 0 .. 2**(nsize-1)-1 (code 0 is zero)."""
    _check_posit(nsize, es)
    return (0.0,) + tuple(_decode_positive_code(c, nsize, es) for c in range(1, 1 << (nsize - 1)))


def posit_value_table(nsize: int, es: int, device=None) -> torch.Tensor:
    """float32 tensor of ``posit_values(nsize, es)``."""
    return torch.tensor(posit_values(nsize, es), dtype=torch.float32, device=device)


def _posit_round_magnitude(bits: torch.Tensor, nsize: int, es: int) -> torch.Tensor:
    """Positive posit code (int64) for float32 bit patterns strictly inside (minpos, maxpos).

    ``bits`` holds |x| as int64 float32 bit patterns. Entries outside the
    open interval give garbage; the caller masks them.
    """
    n1 = nsize - 1
    useed_zeros = 1 << es
    E = (bits >> 23) - 127
    frac = bits & _F32_FRAC_MASK
    k = torch.div(E, useed_zeros, rounding_mode="floor")
    e = E - k * useed_zeros
    # Clamp k to the representable regime range so masked-out entries cannot
    # produce oversized shifts; in-range entries already satisfy this.
    k = k.clamp(-(nsize - 2), nsize - 3)
    pos = k >= 0
    regime = torch.where(pos, ((1 << (k + 1).clamp(min=0)) - 1) << 1, torch.ones_like(k))
    regime_len = torch.where(pos, k + 2, 1 - k)
    total_len = regime_len + es + 23
    s = (regime << (es + 23)) | (e << 23) | frac
    drop = total_len - n1  # >= 1 because total_len >= 24 > n1
    p = s >> drop
    half = (s >> (drop - 1)) & 1
    sticky = (s & ((1 << (drop - 1)) - 1)) != 0
    return p + (half & ((p & 1) | sticky.to(p.dtype)))


def _posit_quantize_torch(x: torch.Tensor, nsize: int, es: int, scale: float) -> torch.Tensor:
    _check_posit(nsize, es)
    if x.dtype != torch.float32:
        raise TypeError(f"posit_quantize expects a float32 tensor, got {x.dtype}")
    xs = x * scale if scale != 1.0 else x
    bits = xs.contiguous().view(torch.int32).to(torch.int64)
    neg = bits < 0
    mag = bits & _F32_ABS
    useed_zeros = 1 << es
    maxreal = (useed_zeros * (nsize - 2) + 127) << 23  # float32 bits of maxpos
    minreal = (useed_zeros * (2 - nsize) + 127) << 23  # float32 bits of minpos

    code = _posit_round_magnitude(mag, nsize, es).clamp(0, (1 << (nsize - 1)) - 1)
    table = posit_value_table(nsize, es, device=x.device)
    val = table[code]
    val = torch.where(mag <= minreal, table[1], val)          # nonzero underflow -> minpos
    val = torch.where(mag >= maxreal, table[-1], val)         # overflow -> maxpos
    val = torch.where(neg, -val, val)
    val = torch.where(mag == 0, torch.zeros_like(val), val)   # +-0 -> +0
    val = torch.where(mag >= _F32_INF, torch.full_like(val, float("-inf")), val)
    return val / scale if scale != 1.0 else val


def _float_quantize_torch(x: torch.Tensor, exp: int, man: int) -> torch.Tensor:
    if not (1 <= exp <= 8 and 0 <= man <= 23):
        raise ValueError(f"unsupported float format exp={exp} man={man}")
    if x.dtype != torch.float32:
        raise TypeError(f"float_quantize expects a float32 tensor, got {x.dtype}")
    old = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    mask = (1 << (23 - man)) - 1
    q = (old + (1 << (23 - man - 1) if man < 23 else 0)) & 0xFFFFFFFF & ~mask
    q_exp = (q & _F32_ABS) >> 23
    min_store = 127 - ((1 << (exp - 1)) - 1)
    max_store = 127 + ((1 << (exp - 1)) - 1)
    old_sign = old & 0x80000000
    max_man = ((1 << man) - 1) << (23 - man)
    over = old_sign | (max_store << 23) | max_man
    under = torch.where(
        (q & _F32_ABS) > ((min_store - 1) << 23),
        old_sign | (min_store << 23),
        torch.zeros_like(q),
    )
    out = torch.where(q_exp > max_store, over, torch.where(q_exp < min_store, under, q))
    out = torch.where(q == 0, q, out)
    out = torch.where(out >= 0x80000000, out - (1 << 32), out)  # back to signed int32 range
    return out.to(torch.int32).view(torch.float32)


def posit_quantize(x: torch.Tensor, nsize: int, es: int, scale: float = 1.0,
                   rounding: str = "nearest") -> torch.Tensor:
    """Round a float32 tensor to the nearest Posit(nsize, es) value.

    Same signature and results as ``qtorch_plus.quant.posit_quantize``;
    qtorch_plus also maps ``rounding="stochastic"`` to nearest.
    """
    if rounding not in ("nearest", "stochastic"):
        raise ValueError(f"invalid rounding mode {rounding!r}")
    if _BACKEND == "qtorch":
        from qtorch_plus.quant import posit_quantize as _q
        return _q(x, nsize=nsize, es=es, scale=scale, rounding="nearest")
    return _posit_quantize_torch(x, nsize, es, float(scale))


def float_quantize(x: torch.Tensor, exp: int, man: int, rounding: str = "nearest") -> torch.Tensor:
    """Round a float32 tensor to a (1, exp, man) float format.

    Same results as ``qtorch_plus.quant.float_quantize(..., rounding="nearest")``.
    """
    if rounding != "nearest":
        raise ValueError("only rounding='nearest' is supported")
    if _BACKEND == "qtorch":
        from qtorch_plus.quant import float_quantize as _q
        return _q(x, exp=exp, man=man, rounding="nearest")
    return _float_quantize_torch(x, exp, man)
