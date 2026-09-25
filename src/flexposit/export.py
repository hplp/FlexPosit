"""Export quantized weights as the integer codes the FlexPosit hardware consumes.

After :func:`flexposit.quantize`, every weight equals ``posit(code) * 2**-l``
for its channel's ``(nsize, es, l)``. This module recovers those codes and
writes GEMM tiles in the format of ``hardware/tb/array_tb.v``:

    tile = flexposit.export.export_tile(model, state, "model.layers.0.mlp.down_proj",
                                        activations, col_start=0, n=16, k_start=0, k=64)
    tile.write_mem("hardware/build")      # then simulate hardware/tb/array_tb.v

Formats (see hardware/README.md):
  * weights: native two's-complement Posit codes; ``stream_code`` converts to
    the sign-magnitude (SerialPosit) order the decoder expects, MSB first.
  * activations: FP8 E4M3 codes (bias 7), as produced by
    ``flexposit.formats.float_quantize(x, 4, 3)``.
  * per-column scale: signed exponent ``-l``, so the RTL computes
    out[i][j] = 2**scale[j] * sum_k act[i][k] * posit(code[j][k]).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from flexposit.api import QuantState
from flexposit.formats import float_quantize, posit_value_table
from flexposit.utils import to_cout_first

__all__ = ["posit_encode", "stream_code", "layer_codes", "fp8_e4m3_encode", "RtlTile", "export_tile"]


def posit_encode(values: torch.Tensor, nsize: int, es: int) -> torch.Tensor:
    """Native Posit(nsize, es) codes (int64) of tensors holding exact Posit values."""
    table = posit_value_table(nsize, es, device=values.device)
    mag = values.abs().float()
    pos = torch.searchsorted(table, mag).clamp(max=table.numel() - 1)
    if not torch.equal(table[pos], mag):
        raise ValueError(f"values are not exact Posit({nsize},{es}) values")
    return torch.where(values < 0, (-pos) & ((1 << nsize) - 1), pos)


def stream_code(code: torch.Tensor | int, nsize: int):
    """Native two's-complement code -> sign-magnitude (SerialPosit) stream order."""
    sign = 1 << (nsize - 1)
    low = sign - 1
    if isinstance(code, int):
        return code if not code & sign else sign | ((-(code & low)) & low)
    return torch.where((code & sign) != 0, sign | ((-(code & low)) & low), code)


@torch.no_grad()
def layer_codes(model: nn.Module, state: QuantState, name: str):
    """(codes [Cout, Cin] int64, nsize [Cout], log2_scale [Cout]) for a quantized layer."""
    ls = state.layers[name]
    mod = model.get_submodule(name)
    w = to_cout_first(mod, mod.weight.detach()).float().cpu()
    scaled = w * torch.pow(2.0, ls.log2_scale.float())[:, None]
    codes = torch.empty_like(w, dtype=torch.int64)
    for n in torch.unique(ls.nsize).tolist():
        rows = ls.nsize == n
        codes[rows] = posit_encode(scaled[rows], int(n), ls.es)
    return codes, ls.nsize.clone(), ls.log2_scale.clone()


def fp8_e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Standard E4M3 codes (bias 7) for ``float_quantize(x, 4, 3)``.

    qtorch_plus-style E4M3 has no subnormals, so values in [2**-7, 2**-6) are
    rounded to the nearest E4M3 subnormal (error <= 2**-10); the rest map exactly.
    """
    q = float_quantize(x.float(), exp=4, man=3)
    bits = q.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = (bits >> 31) & 1
    field = ((bits >> 23) & 0xFF) - 127 + 7
    man = (bits >> 20) & 0x7
    normal = (sign << 7) | (field.clamp(1, 15) << 3) | man
    # field 0 in qtorch's view: value (8 + man) * 2**-10 -> subnormal mantissa (8 + man) / 2, RNE
    sub_man = (8 + man) >> 1
    sub_man = sub_man + ((man & 1) & (sub_man & 1))
    subnormal = (sign << 7) | sub_man  # sub_man == 8 carries into the smallest normal
    code = torch.where(field <= 0, subnormal, normal)
    return torch.where((bits & 0x7FFFFFFF) == 0, torch.zeros_like(code), code)


@dataclass
class RtlTile:
    """One N x N GEMM tile for hardware/tb/array_tb.v."""
    acts: torch.Tensor     # [N, K] FP8 E4M3 codes
    weights: torch.Tensor  # [N, K] native Posit codes (column j = output channel col_start + j)
    scale: torch.Tensor    # [N] signed per-column exponent (= -log2_scale)
    nsize: int
    es: int

    @property
    def n(self) -> int:
        return self.weights.shape[0]

    @property
    def k(self) -> int:
        return self.weights.shape[1]

    def write_mem(self, out_dir: str) -> None:
        """Write act.mem, w.mem, scale.mem as array_tb.v reads them."""
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "act.mem"), "w") as f:
            f.writelines(f"{int(a):02x}\n" for a in self.acts.flatten().tolist())
        with open(os.path.join(out_dir, "w.mem"), "w") as f:
            for col in self.weights.tolist():
                for c in col:
                    s = stream_code(int(c), self.nsize)
                    f.writelines(f"{(s >> b) & 1}\n" for b in range(self.nsize - 1, -1, -1))
        with open(os.path.join(out_dir, "scale.mem"), "w") as f:
            f.writelines(f"{int(s) & 0x1F:02x}\n" for s in self.scale.tolist())


def export_tile(model: nn.Module, state: QuantState, name: str, activations: torch.Tensor,
                col_start: int, n: int, k_start: int = 0, k: int | None = None) -> RtlTile:
    """Build an RTL tile: ``n`` output channels of layer ``name`` against ``n`` activation rows.

    ``activations`` is [n, Cin] (float); it is quantized to FP8 E4M3 here.
    All ``n`` channels must share one Posit size (true inside a channel window).
    """
    codes, nsize, log2_scale = layer_codes(model, state, name)
    cols = slice(col_start, col_start + n)
    k = codes.shape[1] - k_start if k is None else k
    ks = slice(k_start, k_start + k)
    sizes = torch.unique(nsize[cols]).tolist()
    if len(sizes) != 1:
        raise ValueError(f"channels {col_start}..{col_start + n - 1} mix Posit sizes {sizes}; "
                         "pick columns inside one channel window")
    scale = -log2_scale[cols].to(torch.int64)
    if scale.min() < -16 or scale.max() > 15:
        raise ValueError("per-column scale does not fit the RTL's signed 5-bit field")
    if activations.shape[0] != n:
        raise ValueError(f"need {n} activation rows, got {activations.shape[0]}")
    return RtlTile(acts=fp8_e4m3_encode(activations[:, ks].cpu()), weights=codes[cols, ks],
                   scale=scale, nsize=int(sizes[0]), es=state.layers[name].es)
