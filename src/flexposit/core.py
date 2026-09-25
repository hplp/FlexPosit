"""Per-channel Posit quantization: the numerical core of FlexPosit.

Each output channel (row of a [Cout, K] weight) is quantized as

    q = posit_quantize(w * 2**l, nsize, es) / 2**l

where the power-of-two scale ``2**l`` (``l`` in [log2_min, log2_max]) is
chosen per channel to maximize SQNR. ``es`` is fixed (the paper uses es=1);
passing several ``es`` values also searches es per channel. Scales are swept
in ascending order with es innermost, and a candidate must be strictly better
to replace the incumbent, so ties keep the smallest scale / first es.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from flexposit.formats import posit_quantize

EPS = 1e-8


@dataclass
class ChannelSearch:
    """Result of :func:`search_channels` for a block of channels."""
    q: torch.Tensor | None     # [C, K] quantized weights (fp32), if requested
    log2_scale: torch.Tensor   # [C] int64
    es: torch.Tensor           # [C] int64
    sqnr: torch.Tensor         # [C] fp32, dB


def _es_list(nsize: int, es) -> list[int]:
    cands = [es] if isinstance(es, int) else list(es)
    max_es = max(0, nsize - 1)
    return [int(e) for e in cands if int(e) <= max_es] or [0]


@torch.no_grad()
def search_channels(w: torch.Tensor, nsize: int, es=1, log2_min: int = -8,
                    log2_max: int = 9, ch_batch: int = 64, return_quantized: bool = True) -> ChannelSearch:
    """SQNR-optimal power-of-two scale per row of ``w`` [C, K], ``ch_batch`` rows at a time.

    ``es`` is an int, or a sequence of candidates to also search es per row.
    """
    w = w.detach().float()
    dev = w.device
    C, K = w.shape
    es_list = _es_list(nsize, es)
    log2s = list(range(log2_min, log2_max + 1))
    scales = torch.tensor([2.0 ** l for l in log2s], device=dev, dtype=torch.float32)

    q_out = torch.empty_like(w) if return_quantized else None
    l2_out = torch.empty(C, device=dev, dtype=torch.int64)
    es_out = torch.empty(C, device=dev, dtype=torch.int64)
    sqnr_out = torch.empty(C, device=dev, dtype=torch.float32)

    for c0 in range(0, C, ch_batch):
        c1 = min(C, c0 + ch_batch)
        x = w[c0:c1]
        sp = torch.sum(x * x, dim=1) + EPS
        best_sqnr = torch.full((c1 - c0,), -1e30, device=dev, dtype=torch.float32)
        best_q = torch.zeros_like(x) if return_quantized else None
        best_es = torch.zeros(c1 - c0, device=dev, dtype=torch.int64)
        best_l2 = torch.zeros(c1 - c0, device=dev, dtype=torch.int64)
        for l2, s in zip(log2s, scales):
            xs = x * s
            for es in es_list:
                q = posit_quantize(xs, nsize=nsize, es=es, scale=1.0) / s
                noise = torch.sum((x - q) ** 2, dim=1) + EPS
                sqnr = 10.0 * torch.log10(sp / noise)
                better = sqnr > best_sqnr
                best_sqnr = torch.where(better, sqnr, best_sqnr)
                if return_quantized:
                    best_q = torch.where(better[:, None], q, best_q)
                best_es = torch.where(better, es, best_es)
                best_l2 = torch.where(better, l2, best_l2)
        if return_quantized:
            q_out[c0:c1] = best_q
        l2_out[c0:c1] = best_l2
        es_out[c0:c1] = best_es
        sqnr_out[c0:c1] = best_sqnr
    return ChannelSearch(q_out, l2_out, es_out, sqnr_out)


@torch.no_grad()
def quantize_channels(w: torch.Tensor, nsize: int, log2_scale: torch.Tensor,
                      es: torch.Tensor) -> torch.Tensor:
    """Quantize rows of ``w`` [C, K] with given per-row log2 scales and es (no search)."""
    w = w.detach().float()
    out = torch.empty_like(w)
    for e in torch.unique(es).tolist():
        rows = es == e
        s = torch.pow(2.0, log2_scale[rows].float())[:, None]
        out[rows] = posit_quantize(w[rows] * s, nsize=nsize, es=int(e), scale=1.0) / s
    return out
