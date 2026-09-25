"""flexposit.core: the per-channel power-of-two scale search."""
import torch

from flexposit.core import quantize_channels, search_channels
from flexposit.formats import posit_quantize


def _brute_force(row, nsize, es, log2s):
    best = None
    for l2 in log2s:
        s = 2.0**l2
        q = posit_quantize(row * s, nsize, es) / s
        sqnr = 10.0 * torch.log10(((row * row).sum() + 1e-8) / (((row - q) ** 2).sum() + 1e-8))
        if best is None or sqnr > best[0]:
            best = (sqnr, l2, q)
    return best


def test_search_matches_brute_force():
    torch.manual_seed(0)
    w = torch.randn(40, 128) * torch.exp(torch.randn(40, 1))
    res = search_channels(w, nsize=4, es=1, ch_batch=16)
    for c in range(0, 40, 7):
        sqnr, l2, q = _brute_force(w[c], 4, 1, range(-8, 10))
        assert res.log2_scale[c].item() == l2
        assert torch.equal(res.q[c], q)


def test_batch_size_does_not_change_result():
    torch.manual_seed(1)
    w = torch.randn(70, 64)
    a = search_channels(w, 5, ch_batch=64)
    b = search_channels(w, 5, ch_batch=7)
    assert torch.equal(a.q, b.q) and torch.equal(a.log2_scale, b.log2_scale)


def test_quantize_channels_reproduces_search():
    torch.manual_seed(2)
    w = torch.randn(32, 96)
    res = search_channels(w, 4, es=1)
    assert torch.equal(quantize_channels(w, 4, res.log2_scale, res.es), res.q)


def test_zero_row_is_stable():
    w = torch.zeros(3, 16)
    res = search_channels(w, 4)
    assert torch.equal(res.q, w)
