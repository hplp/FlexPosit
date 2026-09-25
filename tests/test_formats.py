"""flexposit.formats: bit-exact with qtorch_plus, plus Posit value-set properties."""
import json
from pathlib import Path

import pytest
import torch

from flexposit.formats import _float_quantize_torch, _posit_quantize_torch, posit_values

GOLDEN = json.loads((Path(__file__).parent / "data" / "qtorch_golden.json").read_text())


def _from_hex(values):
    ints = torch.tensor([int(v, 16) for v in values], dtype=torch.int64)
    return torch.where(ints >= 2**31, ints - 2**32, ints).to(torch.int32).view(torch.float32)


def _bits(t):
    return t.view(torch.int32)


X = _from_hex(GOLDEN["inputs_hex"])


@pytest.mark.parametrize("key", sorted(GOLDEN["posit"]))
def test_posit_matches_qtorch_golden(key):
    n, es = map(int, key.split("_"))
    assert torch.equal(_bits(_posit_quantize_torch(X, n, es, 1.0)), _bits(_from_hex(GOLDEN["posit"][key])))


@pytest.mark.parametrize("key", sorted(GOLDEN["float"]))
def test_float_matches_qtorch_golden(key):
    e, m = map(int, key.split("_"))
    assert torch.equal(_bits(_float_quantize_torch(X, e, m)), _bits(_from_hex(GOLDEN["float"][key])))


@pytest.mark.parametrize("n,es", [(n, es) for n in range(4, 9) for es in range(3)])
def test_posit_value_set_and_idempotence(n, es):
    vals = torch.tensor(posit_values(n, es))
    table = torch.cat([-vals.flip(0), vals])
    assert torch.equal(_posit_quantize_torch(table, n, es, 1.0), table + 0.0)  # exact values are fixed points
    x = torch.randn(20000) * 4.0 ** torch.randint(-6, 6, (20000,)).float()
    q = _posit_quantize_torch(x, n, es, 1.0)
    assert torch.isin(q, table).all()
    order = torch.argsort(x)
    assert (q[order].diff() >= 0).all()  # monotone


def test_posit_edge_cases():
    n, es = 4, 1
    vals = posit_values(n, es)
    minpos, maxpos = vals[1], vals[-1]
    x = torch.tensor([0.0, -0.0, 1e-30, -1e-30, 1e30, -1e30, float("inf"), float("nan")])
    q = _posit_quantize_torch(x, n, es, 1.0)
    assert q[:2].tolist() == [0.0, 0.0] and _bits(q[1:2]).item() == 0  # -0 -> +0
    assert q[2].item() == minpos and q[3].item() == -minpos           # no underflow to zero
    assert q[4].item() == maxpos and q[5].item() == -maxpos           # saturation
    assert q[6].item() == float("-inf") and q[7].item() == float("-inf")


def test_posit_scale_argument():
    x = torch.randn(1000)
    for s in (2.0**-3, 2.0**4):
        assert torch.equal(_posit_quantize_torch(x, 5, 1, s), _posit_quantize_torch(x * s, 5, 1, 1.0) / s)


def test_fp8_e4m3_range():
    # qtorch_plus convention: saturation at 240, and no subnormals -- the lowest
    # binade [2**-7, 2**-6) keeps a full 3-bit mantissa, values above 2**-8
    # snap up to 2**-7, and the rest flush to 0.
    x = torch.tensor([1000.0, -1000.0, 240.0, 1.125 * 2.0**-7, 1.1 * 2.0**-8, 2.0**-9])
    assert _float_quantize_torch(x, 4, 3).tolist() == [240.0, -240.0, 240.0, 1.125 * 2.0**-7, 2.0**-7, 0.0]


def test_rejects_non_float32():
    with pytest.raises(TypeError):
        _posit_quantize_torch(torch.zeros(3, dtype=torch.float16), 4, 1, 1.0)


def test_matches_qtorch_when_installed():
    qt = pytest.importorskip("qtorch_plus.quant")
    x = torch.randn(50000) * 4.0 ** torch.randint(-8, 8, (50000,)).float()
    for n in (4, 5, 8):
        for es in (0, 1, 2):
            assert torch.equal(_bits(_posit_quantize_torch(x, n, es, 1.0)), _bits(qt.posit_quantize(x, nsize=n, es=es)))
    assert torch.equal(_bits(_float_quantize_torch(x, 4, 3)), _bits(qt.float_quantize(x, exp=4, man=3, rounding="nearest")))
