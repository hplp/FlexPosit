"""Regression: HF Conv1D stores weights as (Cin, Cout); the per-channel search
must iterate along Cout, not Cin. This was the bug that produced wrong PPL
for GPT-2 in earlier revisions of the code. See _to_cout_first /
_from_cout_first in flexposit.quantizers.posit.
"""
import pytest
import torch
import torch.nn as nn

from flexposit.quantizers.posit import _to_cout_first, _from_cout_first


def test_conv1d_transposed_to_cout_first():
    from transformers.modeling_utils import Conv1D
    conv1d = Conv1D(nf=3, nx=4)     # Cin=4, Cout=3; storage shape (Cin, Cout)=(4, 3)
    assert tuple(conv1d.weight.shape) == (4, 3)
    W_cout = _to_cout_first(conv1d, conv1d.weight)
    assert tuple(W_cout.shape) == (3, 4), (
        f"Conv1D weights should be transposed to (Cout, Cin)=(3, 4); got {tuple(W_cout.shape)}"
    )


def test_linear_passthrough():
    linear = nn.Linear(4, 3)        # Cin=4, Cout=3; storage shape (Cout, Cin)=(3, 4)
    assert tuple(linear.weight.shape) == (3, 4)
    W_cout = _to_cout_first(linear, linear.weight)
    assert tuple(W_cout.shape) == (3, 4), (
        f"nn.Linear weights are already (Cout, Cin); should be unchanged"
    )
    assert torch.equal(W_cout, linear.weight)


def test_roundtrip_conv1d():
    from transformers.modeling_utils import Conv1D
    conv1d = Conv1D(nf=6, nx=8)
    W_orig = conv1d.weight.detach().clone()
    W_back = _from_cout_first(conv1d, _to_cout_first(conv1d, W_orig))
    assert tuple(W_back.shape) == tuple(W_orig.shape)
    assert torch.equal(W_back, W_orig)


def test_roundtrip_linear():
    linear = nn.Linear(8, 6)
    W_orig = linear.weight.detach().clone()
    W_back = _from_cout_first(linear, _to_cout_first(linear, W_orig))
    assert tuple(W_back.shape) == tuple(W_orig.shape)
    assert torch.equal(W_back, W_orig)
