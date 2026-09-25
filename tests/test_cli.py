"""The command-line MPQ path (flexposit.mpq.channel_window) agrees with the Python API."""
import copy
import sys

import pytest

import flexposit
from flexposit.mpq import channel_window

from .conftest import make_sensitivity


@pytest.mark.parametrize("fixture", ["tiny_llama", "tiny_gpt2"])
def test_cli_windows_match_api(fixture, request):
    ref = request.getfixturevalue(fixture)
    ref_sd = {k: v.detach().clone().float() for k, v in ref.state_dict().items()}
    rows = make_sensitivity(ref)

    api_model = copy.deepcopy(ref)
    state = flexposit.quantize(api_model, flexposit.FlexPositConfig(bits=4.5), sensitivity=rows, progress=False)
    assert state.upgraded_windows

    # CLI flow: Posit(4,1) base checkpoint, then requantize the chosen windows from the FP reference.
    cli_model = copy.deepcopy(ref)
    flexposit.quantize(cli_model, flexposit.FlexPositConfig(bits=4.0), progress=False)
    scales = [2.0 ** k for k in range(-8, 10)]
    windows = [(ly, ws, we, we - ws) for ly, ws, we in state.upgraded_windows]
    channel_window.apply_windows_to_model(cli_model, ref_sd, windows, 5, [1], scales,
                                          skip_lm_head=True, quantize_embeddings=False)

    for (name, a), b in zip(api_model.named_parameters(), cli_model.parameters()):
        assert a.equal(b), name


def test_default_on_flags_can_be_disabled(monkeypatch):
    base = ["channel_window", "--model", "gpt2", "--base_dir", "b", "--sensitivity_csv", "s", "--out_dir", "o"]
    monkeypatch.setattr(sys, "argv", base)
    args = channel_window.get_args()
    assert args.skip_lm_head and args.allow_positive_to_meet_budget

    monkeypatch.setattr(sys, "argv", base + ["--no-skip_lm_head", "--no-allow_positive_to_meet_budget"])
    args = channel_window.get_args()
    assert not args.skip_lm_head and not args.allow_positive_to_meet_budget
