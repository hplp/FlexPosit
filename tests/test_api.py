"""flexposit.quantize / save / load_state on tiny models."""
import pytest
import torch

import flexposit
from flexposit.api import plan_windows, read_sensitivity
from flexposit.formats import posit_values
from flexposit.utils import quantizable_layers, to_cout_first

from .conftest import make_sensitivity


def _assert_on_grid(model, state):
    """Every weight equals posit_value * 2**-log2_scale for its channel's format."""
    for name, mod in quantizable_layers(model):
        ls = state.layers[name]
        w = to_cout_first(mod, mod.weight.detach()).float()
        scaled = w * torch.pow(2.0, ls.log2_scale.float())[:, None]
        for n in torch.unique(ls.nsize).tolist():
            vals = torch.tensor(posit_values(int(n), ls.es))
            grid = torch.cat([-vals, vals])
            assert torch.isin(scaled[ls.nsize == n], grid).all(), name


@pytest.mark.parametrize("fixture", ["tiny_llama", "tiny_gpt2"])
def test_uniform_posit4(fixture, request):
    model = request.getfixturevalue(fixture)
    ref = {k: v.clone() for k, v in model.state_dict().items()}
    state = flexposit.quantize(model, flexposit.FlexPositConfig(bits=4.0), progress=False)
    names = [n for n, _ in quantizable_layers(model)]
    assert set(state.layers) == set(names)
    assert state.avg_bits == 4.0 and state.window_bits == 4.0
    _assert_on_grid(model, state)
    changed = [k for k, v in model.state_dict().items() if not torch.equal(v, ref[k])]
    assert changed and all(k.rsplit(".", 1)[0] in names for k in changed)  # only quantized weights change
    assert "lm_head.weight" not in changed


def test_mixed_precision_budget(tiny_llama):
    rows = make_sensitivity(tiny_llama)
    state = flexposit.quantize(tiny_llama, flexposit.FlexPositConfig(bits=4.5), sensitivity=rows,
                               progress=False)
    total = len(rows)
    assert state.total_windows == total
    assert len(state.upgraded_windows) == -(-total // 2)  # ceil(total * 0.5)
    # the most beneficial (most negative delta_ppl) windows are the ones upgraded
    best = sorted(rows, key=lambda r: (r[3], r[0], r[1], r[2]))[: len(state.upgraded_windows)]
    assert set(state.upgraded_windows) == {(ly, ws, we) for ly, ws, we, _ in best}
    for ly, ws, we in state.upgraded_windows:
        assert (state.layers[ly].nsize[ws:we] == 5).all()
    assert 4.0 < state.avg_bits < 5.0
    _assert_on_grid(tiny_llama, state)


def test_mixed_precision_needs_sensitivity(tiny_llama):
    with pytest.raises(ValueError, match="sensitivity"):
        flexposit.quantize(tiny_llama, flexposit.FlexPositConfig(bits=4.3), progress=False)


def test_plan_windows_endpoints():
    rows = [("a", 0, 8, -1.0), ("a", 8, 16, 0.5), ("b", 0, 8, -2.0), ("b", 0, 8, -2.0)]
    cfg = flexposit.FlexPositConfig
    assert plan_windows(rows, {"a", "b"}, cfg(bits=4.0)) == ([], 3)
    assert plan_windows(rows, {"a", "b"}, cfg(bits=5.0))[0] == [("b", 0, 8), ("a", 0, 8), ("a", 8, 16)]
    assert plan_windows(rows, {"a"}, cfg(bits=4.5)) == ([("a", 0, 8)], 2)


def test_config_validation():
    with pytest.raises(ValueError):
        flexposit.FlexPositConfig(bits=3.5)
    with pytest.raises(ValueError):
        flexposit.FlexPositConfig(bits=4.5, base_nsize=5, upgrade_nsize=4)


def test_save_and_load_state(tiny_llama, tmp_path):
    rows = make_sensitivity(tiny_llama)
    state = flexposit.quantize(tiny_llama, flexposit.FlexPositConfig(bits=4.25), sensitivity=rows,
                               progress=False)
    flexposit.save(tiny_llama, None, state, str(tmp_path))
    loaded = flexposit.load_state(str(tmp_path))
    assert loaded.config == state.config
    assert loaded.upgraded_windows == state.upgraded_windows
    for name, ls in state.layers.items():
        assert torch.equal(loaded.layers[name].nsize, ls.nsize)
        assert torch.equal(loaded.layers[name].log2_scale, ls.log2_scale)
    from transformers import AutoModelForCausalLM
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path))
    for k, v in tiny_llama.state_dict().items():
        assert torch.equal(reloaded.state_dict()[k], v)


def test_perplexity_runs(tiny_gpt2):
    ids = torch.randint(0, 128, (1, 200))
    ppl = flexposit.eval.perplexity(tiny_gpt2, ids, seqlen=64)
    ppl_b = flexposit.eval.perplexity(tiny_gpt2, ids, seqlen=64, batch_size=3)
    assert ppl == pytest.approx(ppl_b, rel=1e-5) and 1.0 < ppl < 1e4


def test_fp8_activation_hooks(tiny_llama):
    handles = flexposit.add_fp8_activation_quant(tiny_llama)
    assert len(handles) == len(list(quantizable_layers(tiny_llama)))
    out = tiny_llama(torch.randint(0, 128, (1, 16))).logits
    assert torch.isfinite(out).all()
    for h in handles:
        h.remove()


def test_shipped_sensitivity_resolves_by_name():
    names = flexposit.shipped_sensitivity()
    assert {"gpt2-large", "phi-2", "llama-2-7b", "mistral-7b", "qwen2.5-14b"} <= set(names)
    for name in names:
        rows = read_sensitivity(name)
        assert rows and all(ws < we for _, ws, we, _ in rows), name
    path = flexposit.sensitivity_csv("phi-2")
    assert flexposit.sensitivity_csv(path) == path


def test_unknown_sensitivity_name():
    with pytest.raises(FileNotFoundError, match="shipped models"):
        flexposit.sensitivity_csv("no-such-model")
