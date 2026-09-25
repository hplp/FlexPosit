"""A quantized layer exports to codes the FlexPosit hardware model computes correctly."""
import importlib.util
import sys
from pathlib import Path

import torch

import flexposit
from flexposit.export import export_tile
from flexposit.formats import float_quantize
from flexposit.utils import to_cout_first

from .conftest import make_sensitivity

HW_MODEL = Path(__file__).resolve().parent.parent / "hardware" / "model" / "flexposit_rtl_exact.py"
_spec = importlib.util.spec_from_file_location("flexposit_rtl_exact", HW_MODEL)
rtl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = rtl
_spec.loader.exec_module(rtl)


def test_exported_tile_matches_float_reference(tiny_llama):
    state = flexposit.quantize(tiny_llama, flexposit.FlexPositConfig(bits=4.5),
                               sensitivity=make_sensitivity(tiny_llama), progress=False)
    name, n, k = "model.layers.0.mlp.up_proj", 8, 32
    acts = torch.randn(n, 64, generator=torch.Generator().manual_seed(0))
    tile = export_tile(tiny_llama, state, name, acts, col_start=0, n=n, k=k)

    w = to_cout_first(tiny_llama.get_submodule(name), tiny_llama.get_submodule(name).weight).float()
    ref = float_quantize(acts[:, :k], 4, 3) @ w[:n, :k].T
    got = torch.tensor([[rtl.decode_fp16(rtl.pipeline_lane_fp16(
        [rtl.product(int(tile.acts[i, kk]), int(tile.weights[j, kk]), tile.nsize, tile.es) for kk in range(k)],
        int(tile.scale[j]))) for j in range(n)] for i in range(n)])
    # the bit-serial multiplier keeps 6 bits below the activation's leading one
    assert torch.allclose(got, ref, rtol=2**-5, atol=2**-5 * ref.abs().max().item())
