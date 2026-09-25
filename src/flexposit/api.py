"""High-level API: quantize a model with FlexPosit in a few lines.

    import flexposit
    model, tok = flexposit.load_model("llama-2-7b")
    cfg = flexposit.FlexPositConfig(bits=4.5)
    state = flexposit.quantize(model, cfg, sensitivity="data/sensitivity/llama-2-7b.csv")
    flexposit.save(model, tok, state, "out/llama-2-7b-flexposit-4.5")

``quantize`` implements the paper's channel-window mixed precision in budget
mode: every Linear/Conv1D weight starts at Posit(base_nsize, es); the
channel windows with the most negative sensitivity (largest PPL gain from
upgrading) are raised to Posit(upgrade_nsize, es) until the requested share
of windows is reached. Each output channel gets its own SQNR-optimal
power-of-two scale (flexposit.core).

Weights are fake-quantized in place: they are stored in the model's dtype but
take only Posit values times a power of two. The returned :class:`QuantState`
records the format of every channel; flexposit.export turns it into the
integer codes the FlexPosit hardware consumes.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from flexposit.core import search_channels
from flexposit.utils import from_cout_first, is_conv1d, quantizable_layers, to_cout_first

__all__ = ["FlexPositConfig", "LayerState", "QuantState", "quantize", "plan_windows",
           "read_sensitivity", "save", "load_state"]

STATE_FILE = "flexposit_state.safetensors"
CONFIG_FILE = "flexposit_config.json"


@dataclass
class FlexPositConfig:
    """FlexPosit quantization settings. Defaults follow the paper."""

    bits: float = 4.0
    """Target average precision in [base_nsize, upgrade_nsize], counted as the
    share of channel windows upgraded (the paper's "average bits")."""
    base_nsize: int = 4
    upgrade_nsize: int = 5
    es: int = 1
    log2_min: int = -8
    log2_max: int = 9
    skip_lm_head: bool = True
    ch_batch: int = 64

    def __post_init__(self):
        if not self.base_nsize <= self.upgrade_nsize:
            raise ValueError("base_nsize must be <= upgrade_nsize")
        if not self.base_nsize <= self.bits <= self.upgrade_nsize:
            raise ValueError(f"bits must be in [{self.base_nsize}, {self.upgrade_nsize}], got {self.bits}")
        if not 0 <= self.es < self.base_nsize:
            raise ValueError(f"es must be in [0, base_nsize), got {self.es}")


@dataclass
class LayerState:
    """Per-output-channel format of one quantized weight (output-channel order)."""
    nsize: torch.Tensor       # [Cout] int8
    log2_scale: torch.Tensor  # [Cout] int8; weight = posit_value * 2**-log2_scale
    es: int
    shape: tuple[int, int]    # (Cout, Cin)
    conv1d: bool              # stored transposed in the model (HF Conv1D)


@dataclass
class QuantState:
    config: FlexPositConfig
    layers: dict[str, LayerState] = field(default_factory=dict)
    upgraded_windows: list[tuple[str, int, int]] = field(default_factory=list)
    total_windows: int = 0

    @property
    def window_bits(self) -> float:
        """The paper's average-bits metric: share of windows upgraded."""
        c = self.config
        if self.total_windows == 0:
            return float(c.base_nsize)
        frac = len(self.upgraded_windows) / self.total_windows
        return c.base_nsize + frac * (c.upgrade_nsize - c.base_nsize)

    @property
    def avg_bits(self) -> float:
        """Average Posit bits per weight element (excludes the per-channel scale)."""
        bits = sum(float(ls.nsize.double().sum()) * ls.shape[1] for ls in self.layers.values())
        n = sum(ls.shape[0] * ls.shape[1] for ls in self.layers.values())
        return bits / n if n else 0.0

    def summary(self) -> str:
        n_up = len(self.upgraded_windows)
        return (f"{len(self.layers)} layers | windows upgraded {n_up}/{self.total_windows} | "
                f"window bits {self.window_bits:.3f} | bits/weight {self.avg_bits:.3f}")


def read_sensitivity(path: str) -> list[tuple[str, int, int, float]]:
    """Rows (layer, win_start, win_end, delta_ppl) from a sensitivity CSV.

    More negative ``delta_ppl`` = more PPL improvement when the window is upgraded.
    """
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            layer = (row.get("layer") or "").strip()
            try:
                ws, we, dp = int(row["win_start"]), int(row["win_end"]), float(row["delta_ppl"])
            except (KeyError, TypeError, ValueError):
                continue
            if layer and not math.isnan(dp):
                rows.append((layer, ws, we, dp))
    return rows


def plan_windows(rows, valid_layers, config: FlexPositConfig) -> tuple[list[tuple[str, int, int]], int]:
    """Choose windows to upgrade, as flexposit.mpq.channel_window does in budget mode.

    Returns (upgraded windows, total unique windows).
    """
    rows = [r for r in rows if r[0] in valid_layers]
    rows.sort(key=lambda r: (r[3], r[0], r[1], r[2]))
    total = len({(ly, ws, we) for ly, ws, we, _ in rows})
    span = config.upgrade_nsize - config.base_nsize
    frac = (config.bits - config.base_nsize) / span if span else 0.0
    n_upgrade = int(math.ceil(total * frac))
    picked, seen = [], set()
    for ly, ws, we, _ in rows:
        if len(picked) >= n_upgrade:
            break
        if (ly, ws, we) not in seen:
            seen.add((ly, ws, we))
            picked.append((ly, ws, we))
    return picked, total


@torch.no_grad()
def quantize(model: nn.Module, config: FlexPositConfig | None = None, sensitivity=None,
             progress: bool = True) -> QuantState:
    """Quantize every Linear/Conv1D weight of ``model`` in place.

    ``sensitivity`` is a CSV path (see data/sensitivity/) or a list of
    (layer, win_start, win_end, delta_ppl) rows. It is required when
    ``config.bits`` is above ``config.base_nsize``; compute one for a new model
    with ``python -m flexposit.sensitivity.fisher`` (fast) or
    ``python -m flexposit.sensitivity.ppl_probe`` (the paper's method).
    """
    config = config or FlexPositConfig()
    layers = list(quantizable_layers(model, skip_lm_head=config.skip_lm_head))
    state = QuantState(config=config)

    if config.bits > config.base_nsize:
        if sensitivity is None:
            raise ValueError(
                f"bits={config.bits} needs a sensitivity ranking to decide which channel windows get "
                f"Posit{config.upgrade_nsize}. Pass sensitivity='data/sensitivity/<model>.csv', or create "
                "one with `python -m flexposit.sensitivity.fisher`.")
        rows = read_sensitivity(sensitivity) if isinstance(sensitivity, (str, os.PathLike)) else list(sensitivity)
        state.upgraded_windows, state.total_windows = plan_windows(rows, {n for n, _ in layers}, config)
        if state.total_windows == 0:
            raise ValueError("No sensitivity rows match this model's layer names.")

    by_layer: dict[str, list[tuple[int, int]]] = {}
    for ly, ws, we in state.upgraded_windows:
        by_layer.setdefault(ly, []).append((ws, we))

    it = layers
    if progress:
        from tqdm.auto import tqdm
        it = tqdm(layers, desc="FlexPosit", unit="layer")
    for name, mod in it:
        w = to_cout_first(mod, mod.weight.detach()).float()
        cout = w.shape[0]
        nsize = torch.full((cout,), config.base_nsize, dtype=torch.int8)
        for ws, we in by_layer.get(name, []):
            nsize[max(0, ws):min(cout, we)] = config.upgrade_nsize
        q = torch.empty_like(w)
        log2_scale = torch.empty(cout, dtype=torch.int8)
        for n in torch.unique(nsize).tolist():
            rows_n = (nsize == n).nonzero().squeeze(1)
            res = search_channels(w[rows_n.to(w.device)], int(n), es=config.es, log2_min=config.log2_min,
                                  log2_max=config.log2_max, ch_batch=config.ch_batch)
            q[rows_n.to(w.device)] = res.q
            log2_scale[rows_n] = res.log2_scale.cpu().to(torch.int8)
        mod.weight.data = from_cout_first(mod, q).to(mod.weight.dtype)
        state.layers[name] = LayerState(nsize=nsize, log2_scale=log2_scale, es=config.es,
                                        shape=(cout, w.shape[1]), conv1d=is_conv1d(mod))
    return state


def save(model: nn.Module, tokenizer, state: QuantState, out_dir: str) -> None:
    """Save the fake-quantized model, tokenizer and FlexPosit metadata to ``out_dir``."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    tensors = {}
    for name, ls in state.layers.items():
        tensors[f"{name}.nsize"] = ls.nsize.contiguous()
        tensors[f"{name}.log2_scale"] = ls.log2_scale.contiguous()
    save_file(tensors, os.path.join(out_dir, STATE_FILE))
    meta = {
        "flexposit_config": asdict(state.config),
        "layers": {n: {"es": ls.es, "shape": list(ls.shape), "conv1d": ls.conv1d}
                   for n, ls in state.layers.items()},
        "upgraded_windows": [list(w) for w in state.upgraded_windows],
        "total_windows": state.total_windows,
        "window_bits": state.window_bits,
        "avg_bits": state.avg_bits,
    }
    with open(os.path.join(out_dir, CONFIG_FILE), "w") as f:
        json.dump(meta, f, indent=2)


def load_state(out_dir: str) -> QuantState:
    """Read the metadata written by :func:`save`."""
    from safetensors.torch import load_file

    with open(os.path.join(out_dir, CONFIG_FILE)) as f:
        meta = json.load(f)
    tensors = load_file(os.path.join(out_dir, STATE_FILE))
    state = QuantState(config=FlexPositConfig(**meta["flexposit_config"]),
                       upgraded_windows=[tuple(w) for w in meta["upgraded_windows"]],
                       total_windows=meta["total_windows"])
    for name, info in meta["layers"].items():
        state.layers[name] = LayerState(nsize=tensors[f"{name}.nsize"], log2_scale=tensors[f"{name}.log2_scale"],
                                        es=info["es"], shape=tuple(info["shape"]), conv1d=info["conv1d"])
    return state
