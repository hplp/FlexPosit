"""Tiny randomly initialized models so tests run on CPU without downloads."""
import pytest
import torch


@pytest.fixture
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=128, n_embd=64, n_layer=2, n_head=4, n_positions=64)
    return GPT2LMHeadModel(cfg).eval()


def make_sensitivity(model, window=16, seed=0):
    """Synthetic (layer, win_start, win_end, delta_ppl) rows over every quantizable layer."""
    from flexposit.utils import quantizable_layers, to_cout_first
    g = torch.Generator().manual_seed(seed)
    rows = []
    for name, mod in quantizable_layers(model):
        cout = to_cout_first(mod, mod.weight).shape[0]
        for ws in range(0, cout, window):
            rows.append((name, ws, min(cout, ws + window), float(torch.randn(1, generator=g))))
    return rows
