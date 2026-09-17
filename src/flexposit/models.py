"""Registry mapping short model names to HuggingFace ids and load flags.

Consumed by every CLI in flexposit.* via `--model <short_name>`.
"""

MODEL_PRESETS = {
    "gpt2":         {"hf_id": "gpt2"},
    "gpt2-medium":  {"hf_id": "gpt2-medium"},
    "gpt2-large":   {"hf_id": "gpt2-large"},
    "gpt2-xl":      {"hf_id": "gpt2-xl"},

    "opt-125m":     {"hf_id": "facebook/opt-125m"},
    "opt-350m":     {"hf_id": "facebook/opt-350m"},
    "opt-1.3b":     {"hf_id": "facebook/opt-1.3b"},
    "opt-2.7b":     {"hf_id": "facebook/opt-2.7b"},
    "opt-6.7b":     {"hf_id": "facebook/opt-6.7b"},

    "bloom-7b1":    {"hf_id": "bigscience/bloom-7b1"},

    "phi-2":        {"hf_id": "microsoft/phi-2"},
    "yi-6b":        {"hf_id": "01-ai/Yi-6B", "trust_remote_code": True},
    "llama-2-7b":   {"hf_id": "meta-llama/Llama-2-7b-hf"},
    "llama-3-8b":   {"hf_id": "meta-llama/Meta-Llama-3-8B"},

    "qwen2.5-14b": {
        "hf_id": "Qwen/Qwen2.5-14B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "qwen2-7b": {
        "hf_id": "Qwen/Qwen2-7B",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    "deepseek-llm-7b": {
        "hf_id": "deepseek-ai/deepseek-llm-7b-base",
        "trust_remote_code": True,
        "use_fast_tokenizer": False,
        "requires_auth": False,
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    "phi-3-small": {
        "hf_id": "microsoft/Phi-3-small-8k-instruct",
        "trust_remote_code": True,
        "use_fast_tokenizer": True,
        "requires_auth": False,
    },
    # GATED
    "llama-2-13b": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "trust_remote_code": False,
        "use_fast_tokenizer": True,
        "requires_auth": True,
    },
}
