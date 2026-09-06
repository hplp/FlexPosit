# FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators (MICRO 2026)

This repo is the FlexPosit mixed-precision quantization framework.

For exact paper reproduction (Tables 2-6, Figs 11-12, Table 10), use the
MICRO 2026 artifact: https://github.com/hplp/FlexPosit_artifact

## Install

    bash install.sh                    # creates conda env `flexposit`
    conda activate flexposit

Prerequisites: `conda` on PATH, an NVIDIA driver compatible with CUDA 11.8,
and a CUDA toolkit with `nvcc >= 11.8` on PATH (any 11.8+ works, including
CUDA 12.x). The toolkit is required because `qtorch_plus` JIT-compiles a
CUDA extension on first import — the torch wheel alone (which only ships
the runtime) is not sufficient. If your default `nvcc` is too old (e.g.
`/usr/bin/nvcc` at 11.5), point PATH at a newer install first:

    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    bash install.sh

## End-to-end flow

Pick a `MODEL` from the shipped set below (each has a pre-computed
sensitivity CSV in `sensitivity/`). Every CLI takes `--model $MODEL` and
looks up the HuggingFace id itself.

| Short name        | HuggingFace id                       |
| ----------------- | ------------------------------------ |
| `gpt2-large`      | `gpt2-large`                         |
| `gpt2-xl`         | `gpt2-xl`                            |
| `phi-2`           | `microsoft/phi-2`                    |
| `opt-2.7b`        | `facebook/opt-2.7b`                  |
| `llama-2-7b`      | `meta-llama/Llama-2-7b-hf`           |
| `mistral-7b`      | `mistralai/Mistral-7B-v0.1`          |
| `deepseek-llm-7b` | `deepseek-ai/deepseek-llm-7b-base`   |
| `qwen2.5-7b`      | `Qwen/Qwen2.5-7B`                    |
| `qwen2.5-14b`     | `Qwen/Qwen2.5-14B`                   |

    export MODEL=phi-2

Then:

    # 1. Quantize weights to Posit(4,1) base
    python -m flexposit.quantizers.posit \
        --model $MODEL --nsize 4 --es_candidates 1 \
        --save_dir out/${MODEL}_posit4

    # 2. Apply mixed-precision sweep using the shipped sensitivity CSV
    python -m flexposit.mpq.channel_window \
        --model $MODEL \
        --base_dir out/${MODEL}_posit4 \
        --sensitivity_csv sensitivity/$MODEL.csv \
        --sweep_bits_start 4.0 --sweep_bits_end 5.0 \
        --es_candidates 1 \
        --out_dir out/mpq_${MODEL}_sweep

`--nsize 4` selects Posit(4,1); pass `--nsize 5` for Posit(5,1). Replace
`--sweep_bits_*` with `--target_avg_bits 4.7` for a single target instead
of a sweep.

## Regenerating sensitivity (optional)

The shipped `sensitivity/$MODEL.csv` files are the ones used in the paper.
Regenerate only if you want a different channel-window, a new model, or an
alternative sensitivity method. Then feed the regenerated CSV to step 2's
`--sensitivity_csv` above.

    # PPL-probe (canonical but slower)
    python -m flexposit.sensitivity.window \
        --model $MODEL --model_dir out/${MODEL}_posit4 \
        --channel_window 256 --es_candidates 1 \
        --out_dir out/sens_${MODEL}
    # For GPT-2 use flexposit.sensitivity.conv1d (Conv1D wrapper) instead.

    # Fisher (faster)
    python -m flexposit.sensitivity.fisher \
        --model $MODEL --channel_window 256 \
        --out_csv out/fisher_${MODEL}.csv \
        --b_low 4 --b_high 5

## Layout

    flexposit/
    ├── ppl.py             # WikiText-2 PPL harness
    ├── quantizers/        # base Posit + comparison baselines (int4, mxfp8)
    ├── mpq/               # mixed-precision drivers (channel-window + layer)
    └── sensitivity/       # sensitivity CSV generators (window, conv1d, fisher)

Nine pre-computed sensitivity CSVs (one per model at the channel-window used
in the paper) ship in `sensitivity/`.

## Paper & citation

Preprint (arXiv): https://arxiv.org/abs/2609.04724

```bibtex
@misc{gao2026flexposit,
  title         = {FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators},
  author        = {Gao, Yimin and Dai, Liangtao and Yin, Jun and Guo, Xinfei and Stan, Mircea},
  year          = {2026},
  eprint        = {2609.04724},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AR},
  doi           = {10.48550/arXiv.2609.04724}
}
```

## Contact

Yimin Gao <yg9bq@virginia.edu>
