# FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators (MICRO 2026)

This repo is the FlexPosit mixed-precision quantization framework.

FlexPosit received all three MICRO 2026 artifact badges:
**Artifacts Available**, **Artifacts Evaluated — Functional**, and **Results Reproduced**. 

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

## Test

    pytest tests/

Six tests: package smoke-imports and a Conv1D-axis regression on
`_to_cout_first` / `_from_cout_first` in `flexposit.quantizers.posit`.
Takes ~20s once the `qtorch_plus` CUDA extension has JIT-compiled.

## Supported models

Every CLI takes `--model <short_name>` and looks up the HuggingFace id itself
(see `src/flexposit/models.py`). Each has a pre-computed sensitivity CSV in
`data/sensitivity/`.

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

## End-to-end flow

Two scripts do the work:

    bash scripts/01_quantize_base.sh phi-2   # quantize weights to Posit base
    bash scripts/02_mpq_sweep.sh    phi-2    # MPQ sweep, using the shipped PPL sensitivity CSV

Outputs land in `out/`. Common overrides:

    NSIZE=5 bash scripts/01_quantize_base.sh phi-2                        # Posit(5,1)
    MPQ_ARGS="--target_avg_bits 4.7" bash scripts/02_mpq_sweep.sh phi-2   # single target

## End-to-end flow (Fisher variant)

Fisher is a faster proxy for the PPL-probe sensitivity and runs on the FP
reference model directly — no `01_quantize_base.sh` dependency:

    bash scripts/regen_sensitivity_fisher.sh phi-2   # regen sensitivity CSV via Fisher
    bash scripts/01_quantize_base.sh         phi-2   # quantize weights to Posit base
    SENS_CSV=out/fisher_phi-2.csv \
        bash scripts/02_mpq_sweep.sh phi-2           # MPQ sweep, using the Fisher CSV

The first two are independent — run them in parallel to save wallclock.

## Regenerating sensitivity (optional)

We ship a PPL-probe CSV per supported model in `data/sensitivity/`.
Regenerate only if you want a new model, a different channel-window, or a
different method. Both regenerators write the same schema; plug the result
into `02_mpq_sweep.sh` via `SENS_CSV=...`.

- **PPL-probe** (canonical, slower). Needs the base checkpoint from
  `01_quantize_base.sh`; auto-dispatches to the Conv1D-aware variant for GPT-2.

      bash scripts/regen_sensitivity_ppl.sh phi-2   # -> out/sens_phi-2/sensitivity.csv

- **Fisher** (faster proxy). No `01_quantize_base.sh` dependency. See the
  Fisher-variant end-to-end flow above.

## Layout

    src/flexposit/         # importable package
    ├── models.py          # MODEL_PRESETS: short-name -> HF id + load flags
    ├── ppl.py             # WikiText-2 PPL harness
    ├── quantizers/        # base Posit + comparison baselines (int4, mxfp8)
    ├── mpq/               # mixed-precision drivers (channel_window, layer)
    └── sensitivity/       # sensitivity CSV generators (ppl_probe,
                           #   ppl_probe_conv1d, fisher)

    data/sensitivity/      # nine pre-computed sensitivity CSVs (one per model
                           #   at the channel-window used in the paper)

    scripts/               # bash wrappers around the CLIs
    ├── _env.sh                          # sourced: activate conda, ensure CUDA
    ├── 01_quantize_base.sh              # -> flexposit.quantizers.posit
    ├── 02_mpq_sweep.sh                  # -> flexposit.mpq.channel_window
    ├── regen_sensitivity_ppl.sh         # -> flexposit.sensitivity.ppl_probe[_conv1d]
    └── regen_sensitivity_fisher.sh      # -> flexposit.sensitivity.fisher

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
