# FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators (MICRO 2026)

[![tests](https://github.com/hplp/FlexPosit/actions/workflows/ci.yml/badge.svg)](https://github.com/hplp/FlexPosit/actions/workflows/ci.yml)

FlexPosit received all three MICRO 2026 artifact badges. To reproduce the
paper's results exactly, use
[FlexPosit_artifact](https://github.com/hplp/FlexPosit_artifact).

<p align="center">
  <a href="https://github.com/hplp/FlexPosit_artifact"><img src="https://raw.githubusercontent.com/hplp/FlexPosit/main/assets/artifacts_available_v1_1.png" height="100" alt="Artifacts Available"></a>
  <a href="https://github.com/hplp/FlexPosit_artifact"><img src="https://raw.githubusercontent.com/hplp/FlexPosit/main/assets/artifacts_evaluated_functional_v1_1.png" height="100" alt="Artifacts Evaluated — Functional"></a>
  <a href="https://github.com/hplp/FlexPosit_artifact"><img src="https://raw.githubusercontent.com/hplp/FlexPosit/main/assets/results_reproduced_v1_1.png" height="100" alt="Results Reproduced"></a>
</p>

FlexPosit is a Posit-based mixed-precision quantization framework for LLMs.
It allocates higher precision to the channel windows whose quantization most
affects perplexity. The paper tunes precision by combining Posit(4,1) and
Posit(5,1) across channel windows; other Posit formats can be selected through
the configuration.
This repo contains:

- **`flexposit`**, a Python package for quantizing HuggingFace models with
  FlexPosit.
- **[`hardware/`](https://github.com/hplp/FlexPosit/blob/main/hardware)**, the RTL of the FlexPosit datapath, part of the
  test-chip version of FlexPosit in an ongoing 12 nm tapeout shuttle.

## Install

```bash
pip install flexposit               # Python >= 3.10; includes WikiText-2 perplexity
pip install "flexposit[eval]"       # optional: adds lm-evaluation-harness for downstream tasks (ARC, HellaSwag, ...)
```

For the command-line scripts and the hardware, clone the repo and install it
in editable mode instead:

```bash
git clone https://github.com/hplp/FlexPosit && cd FlexPosit
pip install -e ".[dev]"
```

## Quick start

Use the Python API to quantize a model from your own code: load it, quantize it
to a target average bit width, then evaluate or save it. For example, Mistral-7B
at 4.4 bits:

```python
import flexposit

model, tok = flexposit.load_model("mistral-7b")          # preset name, HF id or local path

# 4.4 bits on average, using the sensitivity ranking shipped for this model
# (uniform Posit(4,1) is bits=4.0 and needs no sensitivity)
state = flexposit.quantize(model, flexposit.FlexPositConfig(bits=4.4), sensitivity="mistral-7b")

print(flexposit.wikitext2_perplexity(model, tok))        # WikiText-2, seqlen 2048
flexposit.eval.lm_eval(model, tok, ["arc_easy"])         # any lm-eval task
flexposit.save(model, tok, state, "out/mistral-7b-flexposit-4.4")
```

`quantize` rounds each weight to its channel's Posit format in place, so the
model runs anywhere a HuggingFace model runs; `state` records each channel's
Posit size and scale. To sweep many bit widths, use the command line below.

## Command line

The scripts run the paper's workflow from the shell: quantize every weight to
Posit(4,1), then sweep the average bit width from 4.0 to 5.0 and record
WikiText-2 perplexity at each step. Results go to `out/`.

    bash scripts/01_quantize_base.sh phi-2   # quantize weights to Posit(4,1)
    bash scripts/02_mpq_sweep.sh    phi-2    # mixed-precision sweep over 4.0–5.0 bits

The shipped sensitivity CSVs hold the PPL-based sensitivity used in the paper.
You can also profile your own, with a different configuration (e.g. the
channel-window size, i.e. the granularity) or a different method (a Fisher-based
one is provided). See [docs/cli.md](https://github.com/hplp/FlexPosit/blob/main/docs/cli.md) for these options and
the tests.

## Hardware

[`hardware/`](https://github.com/hplp/FlexPosit/blob/main/hardware) holds the RTL of a bit-serial FlexPosit accelerator
whose Posit precision (4–8 bits) changes per channel window at runtime.

```bash
sudo apt install iverilog   # or on macOS: brew install icarus-verilog
pip install numpy           # used by the Python reference model
cd hardware && make test
```

`make test` (about a minute) simulates the RTL and checks it bit for bit against
a Python model: every FP8 × Posit product on a single PE, then random matrix
tiles through the whole array.

## Supported models

Each has a pre-computed sensitivity CSV bundled with the package
(`flexposit.shipped_sensitivity()`); the CLIs and the Python API accept the
short names, both for the model and for `sensitivity=`.

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

## Layout

    src/flexposit/       Python package (API, quantizers, mixed precision, sensitivity)
    src/flexposit/data/  sensitivity CSVs for the supported models
    scripts/             shell wrappers for the command-line workflow
    hardware/            RTL, testbenches and bit-exact model
    docs/cli.md          command-line guide
    tests/               test suite

## Development

    pip install -e ".[dev]"
    ruff check src tests
    pytest                     # CPU, a few seconds, no downloads
    make -C hardware test      # RTL regression, needs Icarus Verilog

Issues and pull requests are welcome.

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

## License

MIT; see [LICENSE](https://github.com/hplp/FlexPosit/blob/main/LICENSE).

## Contact

Yimin Gao <yg9bq@virginia.edu>
