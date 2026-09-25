# FlexPosit command-line guide

The shell scripts in `scripts/` wrap the Python CLIs. After `pip install`, the
main CLIs are also available as `flexposit-quantize`, `flexposit-mpq` and
`flexposit-ppl`. Every module takes `--help` for the full list of options.

The scripts use the current Python if `flexposit` is installed there (`pip
install -e .`); otherwise they activate a conda env named `flexposit`, which
`bash install.sh` creates with the exact library versions used for the paper
(`requirements.txt`).

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
into `02_mpq_sweep.sh` via `SENS_CSV=...` or into `flexposit.quantize` via
`sensitivity=...`.

- **PPL-probe** (canonical, slower). Needs the base checkpoint from
  `01_quantize_base.sh`; auto-dispatches to the Conv1D-aware variant for GPT-2.

      bash scripts/regen_sensitivity_ppl.sh phi-2   # -> out/sens_phi-2/sensitivity.csv

- **Fisher** (faster proxy). No `01_quantize_base.sh` dependency. See the
  Fisher variant above.

## Parameter tuning

The scripts pass through env vars and `MPQ_ARGS` to the underlying CLIs. Every
python module also takes `--help` for the full list. Common tweaks below —
this repo is the framework, so mix and match freely; for locked, exact-paper
reproduction use `FlexPosit_artifact` instead.

**Precision target modes**

    # single target avg bits (budget mode)
    MPQ_ARGS="--target_avg_bits 4.5" bash scripts/02_mpq_sweep.sh phi-2

    # full Pareto sweep, writes ppl_vs_avg_bits.csv
    MPQ_ARGS="--sweep_bits_start 4.0 --sweep_bits_end 5.0 --sweep_bits_step 0.1" \
        bash scripts/02_mpq_sweep.sh phi-2

    # PPL goal instead of bit budget — upgrade greedily until PPL <= target
    MPQ_ARGS="--ppl_goal 12.5" bash scripts/02_mpq_sweep.sh phi-2

**Base Posit precision**

    NSIZE=5 bash scripts/01_quantize_base.sh phi-2                        # Posit(5,1) base

**Sensitivity strategy (ablations)**

    # random window ordering, seeded — for random-baseline comparisons
    MPQ_ARGS="--sweep_bits_start 4.0 --sweep_bits_end 5.0 \
              --sweep_strategy random --random_seed 42" \
        bash scripts/02_mpq_sweep.sh phi-2

    # location-based (deterministic by layer index)
    MPQ_ARGS="--sweep_bits_start 4.0 --sweep_bits_end 5.0 --sweep_strategy location" \
        bash scripts/02_mpq_sweep.sh phi-2

**Channel-window granularity** (when regenerating sensitivity)

    CHANNEL_WINDOW=128 bash scripts/regen_sensitivity_fisher.sh phi-2      # finer
    CHANNEL_WINDOW=512 bash scripts/regen_sensitivity_fisher.sh phi-2      # coarser

**Downgrade mode** (fewer iterations when target > (base+upgrade)/2)

    # start from Posit(5,1) base, downgrade least-sensitive windows to Posit(4,1)
    NSIZE=5 bash scripts/01_quantize_base.sh phi-2
    MPQ_ARGS="--target_avg_bits 4.9 --downgrade" bash scripts/02_mpq_sweep.sh phi-2

**Activation quantization** (FP8-E4M3 per-token dynamic, on top of weight-quant)

    python -m flexposit.ppl --model out/mpq_phi-2_t4.5 --act_quant fp8_e4m3

Full CLI reference: `python -m flexposit.mpq.channel_window --help` and
`python -m flexposit.quantizers.posit --help`.

## Tests

    pytest                     # CPU, a few seconds, no downloads
    make -C hardware test      # RTL regression, needs Icarus Verilog

The Python tests cover the Posit and FP8 kernels (bit-exact against
qtorch_plus, the library the paper's results were produced with), the scale
search, the mixed-precision planner, save/load, and the export to the
hardware.
