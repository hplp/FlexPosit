# FlexPosit Hardware

The FlexPosit accelerator datapath from our 12 nm test-chip design, currently in
a tapeout shuttle. It implements the architecture of our MICRO 2026 paper
([arXiv:2609.04724](https://arxiv.org/abs/2609.04724)): the SerialPosit
decoder, the bit-serial PE, and the output-stationary systolic array with
per-channel power-of-two rescale, with Posit precision (4–8 bits, es = 1) set
per channel window at runtime. Compared with the RTL evaluated in the paper, the
test chip moves to FP8 activations and one accumulator per PE.

## Quick start

```bash
sudo apt install iverilog     # or on macOS: brew install icarus-verilog
cd hardware
make test        # exhaustive single-PE check + random GEMM tiles through the array (~1 min)
make wave        # VCD of one small tile for GTKWave
```

Both tests compare every output bit against `model/flexposit_rtl_exact.py`.
Run your own tile with `python3 scripts/run_array_test.py --n 8 --k 32 --p 5 --es 1`.

## Citation

If you use this RTL in academic work, please cite our paper; the BibTeX is in
the [main README](../README.md#paper--citation).

MIT license; see [LICENSE](../LICENSE).
