# The RTX 5090's sustained load and float64 emulation, 2026-11-18

For QuantEcon/project-benchmarks-jax#60, items 3 and 6, on the GeForce RTX 5090 rented on RunPod's Secure Cloud whose suites are in [`problems/results/runpod-rtx5090/2026-11-18/`](../../../../problems/results/runpod-rtx5090/2026-11-18/README.md), with the card's record there. From QuantEcon/benchmarks at `f085c6d`.

| Folder | What ran |
|---|---|
| `sustained/` | [`sustained.sh`](../../../sustained.sh), from 01:17 to 03:37 UTC: 10 minutes idle, an hour of `bellman_stored` at 40,000 in float64, 10 minutes idle, and an hour of `matmul` at 8,192 in float32 at JAX's default precision, with `nvidia-smi` once a second. `summary.md` compares each hour's first 5 minutes with its last 30 |
| `dot64/` | [`../../2026-11-01/dot64.py`](../../2026-11-01/dot64.py) with `--lib cupy`, in a separate environment with `cupy-cuda13x` 14.2.0 and cuBLAS 13.10.0.20 (`environment-cupy.txt`), from 03:37 to 03:48 UTC. It ran four times, each in a fresh process: native (`cupy-native`), and with `CUBLAS_EMULATE_DOUBLE_PRECISION=1` (`cupy-emulate`), at the default sizes and inputs; then both again over lognormal inputs with σ from 0.5 to 4 at n = 8,192 (`-sigma`). The four `.err` files were empty and aren't kept |

## The sustained load

The card held its clocks for both hours, with a datacentre's cooling. It shows whether the card holds its clocks in that host, not in another case.

| An hour each | Last 30 min ÷ first 5, step time | SM clock | Power | Temperature | Clock-event reasons |
|---|---:|---|---|---|---|
| `bellman_stored`, float64, 40,000 | 1.000 (12.11 ms, 1,057 GB/s) | 3,015 MHz | 472–475 W | 60–61 °C | none |
| `matmul`, float32, 8,192 | 1.005 (9.37 → 9.41 ms, 117 TFLOPS) | 2,782–2,790 MHz | 600 W | 69–70 °C | `SwPowerCap` throughout |

Before each hour the card was idle at 27–28 °C, 31–32 W and 300 MHz. The product is held by the 600 W limit, not by heat: no thermal reason was set in either hour. The stored step stays below the limit.

## Float64 emulation

cuBLAS's float64 emulation runs on this GeForce card, as it did on the RTX PRO 6000 Server Edition ([2026-11-01](../../2026-11-01/README.md)). The table is from `dot64.jsonl`. TFLOPS are float64 operations per second, and the error is the largest on an 8 × 256 corner against an 80-bit long double product, in units of float64's roundoff:

| Inputs | n | Native: TFLOPS, error | Emulated: TFLOPS, error |
|---|---:|---|---|
| Uniform on [0, 1) | 2,048 | 1.54, 36.8 | 6.27, 1.82 |
| | 8,192 | 1.76, 70.6 | 20.50, 1.77 |
| | 16,384 | 1.80, 112.8 | **30.04**, 1.83 |
| Standard normal | 2,048 | 1.55, 3.22 | 6.88, 0.11 |
| | 8,192 | 1.77, 3.09 | 20.13, 0.06 |
| | 16,384 | 1.81, 2.77 | **29.26**, 0.06 |
| Lognormal, σ = 5 | 2,048 to 16,384 | 1.56–1.81, 65–438 | 1.56–1.80, 57–412 |

At n = 16,384 the emulation is 16–17 times native, with a smaller error. With lognormal inputs at σ = 5, the plain `lognormal` kind, it falls back to native speed and native error, as on the Server Edition. The emulation needs the inputs' exponents to stay within a range, and σ = 5 spreads them too far. Across σ at n = 8,192 it stays emulated up to σ = 4:

| σ | 0.5 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Native: TFLOPS, error | 1.78, 85 | 1.78, 85 | 1.78, 96 | 1.78, 119 | 1.78, 116 |
| Emulated: TFLOPS, error | 20.29, 1.5 | 19.61, 1.4 | 18.39, 2.1 | 16.92, 2.1 | 16.25, 2.1 |

On the Server Edition on 2026-11-01, the same script gave 31.7 and 30.6 TFLOPS at 16,384 on uniform and normal inputs, and 27.95 at σ = 0.5 against 19.74 at σ = 4 at 8,192. So the 5090's emulation runs at 95% of the Server Edition's at the largest size. JAX's own float64 product doesn't reach it: in the precision matrix, `matmul` at 8,192 in float64 ran at 2,054 GFLOPS (`results/runpod-rtx5090/2026-11-18.jsonl`), native speed. openxla/xla#53281 asks XLA for a way to use the emulation.
