# Small float32 products on qe-mini1's M6, on macOS 27.2, 2026-11-09

For QuantEcon/project-benchmarks-jax#4: [2026-10-25's checks](../../../2026-10-25/checks/) again, after the upgrade to macOS 27.2 (26B123), in the same quiet window as [the probe](../mac-float64/README.md), from 02:33:46 to 02:35:27 UTC, with the account logged in at the screen. The script is `../mac-float64/session/m6_window.sh`.

| File | What ran |
|---|---|
| `ynn_dot_check.txt` | `../../2026-10-25/checks/ynn_dot_check.py`, from `main` at `ba58764`, in float32 and float64, then three times in float32 with `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=` |
| `repro4.txt` | #4's reproducer, `session/repro4.py`, as #4's draft gives it, in the same three settings |
| `matrix.txt`, `matrix-default/`, `matrix-ynn-cleared/` | `precision/run_matrix.py`'s `kalman` and `matmul` at 1,000 and 10,000, in both precisions, with and without the fusion, from the runs clone at `5fbf033` and its venv |
| `state.txt`, `run.log` | The machine every 5 s, and the window's log |

**The `lowering` column** of `ynn_dot_check.txt` also lists `ynn_dot_check`, since the script's pattern matches its own file name in the compiled module. Only `ynn_fusion` and `Custom` say how the product lowered.

**One interruption:** `mediaanalysisd` took 59% to 98% of a core from 02:33:56 to 02:34:01, during the check's runs with the fusion cleared and the reproducer, and 32% at 02:34:52. These are single-threaded timings on a 12-core Mac, and every figure below agrees with 2026-10-25's within 5%.

## What they show

| | 2026-10-25 (macOS 27.0.1) | 2026-11-09 (macOS 27.2) |
|---|---:|---:|
| One 10 × 10 product in float32, µs | 1.113 | 1.078 |
| The same in float64 | 0.168 | 0.166 |
| The same in float32, the fusion cleared | 0.163 | 0.162 |
| #4's reproducer: float32, float64, float32 cleared, µs | 1.112, 0.173, 0.175 | 1.098, 0.174, 0.169 |
| `kalman` at T = 1,000, float32 by default, ms | 5.08 | 5.04 |
| The same, the fusion cleared | 1.42 | 1.40 |
| The same in float64 | 1.39 | 1.36 |
| `matmul` at 1,000, float32 by default, GFLOPS | 2,017 | 1,938 |
| The same, the fusion cleared | 851 | 849 |

**So #4's cost is unchanged on macOS 27.2:** about 1.1 µs per small float32 product under YNNPACK's fusion, against about 0.17 µs in float64 or with the fusion cleared.
