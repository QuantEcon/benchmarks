# Accelerate against XLA on qe-wharton2's M5 Pro, 2026-11-06

For QuantEcon/project-benchmarks-jax#48: `precision/probes/mac_float64/` at `3806db6`, measured at its full sizes, as on the M6 ([`../../qe-mini1/mac-float64/`](../../qe-mini1/mac-float64/README.md)).

| | |
|---|---|
| Host | `qe-wharton2`: an Apple M5 Pro, with 18 cores (6 Super and 12 Performance) and 64 GB, on macOS 26.7.1 (25G241), on AC power with Low Power Mode off |
| Software | The problem-type suite's venv: JAX 0.12.0.1 and NumPy 2.5.4, whose BLAS and LAPACK are Accelerate's, on Python 3.12.15. MLX 0.32.4, with NumPy 2.5.4, in a venv of its own from `requirements-mlx.txt` |
| When | 2026-11-06, 14:49:16 to 14:51:15 UTC, straight after the problem-type suite (`problems/results/qe-wharton2/2026-11-06/`), in the same window |
| Scripts | `session/probe48_setup.sh` put the probe's commit in a worktree and MLX in its venv, and warmed the imports. Then `session/probe48.sh` ran `run.sh` twice, and step 4 alone as a control, sampling the machine every 5 s into each folder's `state.txt`. Both are kept as they ran |

**Where the results are:**
- **`run1/` and `run2/`:** two full runs, back to back, from 14:49:23 and 14:50:09 UTC.
    - **What each holds:** `run.sh`'s output (`features.txt`, `matmul.jsonl`, `cliff.jsonl`, `step.jsonl`, `summary.md`, `run.log` and `commit.txt`), the optimised modules in `xla-dump/`, and the session's `state.txt`.
    - **Each ended** "done, every step ok".
- **`stepcheck/`:** step 4 alone at its full sizes, from 14:50:54 to 14:51:15 UTC, as a control: `run.sh`'s two `probe.py step` lines for each framework, `bellman 2000,8000,14000` and `pv 1000,4000`.
- **`session/`:** the two scripts and their logs.

**A guest machine, so nothing on it was paused:**
- **Its managed endpoint agents** took up to 17% of a core in each run. Their names are redacted in the `state.txt` files. The run's own account, `matt`, isn't, as in the machine's earlier runs.
- **Spotlight's `mds_stores`** took 50% of a core at one sample in run 1, at 14:49:53, during step 4, just after the cliff wrote its XLA dump. In run 2 it took 12%.
- **The control** came within 2% of the two runs' times, except MLX's `P @ V` at 1,000, 11% slower, and its Bellman step at 2,000, 6% faster.
- **Between runs,** 29 of the 39 cells agree within 5%. The other ten:
    - the four smallest cells, at n = 8, near the timer's resolution;
    - JAX's products at 512, 1,024 and 4,096, four cells that moved by 5% to 9%, and its `P @ V` at 4,000, by 6%;
    - MLX's Bellman step at 2,000, at 2.22 ms and 1.95 ms.

## What they show

The ranges are across the two runs.

- **The M5 Pro reports float64 SME:** `FEAT_SME`, `FEAT_SME2`, `FEAT_SME2p1` and `FEAT_SME_F64F64` are all 1, as on the M6.
- **Large products: JAX leads, the reverse of the M6.**

    | n | NumPy (Accelerate), GFLOPS | JAX, YNNPACK | JAX, XLA's own dot |
    |---:|---:|---:|---:|
    | 512 | 451.6–451.9 | 614.3–626.5 | 517.0–555.7 |
    | 1,024 | 480.2–482.1 | 774.3–813.2 | 656.7–668.0 |
    | 4,096 | 420.5–421.2 | 602.4–655.2 | 704.3–765.7 |

    - **At 1,024,** YNNPACK's product is 1.6 to 1.7 times as fast as Accelerate's. **At 4,096,** XLA's own is 1.7 to 1.8 times as fast.
    - **On the M6,** Accelerate led at both sizes, at 542 to 558 GFLOPS. The M5 Pro's Accelerate is slower than that, and its JAX faster.
    - **Every JAX answer** agrees with NumPy's within 6.3e-15.
- **Small products, one a step for 10,000 steps:**
    - **At n = 64,** NumPy's bare `np.dot` call takes 2.27–2.28 µs, against YNNPACK's 8.91–8.98 and XLA's own 11.39. So it's 3.9 to 5.0 times as fast, as on the M6. NumPy's whole step, with Python's dispatch of three operations, takes 4.61 µs.
    - **At n = 8,** JAX's 0.17–0.18 µs beats NumPy's bare call, at 0.22.
- **The cliff,** in GB/s of the m × m array:

    | m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
    |---:|---:|---:|---:|---:|---:|
    | 16,000 | 1.91 | 199.1–208.1 | 79.5–79.8 | 125.7–126.1 | 242.4–246.2 |
    | 18,000 | 2.41 | 37.7–38.7 | 79.5–79.6 | 126.6–126.7 | 77.7–78.1 |

    - **As on the M6, the drop is in YNNPACK's reduce:** the default falls 5.2- to 5.5-fold past 2 GiB, and the row max alone, which YNNPACK takes, about 3-fold.
    - **With YNNPACK's reduce fusion off,** XLA's own code holds about 80 GB/s at both sizes. That's slower than the default below the cliff, and faster past it.
- **MLX on its CPU device:**
    - **Its `P @ V` runs at Accelerate's speed,** 423 to 424 GFLOPS, so here it's 1.4 to 1.5 times as slow as JAX's at n = 4,000: 301.6–302.6 ms, against 203.8–215.3. On the M6 it was about 1.5 times as fast. Its answer is NumPy's exactly.
    - **Its stored Bellman step** is 13.7 to 13.9 times as slow as JAX's: 105.8 ms at 14,000, against 7.6–7.7.
