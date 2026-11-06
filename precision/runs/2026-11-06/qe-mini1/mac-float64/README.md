# Accelerate against XLA on qe-mini1's M6, 2026-11-06

For QuantEcon/project-benchmarks-jax#48: `precision/probes/mac_float64/` at `3806db6`, measured at its full sizes.

| | |
|---|---|
| Host | `qe-mini1`: an Apple M6 Mac mini, with 2 Super, 4 Performance and 6 Efficiency cores and 32 GB, on macOS 27.0.1 (26A434) (`run1/features.txt`) |
| Software | The benchmarks venv: JAX 0.12.0.1 and NumPy 2.5.4, whose BLAS and LAPACK are Accelerate's. MLX 0.32.4, with NumPy 2.5.4, in a venv of its own from `requirements-mlx.txt` |
| When | 2026-11-06, 08:00:00 to 08:05:15 UTC, in a quiet window agreed with the sessions that run on this Mac, which stayed running |
| Script | `session/m6_probe.sh`, which ran `run.sh` and sampled the machine into `state.txt` |

**Where the results are:**
- **`run1/`, `run2/` and `run3/`:** three full runs, back to back, from 08:00:00, 08:02:15 and 08:03:06 UTC.
    - **What each holds:** `run.sh`'s output (`features.txt`, `matmul.jsonl`, `cliff.jsonl`, `step.jsonl`, `summary.md`, `run.log` and `commit.txt`), the optimised modules in `xla-dump/`, and the session's `state.txt`.
- **`stepcheck/`:** step 4 alone at its full sizes, from 08:04:56 to 08:05:15 UTC, as a control.

**One interruption, and the control for it:**
- **mediaanalysisd,** macOS's media-analysis daemon, took up to 122% of a core in each full run.
    - **When:** from about 30 s in, for 10 to 20 s, during step 4, just after the cliff step had written its XLA dump.
    - **How it was seen:** run 1's `state.txt` samples every 30 s, and runs 2 and 3's every 5 s.
- **The control** ran step 4 alone, with no dump before it, and no other process used more than 10% of a core.
    - **Its times** were within 1% to 4% of the three runs', and mostly a little faster.
    - **So** the daemon cost step 4 a few per cent at most.
- **Between runs,** 34 of the 39 cells agree within 5%. The other five:
    - **NumPy at n = 8:** its single product (0.25 to 0.29 µs, near the timer's resolution) and its recurrence;
    - **JAX's products at n = 512 and at 4,096,** which slowed run by run, while NumPy's held within 0.4%. YNNPACK's at 4,096 took 358, 363 and 381 ms;
    - **MLX's `P @ V` at 1,000,** at 3.2 to 3.5 ms.

**Three notes on the files:**
- **Run 1's folder was renamed** after it ran, so its cliff rows' `xla_flags` name the dump at `2026-11-06-qe-mini1/xla-dump/`.
- **`session/m6_probe.sh` is the script as runs 2 and 3 ran it.** Run 1 ran it before it took a label, and sampled every 30 s rather than every 5 s. Its header still says 30 s, from run 1.
- **The control** ran `run.sh`'s two `probe.py step` lines for each framework by hand, at the full sizes: `bellman 2000,8000,14000` and `pv 1000,4000`.

## What they show

The ranges are across the three runs.

- **The M6 reports float64 SME:** `FEAT_SME`, `FEAT_SME2`, `FEAT_SME2p1` and `FEAT_SME_F64F64` are all 1.
- **Large products:** Accelerate leads, by less than 2×.

    | n | NumPy (Accelerate), GFLOPS | JAX, YNNPACK | JAX, XLA's own dot |
    |---:|---:|---:|---:|
    | 512 | 494–498 | 384–405 | 348–351 |
    | 1,024 | 557–558 | 426–434 | 409–430 |
    | 4,096 | 542–545 | 361–384 | 462–483 |

    - **At 4,096,** Accelerate's product is 1.4 to 1.5 times as fast as YNNPACK's, and 1.1 to 1.2 times as fast as XLA's own.
    - **Every JAX answer** agrees with NumPy's within 6.3e-15.
- **Small products, one a step for 10,000 steps:**
    - **At n = 64,** NumPy's bare `np.dot` call takes 2.43–2.44 µs, against YNNPACK's 8.11–8.36 and XLA's own 9.92–10.34. So it's 3.3 to 4.3 times as fast. NumPy's whole step, with Python's dispatch of three operations, takes 4.12–4.21 µs.
    - **At n = 8,** JAX's 0.14–0.15 µs beats NumPy's bare call, at 0.20.
- **The cliff,** in GB/s of the m × m array:

    | m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
    |---:|---:|---:|---:|---:|---:|
    | 16,000 | 1.91 | 128.2–132.7 | 45.8–45.9 | 59.4–60.1 | 158.7–162.0 |
    | 18,000 | 2.41 | 35.0–35.2 | 45.9–46.2 | 59.8–60.5 | 80.9 |

    - **On the M6, the drop is in YNNPACK's reduce:** the default falls about 3.7-fold past 2 GiB, and the row max alone, which YNNPACK takes, halves.
    - **With YNNPACK's reduce fusion off,** XLA's own code holds 46 GB/s at both sizes. That's slower than the default below the cliff, and faster past it.
- **MLX on its CPU device:**
    - **Its `P @ V` is Accelerate's:** about 1.5 times as fast as JAX's at n = 4,000 (229.5–233.6 ms, against 340.5–350.7), with NumPy's answer exactly.
    - **Its stored Bellman step** is 9 to 11 times as slow as JAX's: 109–112 ms at 14,000, against 12.0–12.1.
