# Accelerate against XLA on qe-mini1's M6, on macOS 27.2, 2026-11-09

For QuantEcon/project-benchmarks-jax#48's follow-up: whether macOS 27.2 changed Accelerate's float64 products. `precision/probes/mac_float64/` ran as on [2026-11-06](../../../2026-11-06/qe-mini1/mac-float64/README.md), unchanged on `main` at `ba58764` since `3806db6` (`run1/commit.txt`).

| | |
|---|---|
| Host | `qe-mini1`: the same Apple M6 Mac mini, on macOS 27.2 (26B123) from 2026-11-09 (`run1/features.txt`). The same day it moved to a rack on the same network |
| Software | The benchmarks venv: JAX 0.12.0.1 and NumPy 2.5.4, whose BLAS and LAPACK are Accelerate's. MLX 0.32.4, in a venv of its own |
| When | 2026-11-09, 02:35:27 to 02:38:05 UTC, in a quiet window agreed with the sessions that run on this Mac, which stayed running. The account was logged in at the screen |
| Script | `session/m6_window.sh login2`: #4's checks first ([`../checks/`](../checks/README.md)), then `session/m6_probe.sh` three times, each sampling the machine into its run's `state.txt` every 5 s |

**Where the results are:** `run1/`, `run2/` and `run3/`, three full runs back to back, from 02:35:27, 02:36:20 and 02:37:12 UTC. Each holds `run.sh`'s output, the optimised modules in `xla-dump/`, and its `state.txt`. They ran as `2026-11-09-qe-mini1-login2-run1` and so on, and were renamed, so their cliff rows' `xla_flags` name the dump under the old folder names.

**Two earlier attempts that day are left out.** Their output stays on the Mac, unpublished.
- **01:26 to 01:31 UTC:** nobody was logged in at the screen after the upgrade's restart, and `audiomxd` (about 73% of a core) and `configd` (about 42%) ran throughout.
- **02:09 to 02:13 UTC:** just after a login, `mediaanalysisd` ran at about a core, with `ANECompilerService` in bursts of up to 5.9 cores, through all three runs.

**One interruption in these runs, as on 2026-11-06:** `mediaanalysisd` took up to 123% of a core for about 10 s from within about 3 s of each run's cliff step, then 18% to 32% through step 4. Once more, in run 3, it took 89% during YNNPACK's products, which explains that run's 335 GFLOPS at 1,024. No control ran this time. 2026-11-06's control bounded the daemon's effect on step 4 to a few per cent.

## What they show

Against 2026-11-06, on macOS 27.0.1. The ranges are across each date's three runs.

| | 2026-11-06 | 2026-11-09 |
|---|---:|---:|
| NumPy (Accelerate), n = 1,024, GFLOPS | 556.6–558.3 | 543.5–550.1 |
| NumPy (Accelerate), n = 4,096 | 542.3–544.7 | 536.0–542.8 |
| JAX, XLA's own dot, n = 1,024 | 409.3–429.5 | 378.9–395.4 |
| JAX, XLA's own dot, n = 4,096 | 461.7–483.1 | 437.5–444.2 |
| JAX, YNNPACK, n = 4,096 | 360.8–383.9 | 340.0–351.2 |
| Accelerate ÷ the faster JAX, n = 1,024 | 1.28–1.30 | 1.31–1.44 |
| Accelerate ÷ the faster JAX, n = 4,096 | 1.12–1.18 | 1.22–1.23 |
| JAX's `P @ V` step, n = 4,000, ms | 340.5–350.7 | 363.1–367.0 |
| JAX's stored Bellman step, n = 14,000, ms | 11.98–12.05 | 12.17–12.25 |

- **Accelerate's products are level** with 2026-11-06's, within 2%.
- **JAX's large float64 products are about 6% slower,** and its `P @ V` step about 5%. Whether that's macOS 27.2 or the Mac's new place isn't established.
- **Unchanged:** the small products (at n = 64, Accelerate's bare call is still 3.4 times as fast as YNNPACK's) and the cliff (default 121.8–126.2 GB/s at 16,000, 34.6–35.1 at 18,000; YNNPACK's reduce fusion off, 45.5–45.8 at both). One `add` row at 18,000 read 24.4 GB/s during the daemon's burst.
- **So Accelerate still leads by less than 2×,** short of the several times that would call for an Accelerate kernel.
