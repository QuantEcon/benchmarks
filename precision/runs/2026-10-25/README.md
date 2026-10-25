# Phase 0 and Phase 1 of #1, 2026-10-25

The precision matrix's first full runs, from `main` at `9bd776a` (#2). The results are in [`results/qe-mini1/2026-10-25.jsonl`](../../../results/qe-mini1/2026-10-25.jsonl) and [`results/qe-wharton1-gpu/2026-10-25.jsonl`](../../../results/qe-wharton1-gpu/2026-10-25.jsonl).

| Run | Machine | Devices | Rows | Driver and log |
|---|---|---|---|---|
| Phase 0 | `qe-mini1`: Apple M6, 12 cores, 32 GB | CPU | 46, one of them skipped for memory | [`phase0.sh`](phase0.sh), [`phase0.log`](phase0.log), [`phase0-state.txt`](phase0-state.txt) |
| Phase 1 | `qe-wharton1-gpu`: two TITAN RTX, Threadripper 2990WX, 128 GB | CPU, one card (`0b:00.0`), two cards | 120, one skipped for memory | [`phase1.sh`](phase1.sh), [`phase1.log`](phase1.log) |
| Phase 1, `interleave` | the same | CPU under `numactl --interleave=all` | 46 | the same |
| Phase 1, `cuda12` | the same | as Phase 1, with `jax[cuda12]` in place of `jax[cuda13]` | 120 | the same |

All runs used JAX 0.12.0.1 (`precision/constraints.txt`): Python 3.12.15 on the Mac, and 3.12.3 with driver 595.91.07 (CUDA 13.2) on the GPU box. The skipped cells are `bellman_stored` at 60k in float64, which needs 26.8 GiB per device: on one 24 GB card, and on the Mac's 32 GB at the default `--memory-fraction`.

## The conditions

- **`qe-mini1` runs agents around the clock.** `phase0.sh` paused them (SIGSTOP), together with the console's animated wallpaper, and resumed them (SIGCONT) on exit, after 2 minutes. `phase0-state.txt` records the machine before, during and after. While paused, the busiest process was WindowServer, at 1.8% CPU.
- **`phase0.sh` is as it ran,** with unanchored `pgrep -f` patterns. They also matched the shell that launched it, which was paused too, harmlessly for the run. [`lectures/mac_set.sh`](../../../lectures/mac_set.sh) has the anchored patterns that fix this.
- **`qe-wharton1-gpu` ran nothing else.** No GitHub runner is registered there, and the second card drives the desktop.

## Checks

[`checks/`](checks/) holds the diagnosis of one anomaly: on the M6, the Kalman filter runs about 3.5× slower in float32 than in float64. Its 10 × 10 matrix products go through XLA's YNNPACK fusion, which on this machine costs about 1.1 µs per call in float32, against 0.16 µs in float64. The precision README's caveats have the details and a workaround. The rows here are as used, with default flags.
