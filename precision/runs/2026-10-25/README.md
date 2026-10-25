# Phase 0 and Phase 1 of #1, 2026-10-25

The precision matrix's first full runs, from `main` at `9bd776a` (#2). The results are in [`results/qe-mini1/2026-10-25.jsonl`](../../../results/qe-mini1/2026-10-25.jsonl) and [`results/qe-wharton1-gpu/2026-10-25.jsonl`](../../../results/qe-wharton1-gpu/2026-10-25.jsonl).

| Run | Machine | Devices | Rows | Driver and log |
|---|---|---|---|---|
| Phase 0 | `qe-mini1`: Apple M6, 12 cores, 32 GB | CPU | 46, one of them skipped for memory | [`phase0.sh`](phase0.sh), [`phase0.log`](phase0.log), [`phase0-state.txt`](phase0-state.txt) |
| Phase 1 | `qe-wharton1-gpu`: two TITAN RTX, Threadripper 2990WX, 128 GB | CPU, one card (`0b:00.0`), two cards | 120, one skipped for memory | [`phase1.sh`](phase1.sh), [`phase1.log`](phase1.log) |
| Phase 1, `interleave` | the same | CPU under `numactl --interleave=all` | 46 | the same |
| Phase 1, `cuda12` | the same | as Phase 1, with `jax[cuda12]` in place of `jax[cuda13]` | 120 | the same |
| Phase 0, the T4 | RunsOn `g4dn.2xlarge`: Tesla T4 (15 GB), Xeon Platinum 8259CL (8 threads), driver 580.105.08 | CPU and the T4, in both CUDA builds (the `cuda12` rows labelled) | 184 | workflow runs [40165642256](https://github.com/QuantEcon/benchmarks/actions/runs/40165642256) (CUDA 13) and [40165644168](https://github.com/QuantEcon/benchmarks/actions/runs/40165644168) (CUDA 12), from `main` at `963b2c2` |

All runs used JAX 0.12.0.1 (`precision/constraints.txt`): Python 3.12.15 on the Mac, and 3.12.3 with driver 595.91.07 (CUDA 13.2) on the GPU box. The skipped cells are `bellman_stored` at 60k in float64, which needs 26.8 GiB per device: on one 24 GB card, and on the Mac's 32 GB at the default `--memory-fraction`.

## The conditions

- **`qe-mini1` runs agents around the clock.** `phase0.sh` paused them (SIGSTOP), together with the console's animated wallpaper, and resumed them (SIGCONT) on exit, after 2 minutes. `phase0-state.txt` records the machine before, during and after. While paused, the busiest process was WindowServer, at 1.8% CPU.
- **`phase0.sh` is as it ran,** with unanchored `pgrep -f` patterns. They also matched the shell that launched it, which was paused too, harmlessly for the run. [`lectures/mac_set.sh`](../../../lectures/mac_set.sh) has the anchored patterns that fix this.
- **`qe-wharton1-gpu` ran nothing else.** No GitHub runner is registered there, and the second card drives the desktop.

## The T4's out-of-memory cell

On the T4, `bellman_stored` at 40k in float64 failed with `RESOURCE_EXHAUSTED` while allocating its 11.92 GiB array, in both builds. The planner had admitted it under its default budget of 80% of 15 GiB, which is 12.0 GiB. JAX's GPU allocator, though, appears to cap itself at 75% by default (`XLA_PYTHON_CLIENT_MEM_FRACTION`), about 11.25 GiB, even with preallocation off. The cell is a correct capacity data point, recorded as an error rather than a skip. Giving each GPU cell an `XLA_PYTHON_CLIENT_MEM_FRACTION` that matches `--memory-fraction` would make the planner and the allocator agree; that fix is still to be tested on a card.

## Checks

[`checks/cpu_cliff.txt`](checks/cpu_cliff.txt) shows why the CPU cells at 20k in float64 are 7× to 23× slower than float32: the stored step's throughput collapses once a float64 array passes about 2 GiB. The precision README's caveats have the details.

The rest of [`checks/`](checks/) holds the diagnosis of a second anomaly: on the M6, the Kalman filter runs about 3.5× slower in float32 than in float64. Its 10 × 10 matrix products go through XLA's YNNPACK fusion, which on this machine costs about 1.1 µs per call in float32, against 0.16 µs in float64. The precision README's caveats have the details and a workaround. The rows here are as used, with default flags.
