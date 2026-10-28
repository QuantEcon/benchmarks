# The SSY workloads on an M1, 2026-10-28

`ssy_operator` and `ssy_jvp` ([`ssy.py`](../../ssy.py)), from `a3e9e04`. The results are in [`results/apple-m1-8gb/2026-10-28.jsonl`](../../../results/apple-m1-8gb/2026-10-28.jsonl).

| Run | Machine | Rows | Command |
|---|---|---|---|
| default | A MacBook Air with an Apple M1: 8 cores (4 performance, 4 efficiency), 8 GB, macOS 26.5.1 | 28: 19 run, 9 skipped for memory | `python precision/run_matrix.py --workloads ssy_operator,ssy_jvp --devices cpu --host apple-m1-8gb --memory-fraction 0.5` |
| `noynn` | the same | 28 | `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type= python precision/run_matrix.py --workloads ssy_operator,ssy_jvp --devices cpu --host apple-m1-8gb --label noynn --memory-fraction 1000` |

JAX 0.12.0.1 (`precision/constraints.txt`), Python 3.12.12, CPU only. Both passes ran from 01:15 UTC.

## The conditions

- **A personal laptop in use.** It ran on battery (65%, discharging) with low-power mode off and no thermal or performance warning. A browser and a terminal were open, and nothing was paused.
- **The default pass used half the memory as its limit**, to leave the machine usable: `ssy_operator` in float64 at n = 12 peaked at 3.9 GB resident, and the larger cells were skipped.
- **The `noynn` pass lifted the limit**, because the planner's CPU model assumes the default fusion's storage. At n = 16 and n = 20, which no default cell reached, a guard that stops a process at 3 GB showed 0.17 GB to 0.18 GB resident before the pass ran.

## Checks

[`checks/`](checks/) compares `ssy.py` with the authors' code at `bfbdab2`.

- **The authors' code doesn't run as it stands.** `ssy_wc_ratio.py` calls `rouwenhorst` without importing it, and imports `jax.config.config`, which JAX no longer has (it fails on 0.12.0.1 and on 0.4.36). [`ssy_reference.py`](checks/ssy_reference.py) runs it unmodified but for the missing import, under JAX 0.4.20, and writes [`ssy_reference.csv`](checks/ssy_reference.csv): their operator applied to a fixed w, and their two solvers' solutions, on two grids with a different size on each axis.
- **[`ssy_check.py`](checks/ssy_check.py) reads that file** ([`ssy.txt`](checks/ssy.txt) is its output). The operator agrees to 2e-16. Newton's method with autodiff, written as the authors wrote it, takes 5 steps to a solution within 2e-5 of theirs, and within 4e-3 of successive approximation's, which stops at a step of 1e-7 after 10,428 iterations. The Jacobian–vector product agrees with the whole Jacobian to 7e-16.
- **Fidelity:** float32 agrees with float64 to 1.3e-7 for the operator, at every size. For the product it is 1e-6 to 5e-5, and 2e-4 at n = 10, whose worst sample is a value of 4e-4 beside a typical 0.3, which differs by 8e-8.
