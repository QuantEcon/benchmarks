# An H200, 2026-11-06

This was one rented session, started from a laptop for QuantEcon/project-benchmarks-jax#37, on one NVIDIA H200.
- **What ran:** the problem-type suite, then Phase 2's precision set, which @mmcky added while the droplet was up.
- **Afterwards:** the droplet was destroyed once its results were copied off.

| | |
|---|---|
| Host | `digitalocean-gpu-h200x1-141gb`: DigitalOcean's `gpu-h200x1-141gb` droplet in atl1, on demand, with 24 vCPUs of an Intel Xeon Platinum 8592+ and 235 GiB. It was the first on-demand GPU size DigitalOcean listed that day, after none had listed a region from 09:52 AEDT |
| Image | DigitalOcean's "NVIDIA AI/ML Ready Image" (`gpu-h100x1-base`, 2026-10-30): Ubuntu 24.04.5, kernel 6.8.0-147-generic |
| Card | NVIDIA H200, 143,771 MiB, power limit 700 W, driver 580.196.07 with CUDA 13.0. JAX's allocator gets 104.85 GiB by default, and 132.81 GiB at a memory fraction of 0.95 (`machine.txt`) |
| Software | JAX 0.12.0.1 with `jax[cuda13]`, on Python 3.12.3, so the rows carry no label (`environment.txt`) |
| Harness | `5cd9d3d` |
| On the machine (UTC) | 01:24 to 01:49, creation to destruction |

**Where the results are:**
- **The problem types:** `problems/results/digitalocean-gpu-h200x1-141gb/`, with [their own notes](../../../problems/results/digitalocean-gpu-h200x1-141gb/2026-11-06/README.md).
- **The precision set:** `results/digitalocean-gpu-h200x1-141gb/2026-11-06.jsonl`, and `lectures/results/digitalocean-gpu-h200x1-141gb/2026-11-06/`, which holds `machine.txt`, `environment.txt`, `precision-summary.md` and the lectures' `whole/`.
- **This folder's `digitalocean-gpu-h200x1-141gb/session/`:** the session's scripts and each step's log.

**The droplet** needed nothing stopped. Only sshd listened on a public port, ufw was inactive and no containers ran.

## The runs

| Step | Start (UTC) | Wall | `run_id` | Rows |
|---|---|---|---|---|
| Problem types, `--quick` | 01:26:18 | 44 s | 20261106T012644Z | Left out of the `.jsonl` |
| Problem types, full | 01:27:29 | 121 s | 20261106T012730Z | 50, all `ok` |
| Problem types, `highest` pass | 01:29:31 | 59 s | 20261106T012931Z | 22, all `ok` |
| Precision, `--quick` | 01:31:43 | 82 s | 20261106T013148Z, and 20261106T013228Z for `highest` | Left out |
| Precision, full | 01:33:48 | 900 s | 20261106T013351Z, and 20261106T014402Z for `highest` | 92 and 12, all `ok` |

- **How each step ran:** through `session/step.sh`, whose log for the step ends with its exit status and wall time. Every step exited 0.
- **The chain:** `session/chain37.sh` chained the problem types' full run and their `highest` pass, `session/highest.sh`, as on the H100.
- **The `--quick` rows are left out,** because the full runs repeat their cells.

## What they show

- **The problem types:** TF32 as on the H100 and the RTX PRO 6000, and every other answer agreed. Their notes have the rest.
- **The precision set,** beside Phase 2's H100 (`results/lambda-gpu_1x_h100_sxm5/2026-10-26.jsonl`), in float64:

    | Precision cell | H200 | H100 |
    |---|---:|---:|
    | Matrix product, n = 8,192, TFLOPS | 62.5 | 63.9 |
    | Stored Bellman step, n = 60,000, GB/s | 4,162 | 3,068 |
    | Recomputing Bellman step, n = 60,000, Gcells/s | 214 | 217 |
    | Kalman filter, 100,000 steps, thousand steps a second | 11.5 | 11.1 |

    So the H200 has the H100's float64 arithmetic, with 36% more of the bandwidth the stored step reaches.
- **Every precision cell ran, and agrees with the droplet's CPU** in float64, by at most:
    - 5.6e-15 on the matrix product;
    - 7.9e-16 on the Bellman steps;
    - 8.7e-7 on the Kalman filter.
- **At the default, float32 matrix products run in TF32.** They're 2.9e-5 to 6.5e-5 off the CPU's, and the Kalman filter is up to 5.5e-2 off.
- **The recomputing step compiles slowly in float64:** 84 s at 40,000 and 82 s at 60,000, against 158 s and 155 s on Phase 2's H100 under CUDA 12.
    - So the slow float64 compiles that Phase 2 found on Hopper recur here under CUDA 13, in about half the time.
    - The 2026-11-01 run notes traced the same effect on Blackwell to XLA's autotuner.
- **`aiyagari_jax` in float32 at a grid of 5,000 stops after 2 iterations,** with a NaN value sum (`aiyagari-summary.md`). Phase 2's H100 did the same, so it isn't new.
- **The five lectures** ran three times each, with every cell ok (`whole/lectures-set.log`).
