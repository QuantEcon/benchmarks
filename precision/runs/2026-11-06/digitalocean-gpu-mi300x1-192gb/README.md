# An MI300X under ROCm 10.1, 2026-11-06

This was one rented session, started from a laptop for QuantEcon/project-benchmarks-jax#32, on one AMD Instinct MI300X (`gfx942`). It ran the same sets as the MI350X the day before ([`precision/runs/2026-11-05/`](../../2026-11-05/README.md)): Phase 2's precision set, the problem-type suite and #34's reproducers. Three additions followed:
- the precision set's GPU cells without XLA's autotuner;
- a matrix product at n = 16,384;
- #38's TPU probe, as a non-TPU reference.

The droplet was destroyed once its results were copied off.

| | |
|---|---|
| Host | `digitalocean-gpu-mi300x1-192gb`: DigitalOcean's `gpu-mi300x1-192gb` droplet in atl1, on demand, with 20 vCPUs of an Intel Xeon Platinum 8568Y+ and 235 GiB. DigitalOcean listed it at 13:41 AEDT, after listing no MI300X all morning |
| Image | "ROCm 10.1 on Ubuntu 24.04" (`amddeveloperclou-rocm101`, 2026-10-07), the MI350X's: Ubuntu 24.04.5, kernel 6.8.0-146-generic |
| Card | AMD Instinct MI300X VF, a virtual function, `gfx942`, with amdgpu 6.19.14, ROCm 10.1.0 and AMD SMI 27.1.0. JAX's allocator gets 143.77 GiB by default, and 182.1 GiB at a memory fraction of 0.95 (`machine.txt`) |
| Software | JAX 0.12.0 with `jax-rocm10-plugin` and `jax-rocm10-pjrt` 0.12.0, from `constraints-rocm.txt` (`environment.txt`) |
| Harness | `5cd9d3d`, with `ROCM_MAJOR=10`, as on the MI350X |
| On the machine (UTC) | 02:42 to 03:41, creation to destruction |

**Where the results are:**
- **The precision set:** `results/digitalocean-gpu-mi300x1-192gb/2026-11-06.jsonl`, and `lectures/results/digitalocean-gpu-mi300x1-192gb/2026-11-06/`, which holds `machine.txt`, `environment.txt`, `precision-summary.md` and the lectures' `whole/`.
- **The problem types:** `problems/results/digitalocean-gpu-mi300x1-192gb/`, with [their own notes](../../../../problems/results/digitalocean-gpu-mi300x1-192gb/2026-11-06/README.md).
- **This folder:** the session's scripts and logs (`session/`), the reproducers and their output (`repro/`), and the probe's run (`tpu-probe/`).

**The image needed nothing added:** its ROCm 10.1 libraries are for `gfx942`, this card's target. The MI350X had needed AMD's `gfx950` packages.
- **The ROCm version file is elsewhere:** `/opt/rocm/.info/version` is missing, as on the MI350X, so both `run_by_hand.sh` scripts ran with `ROCM_MAJOR=10`.
- **A public notebook server:** the image's JupyterLab container ran as root on public port 8888, and was stopped. Then only sshd and the image's Caddy, on port 80, listened publicly.
- **Its firewall limits SSH,** so the session kept to one multiplexed connection.

## The runs

| Step | Start (UTC) | Wall | `run_id` | Rows |
|---|---|---|---|---|
| Precision, `--quick` | 02:57:10 | 159 s | 20261106T025725Z, and 20261106T025851Z for `highest` | Left out of the `.jsonl` |
| Problem types, `--quick` | 02:59:49 | 79 s | 20261106T025951Z | Left out |
| Precision, full | 03:01:08 | 990 s | 20261106T030111Z, and 20261106T031141Z for `highest` | 92 and 12, all `ok` |
| Problem types, full | 03:17:38 | 450 s | 20261106T031739Z | 50, all `ok` |
| Problem types, `highest` pass | 03:25:08 | 133 s | 20261106T032508Z | 22, all `ok` |
| Reproducers | 03:27:21 | 158 s | | `repro/` |
| Added: the precision set's GPU cells with `XLA_FLAGS=--xla_gpu_autotune_level=0`, labelled `autotune0` | 03:30:24 | 513 s | 20261106T033024Z | 46, all `ok` |
| Added: the matrix product at n = 16,384, the same way | 03:38:57 | 16 s | 20261106T033857Z | 2, all `ok` |
| Added: #38's TPU probe, standalone | 03:39:13 | 68 s | | `tpu-probe/` |

- **How each step ran:** through `session/step.sh`, whose log for the step ends with its exit status and wall time. Every step exited 0.
    - `session/chain34-mi300x.sh` ran the two checks, then `session/full.sh` once both had exited 0.
    - `session/extras-mi300x.sh` ran the additions.
    - `full.sh` and `extras-mi300x.sh` don't check a step's exit before starting the next, so each step's log is the record of how it ended.
- **The scripts in this folder** are kept as they ran, beside the logs they wrote.
- **The precision tarball** was re-packed after the additions. Its sha256 before and after is in `session/`. `extras-mi300x.sh` named it by the UTC date, as `precision/run_by_hand.sh` does, and both ran on 2026-11-06 UTC. So both hashes are of the precision run's `digitalocean-gpu-mi300x1-192gb-2026-11-06.tgz`.
- **How the reproducers ran:** through `repro/run_repro.sh`, which fetched ROCm/jax#895's and ROCm/jax#896's scripts from the issues' bodies at run time.
    - **The saved scripts:** the fetched scripts are saved beside it, and they're the MI350X's, byte for byte. [Its notes](../../2026-11-05/README.md) give their sha256. Neither body has changed since 2026-10-21, before both runs.
    - **To re-run them,** use the saved copies rather than fetching again, since an issue's body can change.
- **Both summaries were regenerated** once the `--quick` rows came out:
    - `precision-summary.md` has the `autotune0` rows, which the run's own summary predates;
    - `summary.md` has the `highest` tables.

## What they show

- **Every cell ran, and in float64 the card agrees with its CPU,** by at most:
    - 4.8e-15 on the matrix product;
    - 7.9e-16 on the Bellman steps;
    - 7.3e-8 on the Kalman filter.
- **At JAX's default precision, float32 matrix products are reduced precision: TF32's 10 fraction bits, truncated.**
    - **The precision suite:** the matrix product is about 7e-4 off the CPU at every size, against a few parts in a million under `highest`.
    - **P3:** its float32 answers are 0.25 to 0.26 off NumPy's float64 answer, after 494 to 901 iterations. Under `highest` they take 2,204 to 2,212 iterations and agree.
    - **The match:** those counts and errors are [`problems/checks/tf32_egm.py`](../../../../problems/checks/tf32_egm.py)'s truncated case: 629, 816, 878 and 913 iterations at 1,000 to 50,000 points, 0.25 to 0.26 off. The case rounded to nearest is the one that matched the NVIDIA cards.
    - **Why the matrix product is further off than TF32's:** truncation's errors all have the same sign, so they add up along a product rather than cancelling.
    - **Not the autotuner:** with it off, the error is the same, so the reduced precision is the default precision's, not the autotuner's choice of kernel.
    - **P7** loses accuracy within the check: its float32 answers are up to 7.1e-4 off at the default, against under 1e-6 with `highest`.
- **The float64 matrix product reaches 20.4 TFLOPS at n = 8,192,** 12% of AMD's 163.4 TFLOPS float64 matrix figure, and 17.6 TFLOPS at 16,384 without the autotuner. The MI350X reached 97% of its own figure.
    - In float32 it reaches 228 TFLOPS at the default, in reduced precision, and 119 under `highest`.
    - The stored Bellman step reaches 3,136 GB/s at 60,000, 59% of the card's 5.3 TB/s.
- **Without the autotuner:**
    - float32 compiles take about 1 s instead of 3 s to 6.5 s;
    - the float32 matrix product slows to 40 TFLOPS at 8,192;
    - the Kalman filter's warm speed drops from 6.4 to 3.9 thousand steps a second;
    - float64 compiles don't change: about 6.7 s for the matrix product and 19 s for the Kalman filter.
- **Its float64 compiles are slower than the MI350X's** on the same image and harness. A Kalman filter in the problem types takes 14.1 s to 14.9 s cold, against 2.8 s to 3.2 s. The host CPUs differ: 20 vCPUs of a Xeon 8568Y+ here, 24 of an EPYC 9575F there.
- **The reproducers found nothing wrong** (`repro/`), against the MI350X's results:

    | Reproducer | MI300X | MI350X |
    |---|---|---|
    | ROCm/jax#895's three-operand sort | 0 of 100 calls wrong | 19 of 100 |
    | `jnp.argsort(x, stable=True)` | 0 of 100 | 0 of 100 |
    | ROCm/jax#896's scatter-add, at the default and with either flag | 0 of 100 | 0 at the default, 3 with each flag |

    So both bugs look specific to `gfx950`, as their reports' MI300A, a `gfx942` part, suggested.
- **The five lectures** ran three times each, with every cell ok (`whole/lectures-set.log`).
- **The TPU probe** ran here as a non-TPU reference for #38's v6e. It shows float64 to IEEE precision, with subnormals kept. It's a diagnostic, and nothing from it is in `results/`.
