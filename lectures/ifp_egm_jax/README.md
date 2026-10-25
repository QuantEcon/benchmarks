# `ifp_egm_jax` in float32 and float64

The second float64 lecture project-compute asked for, beside [`aiyagari_jax`](../aiyagari_jax/). It tests a different kernel shape. The endogenous grid method interpolates the policy at s × 25 × 25 points and evaluates `c**(-γ)` there, with no dense choice grid.

The lecture is `lectures/ifp_egm_jax.md` from [lecture-jax](https://github.com/QuantEcon/lecture-jax) at `4b939ec` (2026-10-12). Nothing in it was changed, except the `jax_enable_x64` line in float32 runs. Its `!pip install --upgrade quantecon` cell was skipped; quantecon 0.13.1 was installed beforehand.

## What was run

| Script | What it measures |
|---|---|
| [`ifp_check.py`](ifp_check.py) | The lecture's JAX solve (`successive_approx_jax`) at the lecture's grid, cold and warm, with its own tolerance, cap and `print_skip`: iterations, convergence, time per iteration, the operator's warm time, and the policy. Three processes per configuration |
| [`ifp_scale.py`](ifp_scale.py) | The lecture's jitted EGM operator, warm, at savings grids from 200 (the lecture's) to 50,000, with the income grid at 25. Three processes per configuration |
| [`../run_lecture.py`](../run_lecture.py) | The whole lecture as used, in float64 only: three runs each on the card and the CPU |
| [`summarise_ifp.py`](summarise_ifp.py) | The tables and the fidelity check, from the results |
| [`ifp_set.sh`](ifp_set.sh), [`ifp_one.sh`](ifp_one.sh) | The drivers, as run inside `tmux` on the machine (paths are that machine's) |

The lecture's code is SHA-256 `e7a29a92f70cd9c74d6bb5c77762e49f8cf2c6a73e3fcaed38cd5a63decf4d67`.

**Why there is no whole-lecture float32 run.** The lecture also solves the model with Numba, on the CPU, as a cross-check. That half takes about 70 s in float64, and in float32 it ran for over 7 minutes before it was stopped. The JAX solve converges in float32 (`ifp_check.py`), so the stall is in the Numba half. That half runs the same on every JAX platform, so it says nothing about the card.

## Where and how

As for `aiyagari_jax`: `qe-wharton1-gpu` on 2026-10-25 (UTC), the TITAN RTX at `0b:00.0` and the Threadripper beside it, `jax[cuda13]` 0.12.0.1, with preallocation off. This lecture ran in its own environment, which adds `quantecon` 0.13.1 and `numba` 0.68.0. Its package list is `environment.txt` beside the results.

The same CPU runs were made on `qe-mini1`'s M6 with [`../mac_set.sh`](../mac_set.sh), with the agents paused.

## Results

In [`../results/qe-wharton1-gpu/2026-10-25/ifp_egm_jax/`](../results/qe-wharton1-gpu/2026-10-25/ifp_egm_jax/) and [`../results/qe-mini1/2026-10-25/ifp_egm_jax/`](../results/qe-mini1/2026-10-25/ifp_egm_jax/): `check/`, `scale/` and `as-used/`, one JSON file per process. To rebuild the tables from the repository root:

```bash
python lectures/ifp_egm_jax/summarise_ifp.py lectures/results/qe-wharton1-gpu/2026-10-25/ifp_egm_jax
```
