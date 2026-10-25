# `aiyagari_jax` in float32 and float64

project-compute asked for this measurement before a card purchase. The request was to run `aiyagari_jax` in float32 and float64 on `qe-wharton1-gpu`, timed in a fresh process with compile included. Its decision rule: about 2× slower in float64 would mean bandwidth binds, and much more than 2× would mean FP64 arithmetic does ([its research file](https://github.com/QuantEcon/project-compute/blob/main/research/2026-10-25-dgx-spark-for-jax-and-a-usd-20k-machine.md), §5).

The lecture is `lectures/aiyagari_jax.md` from [lecture-jax](https://github.com/QuantEcon/lecture-jax) at `4b939ec` (2026-10-12), SHA-256 `3ec2c8f04521cc9af6739d3950c2fa61b1bf15913566a5ee1e5019c41808cc95`. Nothing in it was changed, except the `jax_enable_x64` line in float32 runs.

## What was run

| Script | What it measures |
|---|---|
| [`../run_lecture.py`](../run_lecture.py) | The whole lecture, as used. Three runs each of the TITAN RTX and its host CPU, in float64 and float32, interleaved |
| [`aiyagari_scale.py`](aiyagari_scale.py) | The lecture's household operators, warm, at asset grids from 200 (the lecture's) to 10,000; three processes per configuration. It adds one variant the lecture doesn't have: the greedy step over a stored reward array, which reads memory where the lecture's step computes logs |
| [`hpi_check.py`](hpi_check.py) | Howard iterations per solve over the lecture's 50 capital values, by precision |
| [`bicgstab_check.py`](bicgstab_check.py) | Whether `get_value`'s bicgstab solve runs to its iteration cap, by precision |
| [`nan_check.py`](nan_check.py), [`bicgstab_jit_trace.py`](bicgstab_jit_trace.py) | Where float32 goes NaN on the card at a 5,000-point grid. At policy-iteration loop 2, JAX's bicgstab stalls, and at its 46th step `⟨r̂, q⟩` rounds to exactly zero, so `alpha` is infinite and `omega` NaN. Its breakdown test checks only for exact zeros, so the NaN is returned; the lecture's `while error > tol` then stops, because NaN > tol is false. Output in `checks/bicgstab_nan_trace.txt` |
| [`summarise_runs.py`](summarise_runs.py), [`summarise_scale.py`](summarise_scale.py), [`policy_gap.py`](policy_gap.py) | The tables and the fidelity checks, from the results |
| [`run_set.sh`](run_set.sh), [`scale_set.sh`](scale_set.sh) | The drivers, as run inside `tmux` on the machine (paths are that machine's) |

## Where and how

| | |
|---|---|
| Machine | `qe-wharton1-gpu`, 2026-10-25 (UTC). Nothing else ran on it, and no runner is registered there |
| Card | The TITAN RTX at `0b:00.0`: `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`. NVIDIA driver 595.91.07 (CUDA 13.2) |
| CPU | The Threadripper 2990WX beside it: `JAX_PLATFORMS=cpu`, with default NUMA placement |
| Software | `jax[cuda13]` 0.12.0.1 on Python 3.12.3, with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The full package list is `environment.txt` beside the results |

**The Mac.** The same CPU runs were made on `qe-mini1`, an M6 with 32 GB, with `jax` 0.12.0.1 on Python 3.12.15, using [`../mac_set.sh`](../mac_set.sh). Its results are under [`../results/qe-mini1/2026-10-25/aiyagari_jax/`](../results/qe-mini1/2026-10-25/aiyagari_jax/), and [`../compare_hosts.py`](../compare_hosts.py) sets the three devices side by side.

`env.git_commit` in each as-used result (`91baa22`) is the benchmarks checkout that supplied `env_info.py`, not these scripts. The scripts are as committed here, except for one change to `run_lecture.py` made after these runs: skipping `!pip install` cells. This lecture has no such cell.

## Reproducing a run

From the repository root, with the lecture fetched as in [`../README.md`](../README.md). On the card:

```bash
DEV="JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg"
env $DEV python lectures/run_lecture.py aiyagari_jax.md --dtype float64 --out cuda-float64-1.json \
    --env-info precision \
    --record "K_star" "K_vals.tolist()" "np.asarray(σ_star).ravel().tolist()" "str(household.a_grid.dtype)"
env $DEV python lectures/aiyagari_jax/aiyagari_scale.py aiyagari_jax.md --dtype float64 \
    --sizes 200,1000,2000,5000,10000 --out cuda-float64-1.json
```

- **`summarise_runs.py` reads all four `--record` values.**
- **For the CPU,** use `JAX_PLATFORMS=cpu` and drop the `CUDA_` variables.
- **For float32,** pass `--dtype float32`.
- **The full sets,** interleaved and repeated three times, are [`run_set.sh`](run_set.sh), [`scale_set.sh`](scale_set.sh) and [`../mac_set.sh`](../mac_set.sh).

## Results

In [`../results/qe-wharton1-gpu/2026-10-25/aiyagari_jax/`](../results/qe-wharton1-gpu/2026-10-25/aiyagari_jax/):

- `as-used/<platform>-<dtype>-<run>.json`: the whole-lecture runs. `set.log` adds the card's clocks, temperature, power and throttle reasons after each run.
- `scale/<platform>-<dtype>-<run>.json`: the operator timings, with `scale.log`.
- `checks/`: the two diagnostics' output, as printed.
- `environment.txt`: `uv pip freeze` of the environment.

To rebuild the tables from the repository root:

```bash
R=lectures/results/qe-wharton1-gpu/2026-10-25/aiyagari_jax
python lectures/aiyagari_jax/summarise_runs.py $R/as-used/*.json
python lectures/aiyagari_jax/summarise_scale.py $R/scale/*.json
python lectures/aiyagari_jax/policy_gap.py $R/scale/
```

What the numbers mean is written up in [project-benchmarks-jax](https://github.com/QuantEcon/project-benchmarks-jax/tree/main/findings).
