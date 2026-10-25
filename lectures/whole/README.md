# Whole lectures, as used

Every lecture in [lecture-jax](https://github.com/QuantEcon/lecture-jax), run as written in one fresh process each, timed cell by cell, on three devices. This is the first part of [#4](https://github.com/QuantEcon/benchmarks/issues/4): which lectures run faster on a CPU than on the GPU they are built on, and where each lecture's time goes.

Still to come under #4: the JAX lectures in lecture-python.myst, `sargent_surico` among them; float32 runs of the float64 lectures; `qe-wharton2`'s M5 Pro; and the T4 that the lectures' CI uses.

## Results

Lecture time is the sum of the cell times, compile included, as a reader or a CI build pays it: the median of three runs, in seconds. The card is the TITAN RTX at `0b:00.0` in `qe-wharton1-gpu`; the Threadripper 2990WX is the CPU beside it; the M6 is `qe-mini1`'s CPU. The last column is the cell that carries the most time on the slower of the card and the M6.

| Lecture | Precision, as written | TITAN RTX | Threadripper | M6 | Faster | Heaviest cell |
|---|---|---|---|---|---|---|
| `overborrowing` | float32 | 18.3 | 325.3 | 442.7 | card 24.3× | #31: 438.3 s on the M6, 4.6 s on the card |
| `wealth_dynamics_jax` | float32 | 53.4 | 761.3 | 391.4 | card 7.3× | #37: 54.8 s on the M6, 6.3 s on the card |
| `schelling_shocks` | float32 | 12.1 | 347.4 | 86.2 | card 7.1× | #7: 85.6 s on the M6, 7.5 s on the card |
| `opt_invest` | float64 | 14.0 | 358.4 | 68.7 | card 4.9× | #32: 62.3 s on the M6, 6.5 s on the card |
| `hopenhayn` | float64 | 18.9 | 134.7 | 45.5 | card 2.4× | #35: 37.1 s on the M6, 7.6 s on the card |
| `kesten_processes_jax` | float32 | 10.7 | 98.2 | 22.9 | card 2.1× | #9: 6.4 s on the M6, 1.6 s on the card |
| `opt_savings_2` | float64 | 12.8 | 154.9 | 25.8 | card 2.0× | #33: 22.2 s on the M6, 5.3 s on the card |
| `inventory_ssd` | float64 | 7.1 | 24.9 | 3.8 | M6 1.9× | #0: 2.0 s on the card, 0.5 s on the M6 |
| `schelling_numpy` | float32 | 3.3 | 3.2 | 1.5 | M6 2.2× | #9: 2.7 s on the card, 1.3 s on the M6 |
| `newtons_method` | float64 | 150.8 | 165.9 | 66.3 | M6 2.3× | #12: 122.5 s on the card, 59.1 s on the M6 |
| `markov_asset_jax` | float64 | 9.2 | 59.4 | 3.6 | M6 2.5× | #8: 2.2 s on the card, 0.1 s on the M6 |
| `jax_intro_jax` | float64 | 23.7 | 37.3 | 8.3 | M6 2.9× | #64: 11.2 s on the card, 0.2 s on the M6 |
| `lucas_model_jax` | float32 | 17.3 | 15.3 | 5.3 | M6 3.3× | #4: 6.7 s on the card, 2.3 s on the M6 |
| `schelling_python` | float32 | 18.8 | 18.7 | 5.4 | M6 3.5× | #7: 18.2 s on the card, 5.2 s on the M6 |
| `ifp_egm_jax` | float64 | 156.2 | 153.5 | 42.4 | M6 3.7× | #11: 71.1 s on the card, 20.7 s on the M6 |
| `inventory_dynamics_jax` | float32 | 8.9 | 12.0 | 2.2 | M6 4.1× | #6: 1.7 s on the card, 0.3 s on the M6 |
| `arellano_jax` | float64 | 22.3 | 17.0 | 5.4 | M6 4.1× | #22: 5.3 s on the card, 0.4 s on the M6 |
| `schelling_jax_parallel` | float32 | 5.0 | 5.0 | 1.1 | M6 4.4× | #7: 3.3 s on the card, 0.3 s on the M6 |
| `mle_jax` | float64 | 11.7 | 9.3 | 2.0 | M6 5.7× | #0: 2.7 s on the card, 0.9 s on the M6 |
| `opt_savings_1` | float64 | 35.7 | 40.5 | 6.0 | M6 5.9× | #8: 15.2 s on the card, 2.0 s on the M6 |
| `ifp_dl` | float32 | 27.9 | 19.5 | 3.7 | M6 7.6× | #26: 13.3 s on the card, 2.5 s on the M6 |
| `short_path_jax` | float32 | 3.5 | 1.9 | 0.4 | M6 8.1× | #2: 1.2 s on the card, 0.0 s on the M6 |
| `autodiff` | float32 | 12.3 | 8.6 | 1.4 | M6 8.7× | #6: 2.2 s on the card, 0.2 s on the M6 |
| `job_search` | float64 | 9.1 | 5.7 | 1.0 | M6 8.8× | #0: 1.8 s on the card, 0.5 s on the M6 |
| `jax_nn` | float32 | 45.1 | 23.4 | 4.5 | M6 10.1× | #9: 34.6 s on the card, 1.7 s on the M6 |
| `keras` | float32 | 35.5 | 13.9 | 2.2 | M6 16.0× | #18: 16.0 s on the card, 0.9 s on the M6 |
| `aiyagari_jax` | float64 | 91.1 | 27.4 | 5.1 | M6 17.8× | #31: 55.2 s on the card, 2.7 s on the M6 |
| `schelling_jax` | float32 | 38.5 | 8.6 | 1.1 | M6 36.5× | #9: 34.8 s on the card, 0.7 s on the M6 |
| **All 28** | | **873** | **2,851** | **1,256** | | |

`about_jax` has no code cells, so 28 of the 29 are listed. Every cell ran without error, except the cell in `jax_intro_jax` that the lecture tags `raises-exception` to show an error.

What the table shows:

- **The M6 runs 21 of the 28 lectures faster than the card, 20 of them by more than 2×.** The Threadripper is fastest on none.
- **The card wins on the seven lectures with one large vectorised computation:** simulating a large population (`wealth_dynamics_jax`, `schelling_shocks`, `kesten_processes_jax`) or solving a large dynamic program (`overborrowing`, `opt_invest`, `hopenhayn`, `opt_savings_2`). There it is 2× to 24× ahead of the M6.
- **Over the whole set the card is faster, 873 s to 1,256 s, because of two lectures.** `overborrowing` and `wealth_dynamics_jax` take 834 s of the M6's 1,256 and 72 s of the card's 873.
- **Where the card loses, the work is sequential or small.**
    - `schelling_jax` calls a small jitted function once per agent from a Python loop, so each step pays the card's launch and sync latency: 34.8 s in one cell, against 0.7 s on the M6.
    - `keras` trains a small network for 2,000 full-batch steps, and `jax_nn` does the same in its slowest cell. Each step is a few small kernels, the pattern that is latency-bound in the precision benchmark's `kalman` workload. `aiyagari_jax`'s slow steps are diagnosed in [`../aiyagari_jax/`](../aiyagari_jax/).
    - Small lectures take 3 s to 12 s on the card, where the M6 takes 0.4 s to 2 s. On the card the first JAX operation costs 1 s to 2 s while CUDA starts (`short_path_jax`'s `inf = jnp.inf`: 1.16 s, against under 0.04 s on the M6). Each first call of a jitted function also costs more: `job_search` spends 1.4 s to 1.6 s on each model it builds, against 0.1 s on the M6.
- **`schelling_numpy` and `schelling_python` barely use JAX,** so their card column measures the Threadripper beside it, and their gap to the M6 is one CPU against another.

`summarise_whole.py` prints a fuller table with each device's two slowest cells.

## Method

- **The lectures:** lecture-jax at `4b939ec` (2026-10-12), all 29 lectures except `intro`, `status`, `troubleshooting` and `zreferences`, listed in [`lectures.txt`](lectures.txt). Nothing in them was changed.
- **The runner:** [`../run_lecture.py`](../run_lecture.py) with `--dtype as-written`, so each lecture runs at the precision it sets. Every code cell runs through IPython in one fresh process with matplotlib's Agg backend, and is timed. Cells that only `!pip install` are skipped; their packages were installed beforehand.
- **Platforms:** `JAX_PLATFORMS=cuda` with `CUDA_VISIBLE_DEVICES=0` in PCI bus order for the card, and `JAX_PLATFORMS=cpu` for the CPUs, with JAX's defaults otherwise.
- **Three runs of each lecture on each device,** interleaved: every lecture once, then again, so a slow spell on a machine spreads across lectures rather than landing on one. The tables use medians. On the M6 the first run of a few lectures was up to 2× slower than the other two (`jax_nn`: 10.1 s against 4.4 s and 4.5 s).
- **An hour's cap on each lecture.** None reached it.
- **Environments:** one per machine, JAX 0.12.0.1 (`precision/constraints.txt`), with `jax[cuda13]` on `qe-wharton1-gpu`, and the lectures' imports. `environment.txt` beside the results lists every package.
- **`qe-mini1`'s agents and its animated wallpaper were paused** for its runs, from 16:45 to 17:49 AEDT on 2026-10-25, and resumed after. `state.txt` records the machine before, during and after. No GitHub runner process was on `qe-wharton1-gpu`.
- **Expected errors:** the machines ran `run_lecture.py` as it was before it recorded each cell's tags. The cells run are the same. `summarise_whole.py` reads the tags from the lecture's source with the current `code_cells`, so a cell tagged `raises-exception` counts as expected in these runs too.
- **No cross-device check of the outputs.** The runs didn't use `--record`, so a device that changed a lecture's answer would show only as an error. Each cell's last 3,000 characters of output are kept in the results, for a check later.

## Running it

[`lectures_set.sh`](lectures_set.sh) runs every lecture in `lectures.txt` from `~/bench/lectures-all`, which holds `.venv`, `lectures/<lecture>.md` and a copy of `run_lecture.py`:

```bash
./lectures_set.sh results 3 cuda cpu     # qe-wharton1-gpu: the card and the CPU beside it
./mac_lectures.sh                        # qe-mini1: pauses the agents, then ./lectures_set.sh results 3 cpu
python lectures/whole/summarise_whole.py lectures/results/*/2026-10-25/whole --lectures <directory of the lectures' .md files>
```

[`mac_lectures.sh`](mac_lectures.sh) pauses `qe-mini1`'s agents and wallpaper with SIGSTOP and resumes them on exit, even after a failure.

## Results files

`lectures/results/<host>/2026-10-25/whole/<lecture>/<platform>-as-written-<run>.json`, one per run: every cell's time, first line, success and tail of output, with the flags, the JAX version and device, and the `env` block from `precision/env_info.py`. Beside them are each machine's `environment.txt` and `lectures-set.log`, and on `qe-mini1`, `state.txt`.
