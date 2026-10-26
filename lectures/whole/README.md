# Whole lectures, as used

Every lecture in [lecture-jax](https://github.com/QuantEcon/lecture-jax), run as written in one fresh process each, timed cell by cell, on four of QuantEcon's devices and on the T4 runner that the lectures' CI uses. This is the first part of [#4](https://github.com/QuantEcon/benchmarks/issues/4): which lectures run faster on a CPU than on the GPU they are built on, and where each lecture's time goes.

Still to come under #4: the JAX lectures in lecture-python.myst, `sargent_surico` among them.

## Results

Lecture time is the sum of the cell times, compile included, as a reader or a CI build pays it: the median of three runs, in seconds. The card is the TITAN RTX at `0b:00.0` in `qe-wharton1-gpu`; the Threadripper 2990WX is the CPU beside it; the M6 is `qe-mini1`'s CPU, and the M5 Pro is `qe-wharton2`'s. The last column is the cell that carries the most time on the slower of the card and the M6.

| Lecture | Precision, as written | TITAN RTX | Threadripper | M6 | M5 Pro | Faster | Heaviest cell |
|---|---|---|---|---|---|---|---|
| `overborrowing` | float32 | 18.3 | 325.3 | 442.7 | 66.0 | card 24.3× | #31: 438.3 s on the M6, 4.6 s on the card |
| `wealth_dynamics_jax` | float32 | 53.4 | 761.3 | 391.4 | 298.7 | card 7.3× | #37: 54.8 s on the M6, 6.3 s on the card |
| `schelling_shocks` | float32 | 12.1 | 347.4 | 86.2 | 78.0 | card 7.1× | #7: 85.6 s on the M6, 7.5 s on the card |
| `opt_invest` | float64 | 14.0 | 358.4 | 68.7 | 45.2 | card 4.9× | #32: 62.3 s on the M6, 6.5 s on the card |
| `hopenhayn` | float64 | 18.9 | 134.7 | 45.5 | 37.0 | card 2.4× | #35: 37.1 s on the M6, 7.6 s on the card |
| `kesten_processes_jax` | float32 | 10.7 | 98.2 | 22.9 | 25.2 | card 2.1× | #9: 6.4 s on the M6, 1.6 s on the card |
| `opt_savings_2` | float64 | 12.8 | 154.9 | 25.8 | 19.8 | card 2.0× | #33: 22.2 s on the M6, 5.3 s on the card |
| `inventory_ssd` | float64 | 7.1 | 24.9 | 3.8 | 3.6 | M6 1.9× | #0: 2.0 s on the card, 0.5 s on the M6 |
| `schelling_numpy` | float32 | 3.3 | 3.2 | 1.5 | 1.6 | M6 2.2× | #9: 2.7 s on the card, 1.3 s on the M6 |
| `newtons_method` | float64 | 150.8 | 165.9 | 66.3 | 69.4 | M6 2.3× | #12: 122.5 s on the card, 59.1 s on the M6 |
| `markov_asset_jax` | float64 | 9.2 | 59.4 | 3.6 | 3.9 | M6 2.5× | #8: 2.2 s on the card, 0.1 s on the M6 |
| `jax_intro_jax` | float64 | 23.7 | 37.3 | 8.3 | 8.4 | M6 2.9× | #64: 11.2 s on the card, 0.2 s on the M6 |
| `lucas_model_jax` | float32 | 17.3 | 15.3 | 5.3 | 5.6 | M6 3.3× | #4: 6.7 s on the card, 2.3 s on the M6 |
| `schelling_python` | float32 | 18.8 | 18.7 | 5.4 | 5.6 | M6 3.5× | #7: 18.2 s on the card, 5.2 s on the M6 |
| `ifp_egm_jax` | float64 | 156.2 | 153.5 | 42.4 | 48.8 | M6 3.7× | #11: 71.1 s on the card, 20.7 s on the M6 |
| `inventory_dynamics_jax` | float32 | 8.9 | 12.0 | 2.2 | 2.5 | M6 4.1× | #6: 1.7 s on the card, 0.3 s on the M6 |
| `arellano_jax` | float64 | 22.3 | 17.0 | 5.4 | 4.6 | M6 4.1× | #22: 5.3 s on the card, 0.4 s on the M6 |
| `schelling_jax_parallel` | float32 | 5.0 | 5.0 | 1.1 | 1.1 | M6 4.4× | #7: 3.3 s on the card, 0.3 s on the M6 |
| `mle_jax` | float64 | 11.7 | 9.3 | 2.0 | 2.3 | M6 5.7× | #0: 2.7 s on the card, 0.9 s on the M6 |
| `opt_savings_1` | float64 | 35.7 | 40.5 | 6.0 | 6.2 | M6 5.9× | #8: 15.2 s on the card, 2.0 s on the M6 |
| `ifp_dl` | float32 | 27.9 | 19.5 | 3.7 | 4.2 | M6 7.6× | #26: 13.3 s on the card, 2.5 s on the M6 |
| `short_path_jax` | float32 | 3.5 | 1.9 | 0.4 | 0.5 | M6 8.1× | #2: 1.2 s on the card, 0.0 s on the M6 |
| `autodiff` | float32 | 12.3 | 8.6 | 1.4 | 1.7 | M6 8.7× | #6: 2.2 s on the card, 0.2 s on the M6 |
| `job_search` | float64 | 9.1 | 5.7 | 1.0 | 1.3 | M6 8.8× | #0: 1.8 s on the card, 0.5 s on the M6 |
| `jax_nn` | float32 | 45.1 | 23.4 | 4.5 | 5.2 | M6 10.1× | #9: 34.6 s on the card, 1.7 s on the M6 |
| `keras` | float32 | 35.5 | 13.9 | 2.2 | 2.7 | M6 16.0× | #18: 16.0 s on the card, 0.9 s on the M6 |
| `aiyagari_jax` | float64 | 91.1 | 27.4 | 5.1 | 5.1 | M6 17.8× | #31: 55.2 s on the card, 2.7 s on the M6 |
| `schelling_jax` | float32 | 38.5 | 8.6 | 1.1 | 1.2 | M6 36.5× | #9: 34.8 s on the card, 0.7 s on the M6 |
| **All 28** | | **873** | **2,851** | **1,256** | **756** | | |

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
- **The M5 Pro runs the whole set fastest: 756 s, against the card's 873 s and the M6's 1,256 s.**
    - Across the 28 it is 5% slower than the M6 at the median. On the small lectures it is up to 1.3× slower (`job_search`).
    - On the large ones it is faster: `opt_invest` 1.5×, `wealth_dynamics_jax` and `opt_savings_2` 1.3×. It has 18 cores to the M6's 12.
    - The card still wins the same seven lectures, but by 1.5× to 6.4× rather than 2× to 24×.
- **The M6's `overborrowing` time is mostly swap.** Under `/usr/bin/time -l` on `qe-wharton2`, the lecture peaks at a 35.0 GB footprint ([`checks/overborrowing-memory.txt`](../results/qe-wharton2/2026-10-25/checks/overborrowing-memory.txt)), more than the M6's 32 GB. Its cell 31 takes 426 s to 459 s on the M6 and 62 s on the M5 Pro, though the two are otherwise close. So the card's 24.3× lead there is over the M6's memory; over the M5 Pro it is 3.6×.

`summarise_whole.py` prints a fuller table with each device's two slowest cells.

## The CI runner: the T4 and its own CPU

The lectures' CI builds on RunsOn's `g4dn.2xlarge`: a T4, and a Xeon Platinum 8259CL with 4 cores, 8 threads and 32 GB. The same set ran there three times on each, as above. The question is the one a build faces: on this runner, which lectures would build faster with `JAX_PLATFORMS=cpu`? Times are medians in seconds. The last column is the cell that carries the most time on the slower of the two.

| Lecture | Precision, as written | T4 | Xeon | Faster | Heaviest cell |
|---|---|---|---|---|---|
| `overborrowing` | float32 | 20.3 | out of memory | T4 | #31: 8.4 s on the T4; on the Xeon it asks for 33.4 GB |
| `schelling_shocks` | float32 | 19.2 | 462.6 | T4 24.1× | #7: 460.1 s on the Xeon, 16.0 s on the T4 |
| `opt_invest` | float64 | 17.8 | 194.1 | T4 10.9× | #32: 174.8 s on the Xeon, 10.5 s on the T4 |
| `wealth_dynamics_jax` | float32 | 167.7 | 1,369.3 | T4 8.2× | #40: 161.0 s on the Xeon, 13.8 s on the T4 |
| `markov_asset_jax` | float64 | 7.3 | 55.5 | T4 7.7× | #36: 50.3 s on the Xeon, 0.6 s on the T4 |
| `kesten_processes_jax` | float32 | 8.0 | 57.5 | T4 7.2× | #9: 12.5 s on the Xeon, 1.3 s on the T4 |
| `hopenhayn` | float64 | 18.5 | 124.7 | T4 6.7× | #35: 99.6 s on the Xeon, 9.2 s on the T4 |
| `opt_savings_2` | float64 | 16.1 | 67.7 | T4 4.2× | #33: 56.7 s on the Xeon, 9.2 s on the T4 |
| `jax_intro_jax` | float64 | 16.7 | 42.5 | T4 2.6× | #83: 11.4 s on the Xeon, 1.5 s on the T4 |
| `inventory_ssd` | float64 | 6.8 | 16.5 | T4 2.4× | #11: 6.5 s on the Xeon, 1.2 s on the T4 |
| `arellano_jax` | float64 | 15.9 | 31.7 | T4 2.0× | #15: 8.8 s on the Xeon, 2.9 s on the T4 |
| `schelling_jax_parallel` | float32 | 3.7 | 4.5 | T4 1.2× | #8: 2.2 s on the Xeon, 0.7 s on the T4 |
| `inventory_dynamics_jax` | float32 | 6.6 | 7.1 | T4 1.1× | #16: 1.2 s on the Xeon, 0.8 s on the T4 |
| `newtons_method` | float64 | 178.7 | 187.6 | T4 1.0× | #12: 156.2 s on the Xeon, 154.9 s on the T4 |
| `opt_savings_1` | float64 | 45.4 | 46.3 | T4 1.0× | #9: 20.1 s on the Xeon, 20.1 s on the T4 |
| `schelling_numpy` | float32 | 2.7 | 2.7 | Xeon 1.0× | #9: 2.2 s on the T4, 2.2 s on the Xeon |
| `schelling_python` | float32 | 18.2 | 18.1 | Xeon 1.0× | #7: 17.7 s on the T4, 17.6 s on the Xeon |
| `ifp_egm_jax` | float64 | 139.5 | 137.1 | Xeon 1.0× | #11: 66.1 s on the T4, 66.0 s on the Xeon |
| `lucas_model_jax` | float32 | 16.2 | 14.9 | Xeon 1.1× | #4: 6.3 s on the T4, 6.3 s on the Xeon |
| `autodiff` | float32 | 8.7 | 6.7 | Xeon 1.3× | #6: 1.7 s on the T4, 1.3 s on the Xeon |
| `job_search` | float64 | 6.0 | 4.4 | Xeon 1.4× | #0: 1.8 s on the T4, 1.8 s on the Xeon |
| `mle_jax` | float64 | 9.7 | 6.7 | Xeon 1.4× | #0: 2.4 s on the T4, 2.4 s on the Xeon |
| `short_path_jax` | float32 | 2.4 | 1.5 | Xeon 1.6× | #2: 0.6 s on the T4, 0.2 s on the Xeon |
| `jax_nn` | float32 | 32.7 | 18.9 | Xeon 1.7× | #9: 23.6 s on the T4, 10.6 s on the Xeon |
| `aiyagari_jax` | float64 | 66.1 | 36.7 | Xeon 1.8× | #31: 40.2 s on the T4, 16.4 s on the Xeon |
| `keras` | float32 | 26.6 | 13.0 | Xeon 2.0× | #18: 12.1 s on the T4, 5.8 s on the Xeon |
| `ifp_dl` | float32 | 27.0 | 11.8 | Xeon 2.3× | #26: 14.5 s on the T4, 7.1 s on the Xeon |
| `schelling_jax` | float32 | 17.2 | 5.7 | Xeon 3.0× | #9: 14.5 s on the T4, 3.9 s on the Xeon |
| **All 28** | | **922** | **2,946 for 27** | | |

What the table shows:

- **Precision doesn't decide which is faster.** Of the 13 float64 lectures, the T4 is clearly faster on 7, by 2.0× to 10.9×, and the CPU on 3, by 1.4× to 1.8×; the other 3 are within 5%. Across both precisions, the CPU wins where the work is small or sequential: training loops (`keras`, `jax_nn`, `ifp_dl`), a jitted function called once per agent from Python (`schelling_jax`), and small models (`job_search`, `mle_jax`, `short_path_jax`).
- **Building each lecture on the faster of the two would save about 95 s of the T4's 922 s,** if the lectures ran one after another. Five lectures carry 84 s of it: `aiyagari_jax`, `ifp_dl`, `jax_nn`, `keras` and `schelling_jax`.
- **Building everything on the CPU would take 3.3× as long, and `overborrowing` would fail.** Its cell 31 asks for 33.4 GB, more than the runner's 32 GB, though the T4 runs it in 8.4 s. Without it, the CPU takes 2,946 s against the T4's 901 s, and four lectures carry 2,151 s of that: `wealth_dynamics_jax`, `schelling_shocks`, `opt_invest` and `hopenhayn`.
- **The T4 and the TITAN RTX take about the same time over the set,** 922 s and 873 s. The TITAN is faster on 7 lectures, the large simulations and dynamic programs among them (`wealth_dynamics_jax`, 53 s against 168 s), and the T4 on the other 21, by up to 2.2× (`schelling_jax`). These runs don't separate why.
- **These runs are sequential.** A build executes several notebooks at once, sharing the one T4 or the 8 threads, which this doesn't measure.

## float32 against float64

The 13 lectures that set `jax_enable_x64`, listed in [`lectures-float64.txt`](lectures-float64.txt), each ran in float64 and then in float32, three times on each device. The table gives float64's median time over float32's, so above 1× float32 is faster. The last column says what float32 changes in what the lecture prints, from [`summarise_precision.py`](summarise_precision.py).

| Lecture | TITAN RTX | Threadripper | M6 | M5 Pro | What float32 changes |
|---|---|---|---|---|---|
| `hopenhayn` | 1.21× | 0.97× | 1.81× | 1.37× | The equilibrium price comes out 12% higher (1.53 against 1.36) |
| `arellano_jax` | 1.19× | 1.33× | 1.65× | 1.49× | 3 cells print different numbers |
| `inventory_ssd` | 1.09× | 1.56× | 0.91× | 1.08× | Nothing printed changes |
| `jax_intro_jax` | 1.07× | 1.23× | 1.30× | 1.24× | 11 cells print different numbers |
| `opt_savings_1` | 1.04× | 1.04× | 1.10× | 1.09× | Nothing printed changes |
| `job_search` | 1.00× | 1.00× | 1.02× | 1.01× | Nothing printed changes |
| `newtons_method` | 1.00× | 1.02× | 1.01× | 1.01× | 3 to 5 cells print different numbers |
| `mle_jax` | 0.89× | 0.91× | 0.94× | 0.93× | 2 cells print different numbers |
| `opt_invest` | 0.90× | 0.66× | 0.74× | 0.64× | Policy iteration never settles, and runs to its 250-loop cap |
| `opt_savings_2` | 0.92× | 0.08× | 0.51× | 0.45× | Policy iteration never settles, and runs to its 250-loop cap |
| `ifp_egm_jax` | 0.34× | 0.34× | 0.42× | 0.41× | The Numba solve takes 2.5× to 3× as long, in about the same number of iterations |
| `aiyagari_jax` | 0.08× | 0.10× | 0.12× | 0.13× | Policy iteration runs to its 10,000-loop cap at some capital values |
| `markov_asset_jax` | fails | fails | fails | fails | Its stability check fails, in every run on every device |
| **The 12 that ran in both,** float64 → float32 | **556 → 1,850 s** | **1,140 → 3,694 s** | **278 → 397 s** | **252 → 390 s** | |

What the table shows:

- **float32 is not a speed-up on the card for these lectures.** The TITAN RTX gains at most 1.21× (`hopenhayn`), though float32 is 24.5× faster for it on dense matmul in the precision benchmark. At the lectures' sizes the card isn't limited by float64 arithmetic.
- **Over the set, float32 is slower on every device:** 3.3× on the card and 1.4× on the M6. Where it is slower, a loop that stops at a tolerance takes longer or never stops, or code outside JAX slows down.
    - In `opt_invest` and `opt_savings_2`, policy iteration counts the policy points that change. In float64 that count reaches 0 after 10 to 12 loops; in float32 it stays at 1, so the loop runs to its cap of 250. That costs the card little. On the Threadripper, `opt_savings_2` takes 13× as long, mostly in cell #33, which reruns the solvers for 15 values of m: 1,867 s against 137 s.
    - In `aiyagari_jax`, policy iteration runs to its 10,000-loop cap at 2 of the lecture's 50 capital values ([`hpi_check.txt`](../results/qe-wharton1-gpu/2026-10-25/aiyagari_jax/checks/hpi_check.txt), diagnosed in [`../aiyagari_jax/`](../aiyagari_jax/)). Its sweep over β slows most: cell #36 takes 714 s on the card in float32, against 24 s.
    - In `ifp_egm_jax`, the Numba version of the solve converges in about the same number of iterations (2,203 against 2,192 on `qe-wharton1-gpu`), but takes 2.5× as long on the Macs and 3× on the Threadripper. Why is undiagnosed.
- **`markov_asset_jax` fails in float32.** Cell #15 asserts that the model's spectral radius is below 1, using a 15-step power iteration. In float32 that estimate comes out at 1.026, against 0.905 in float64; the exact radius is 0.994 ([`checks/markov_asset-spectral-radius.txt`](../results/qe-wharton2/2026-10-25/checks/markov_asset-spectral-radius.txt)).
- **Mostly the same cells change on every device.** Where a lecture's printed numbers move in float32, the card, the Threadripper and both Macs flag largely the same cells.

A flagged cell marks where to look, not an error: the comparison counts every number a cell prints that is the same in all three float64 runs, convergence diagnostics among them. `summarise_precision.py` skips cells that run a shell command, such as `!nvidia-smi`, which print the machine's state.

## Method

- **The lectures:** lecture-jax at `4b939ec` (2026-10-12), all 29 lectures except `intro`, `status`, `troubleshooting` and `zreferences`, listed in [`lectures.txt`](lectures.txt). Nothing in them was changed.
- **The runner:** [`../run_lecture.py`](../run_lecture.py) with `--dtype as-written`, so each lecture runs at the precision it sets. Every code cell runs through IPython in one fresh process with matplotlib's Agg backend, and is timed. Cells that only `!pip install` are skipped; their packages were installed beforehand.
- **Platforms:** `JAX_PLATFORMS=cuda` with `CUDA_VISIBLE_DEVICES=0` in PCI bus order for the cards, and `JAX_PLATFORMS=cpu` for the CPUs, with JAX's defaults otherwise.
- **`qe-wharton2`:** `main` at `736a9ee`, on 2026-10-25 from 22:06 to 22:44 UTC, with the M6's exact packages and the current `run_lecture.py`. Before the runs the new environment's bytecode was compiled and each package imported once: on this machine the first imports took 48 s, against 1.6 s after, while its security software checked the new files.
- **The T4 runner:** the workflow's `run_lectures` job (#10) at `8493bbf`, which is `main` with that job added, on 2026-10-25 from 09:37 to 12:55 UTC. Each lecture ran on the T4 and then on the Xeon before the next.
- **Three runs of each lecture on each device,** interleaved: every lecture once, then again, so a slow spell on a machine spreads across lectures rather than landing on one. The tables use medians. On the M6 the first run of a few lectures was up to 2× slower than the other two (`jax_nn`: 10.1 s against 4.4 s and 4.5 s).
- **An hour's cap on each lecture.** None reached it.
- **Environments:** one per machine, JAX 0.12.0.1 (`precision/constraints.txt`), with `jax[cuda13]` on `qe-wharton1-gpu`, and the lectures' imports. The T4 runner installed the TITAN RTX run's exact packages from its `environment.txt`. `environment.txt` beside the results lists every package.
- **`qe-mini1`'s agents and its animated wallpaper were paused** for its runs, from 16:45 to 17:49 AEDT on 2026-10-25, and resumed after. `state.txt` records the machine before, during and after. No GitHub runner process was on `qe-wharton1-gpu`.
- **`qe-wharton2` is a guest machine, so nothing on it was paused.** Its console had been idle for 26 hours and stayed idle, it ran on AC power with the lid open, and macOS reported no thermal or performance warning. Its managed endpoint agents ran throughout, at most about 14% of one of its 18 cores. `state.txt` samples it every ten minutes, with the console user's name and the managed software's names redacted.
- **Expected errors:** `qe-wharton1-gpu` and `qe-mini1` ran `run_lecture.py` as it was before it recorded each cell's tags; the T4 runner and `qe-wharton2` ran the current one. The cells run are the same. `summarise_whole.py` reads the tags from the lecture's source with the current `code_cells`, so a cell tagged `raises-exception` counts as expected in these runs too.
- **The float32 pass:** [`lectures_set.sh`](lectures_set.sh) at `e5b3f8f` with `DTYPES="float64 float32"` and `LIST=lectures-float64.txt`, so each lecture runs with `--dtype float64` and then `--dtype float32` on each device before the next. On `qe-mini1` from 23:17 to 23:51 UTC on 2026-10-25, with its agents paused; on `qe-wharton2` from 23:16 to 23:49 UTC, with nothing paused; and on `qe-wharton1-gpu`, card and CPU, from 23:15 UTC to 05:25 UTC on 2026-10-26, with both cards idle at the start and the end. `float32-pass-state.txt` and `lectures-set-float32.log` beside each host's results record the runs.
- **No cross-device check of the outputs.** The runs didn't use `--record`, so a device that changed a lecture's answer would show only as an error. Each cell's last 3,000 characters of output are kept in the results, for a check later.

## Running it

[`lectures_set.sh`](lectures_set.sh) runs every lecture in `lectures.txt` from `~/bench/lectures-all`, which holds `.venv`, `lectures/<lecture>.md` and a copy of `run_lecture.py`:

```bash
./lectures_set.sh results 3 cuda cpu     # qe-wharton1-gpu: the card and the CPU beside it
./mac_lectures.sh                        # qe-mini1: pauses the agents, then ./lectures_set.sh results 3 cpu
./lectures_set.sh results 3 cpu          # qe-wharton2, from precision/runs/2026-10-25/wharton2_run.sh
python lectures/whole/summarise_whole.py lectures/results/*/2026-10-25/whole --lectures <directory of the lectures' .md files>
```

[`mac_lectures.sh`](mac_lectures.sh) pauses `qe-mini1`'s agents and wallpaper with SIGSTOP and resumes them on exit, even after a failure.

On RunsOn, the workflow's `run_lectures` job runs the same set three times on the GPU and its host CPU of `lectures_instance`, by default the T4's `g4dn.2xlarge` that the lectures' CI uses. It installs the exact packages of the TITAN RTX run, from that run's `environment.txt`, and uploads the results as an artifact. It stops after five hours.

## Results files

`lectures/results/<host>/2026-10-25/whole/<lecture>/<platform>-as-written-<run>.json`, one per run: every cell's time, first line, success and tail of output, with the flags, the JAX version and device, and the `env` block from `precision/env_info.py`. Beside them are each machine's `environment.txt` and `lectures-set.log`, and on `qe-mini1` and `qe-wharton2`, `state.txt`. `qe-wharton2`'s memory check is in `checks/` beside its `whole/`, with the `markov_asset_jax` check. The float32 pass's runs are `<platform>-float64-<run>.json` and `<platform>-float32-<run>.json`, beside the as-written ones.
