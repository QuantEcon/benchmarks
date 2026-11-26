# P7's Kalman filters by filters and states on qe-wharton2's M5 Pro, 2026-11-10

For [#58](https://github.com/QuantEcon/benchmarks/issues/58): the same set as [the M6's](../../qe-mini1/2026-11-10/README.md), on a second Apple CPU, so that the crossover with [the TITAN RTX](../../qe-wharton1-gpu/2026-11-10/README.md) can be checked on two chips (Matt, 2026-11-11). The rows are in [`../2026-11-10.jsonl`](../2026-11-10.jsonl), labelled `kalman-grid`, and `summary.md` tables them.

| | |
|---|---|
| Host | `qe-wharton2`: Apple M5 Pro, 18 cores (6 Super and 12 Performance), 64 GB, macOS 26.7.1 (25G241). A guest machine, on AC power |
| Software | The runs checkout at `62df721`, whose harness is #60's (`0222630`), with its venv: NumPy 2.5.4 (Accelerate), Numba 0.68.1, JAX 0.12.0.1 (`setup.txt`), the packages of its 2026-11-06 run |
| When | 2026-11-10, 13:21:55 to 15:01:14 UTC (00:21 to 02:01 AEDT on 2026-11-11) |
| Script | [`session/m5pro_kalman58.sh`](session/m5pro_kalman58.sh), in `screen` under `caffeinate -ims`. It's the M6's script with this host's name, and everything it wrote went in a `.noindex` folder |

## What ran

The same four parts as on the M6, in the same order, in both precisions:

| Part | Sweeps | Filters, or states | Stacks | Cell limit | Run | When (UTC) |
|---|---|---|---|---|---|---|
| quick | `batch20`, `batch50`, `batch100` | 10 | all four | 900 s | a check; its rows are left out, and `run.log` keeps its lines | 13:21:55 to 13:22:52 |
| c | `batch`, `state` | as Phase 1 | all four | 900 s | `20261110T132252Z` | 13:22:52 to 13:32:25 |
| a | `batch20`, `batch50`, `batch100` | 1, 10, 100, 300 | all four | 1,800 s | `20261110T133225Z` | 13:32:25 to 13:43:18 |
| b | `batch20`, `batch50`, `batch100` | 1,000, 3,000, 10,000 | `numpy`, `numba-parallel`, `jax-cpu` | 1,800 s | `20261110T134318Z` | 13:43:18 to 15:01:14 |

## Checks

- **All 246 cells ran,** status `ok`. The longest was `numpy` at 10,000 filters and 100 states in float64, 458 s for the process.
- **Every answer agrees with NumPy's in float64** at the same size: `summary.md` lists no exceptions.
- **`batch` and `state` agree with 2026-11-06's** ([its run](../2026-11-06/README.md), the same packages) within 5% in 21 of the 24 cells. The other three, all `batch` in float64, are 7% to 13% faster now: 1, 100 and 1,000 filters.

## The machine during the run

- **Nobody was using it:** the owner's account was at the console, with no input for 4.6 days before the start. Nothing on it was paused.
- **The owner's institution's managed agents ran throughout,** shown as `<managed agent>` in `state.txt`. The busiest took 6% of a core at the median of the 30 s samples, and 40% at most.
- **Spotlight** (`mds_stores`, `mdworker_shared`, `mds`) used about 89 CPU-seconds in 1 h 39 min, in bursts up to 132% of a core every few minutes. The run's own files were in a `.noindex` folder, so this was indexing elsewhere. `mediaanalysisd` stayed under 3%.
    - **The 24 samples at half a core or more** fell in 15 cells. In 11 of them the warm runs agree within 2% of each other.
    - **Two `jax-cpu` cells drifted up** by 3.8% and 5.2% across their warm runs. `jax-cpu` wasn't the fastest stack at either.
    - **Four cells at 10,000 filters had one warm run each,** so they can't be checked this way.
- **`state.txt`** has the busiest processes every 30 s, redacted with the same rule as the 2026-11-06 run's. `pmset -g therm` recorded no thermal or performance warning, and the battery stayed at 80%, on AC and not charging.
