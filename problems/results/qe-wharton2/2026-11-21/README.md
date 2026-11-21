# Phase 2's four problem types on qe-wharton2's M5 Pro, 2026-11-21

For [#5](https://github.com/QuantEcon/benchmarks/issues/5)'s Phase 2: the same set as [the M6's](../../qe-mini1/2026-11-21/README.md), on a second Apple CPU, and in the same hours: P5, P2, P6 and P8 with its three companions, every size in both precisions, on the four CPU stacks. The rows are in [`../2026-11-21.jsonl`](../2026-11-21.jsonl), unlabelled, and `summary.md` tables them.

| | |
|---|---|
| Host | `qe-wharton2`: Apple M5 Pro, 18 cores (6 Super and 12 Performance), 64 GB, macOS 26.7.2 (25G332). A guest machine, on AC power, Low Power Mode off |
| Software | A fresh clone at `11c1c59`, #82's merge, with a venv at `problems/constraints.txt`: NumPy 2.5.4 (Accelerate), SciPy 1.18.1, Numba 0.68.1, JAX 0.12.0.1, Python 3.12.15 (`setup.txt`) |
| When | 2026-11-21, 03:22:22 to 04:40:20 UTC (14:22 to 15:40 AEDT), under the machine's shared lock, so nothing else was booked on it |
| Script | [`session/chain.sh`](session/chain.sh), in `screen` under `caffeinate -i -s`, which runs each step through [`session/step.sh`](session/step.sh) |

## What ran

`chain.sh` ran six steps in turn, and would have stopped at the first that failed:

| Step | What it does | When (UTC) |
|---|---|---|
| [`machine.sh`](session/machine.sh) | records the machine, and stops on battery or in Low Power Mode | 03:22:22 |
| [`tools.sh`](session/tools.sh) | a git from conda-forge, since `/usr/bin/git` there is the Xcode stub | 03:22:22 to 03:22:31 |
| [`setup.sh`](session/setup.sh) | the clone, its venv, and imports warmed twice | 03:22:31 to 03:22:54 |
| quick | `run_suite.py --quick`, a check; its rows are left out, and `run.log` keeps its lines | 03:22:54 to 03:24:40 |
| full | `run_suite.py --problems stationary,hpi,time_iter,mle,mle_hand,mle_onepass,mle_blocked --host qe-wharton2`, run `20261121T032440Z`, with the default limit of 900 s a cell | 03:24:40 to 04:40:20 |
| [`state.sh`](session/state.sh) | the machine's state, before and after the full run | 03:24:40 and 04:40:20 |

The full run's problems took the same stacks and cells as on the M6:

| Problem | Cells | When (UTC) |
|---|---:|---|
| P5 `stationary` | 88 | 03:24:40 to 03:41:00 |
| P2 `hpi`, float64 only | 44 | 03:41:00 to 03:43:09 |
| P6 `time_iter` | 80 | 03:43:09 to 04:23:09 |
| P8 `mle` | 96 | 04:23:09 to 04:30:34 |
| P8 `mle_hand` | 24 | 04:30:34 to 04:31:54 |
| P8 `mle_onepass` | 44 | 04:31:54 to 04:35:15 |
| P8 `mle_blocked` | 48 | 04:35:15 to 04:40:20 |

## Checks

- **All 424 cells ran,** status `ok`, and none came near its limit. The longest was P6's `numpy` at a grid of 50,000 in float32: 328 s for the process, with one warm run of 163 s.
- **Every answer agrees with NumPy's in float64** at the same size, within `spec.py`'s tolerances, P8's companions' with `mle`'s NumPy: `summary.md` lists no exceptions.
- **The iteration counts that differ, all in float32, are the M6's,** at every size: P5's `sparse` sweep, at its looser float32 tolerance; P8's NumPy one Newton step more in float32 at 3,000,000 and 10,000,000 observations; `jax-cpu` 19 steps at 10,000,000 in `mle` and `mle_hand`; and the Numba stacks of `mle_onepass` and `mle_blocked` float64's counts.

## The machine during the run

- **The chain recorded the machine at the start, and before and after the full run,** in `state.txt`, but not during it. Nothing else was booked on the machine, since the chain held its lock throughout.
- **`state.txt` is redacted** with the same rule as the 2026-11-06 run's: software the owner's institution manages is shown as `<managed agent>`, with its CPU figures as recorded, and the account's name and home folder are left out. The busiest of those agents took 6% to 12% of a core in its three samples, at the start, before the full run and after it.
- **The console's animated wallpaper was playing,** at 2% to 7% of a core with its video decoder at 1% to 2%, and wasn't paused. WindowServer took 22% to 31%, and a hardware monitor, `macmon`, 0.5% at the end.
- **`pmset -g therm` recorded no thermal or performance warning,** before or after, and the battery stayed at 80%, on AC and not charging.
