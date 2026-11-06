# `qe-wharton2`, 2026-11-06: the problem-type suite on an M5 Pro

This is the suite's four CPU backends on an Apple M5 Pro, every size in both precisions: the M5 Pro step QuantEcon/benchmarks#5 planned, beside the M6's run of 2026-11-02 (QuantEcon/benchmarks#25). It ran through [`session/chain.sh`](session/chain.sh), which ran each step through `session/step.sh`. The same session then ran QuantEcon/project-benchmarks-jax#48's probe, whose notes are with the precision runs.

| | |
|---|---|
| Host | `qe-wharton2`: an Apple M5 Pro, with 18 cores (6 Super and 12 Performance) and 64 GB, on macOS 26.7.1 (25G241), on AC power with Low Power Mode off (`machine.txt`) |
| Software | The checkout at `4ed7c58`, with its `.venv`: Python 3.12.15, JAX 0.12.0.1, Numba 0.68.1, NumPy 2.5.4 on Accelerate and SciPy 1.18.1, as each row's `env` records |
| When | 2026-11-06, 13:34:34 to 14:49:16 UTC, in `screen` under `caffeinate -i -s` |

## The runs

- **The machine,** 13:34:34 UTC: `session/machine.sh` recorded it, and would have stopped the chain on battery power or in Low Power Mode (`machine.txt`).
- **The set-up,** 13:34:34 to 13:34:48: `session/setup.sh` fast-forwarded the checkout to `main` at `4ed7c58`, and added Numba 0.68.1 and llvmlite 0.50.0 to the `.venv` at the suite's pins. Then it compiled the packages' bytecode and imported them twice, in 5.6 s and then 0.26 s, since this machine's security software checks new files. Its log is left out, since it's mostly git's list of the files the fast-forward changed.
- **The `--quick` check,** 13:34:49 to 13:35:46: 36 rows, all ok. They're left out of the `.jsonl` (`run_id` 20261106T133449Z), since the full run repeats their cells.
- **The full run,** 13:35:47 to 14:49:16, `run_id` 20261106T133547Z: every size in both precisions, 200 cells, all ok (`run.log`).
- **Each step's exit:** `step.sh` logged each step's exit status and wall time, and every step exited 0.

**`qe-wharton2` is a guest machine, so nothing on it was paused.**
- It ran on AC power, and macOS reported no thermal or performance warning.
- Its managed endpoint agents ran throughout. `machine.txt`, `state-before.txt` and `state-after.txt` record the busiest processes at the start, and before and after the full run. In them, the agents took at most about 15% of one of the 18 cores.
- In those records the managed software's names are redacted.

**The files:**
- `summary.md`: this host's tables, from `python problems/summarise.py problems/results/`;
- `machine.txt`, `state-before.txt` and `state-after.txt`;
- `run.log`: the full run's output;
- `session/`: the chain's scripts.

## What they show

- **Every cell ran, and every answer agreed** with NumPy's float64 answer at the same size. The only flags are P3's float32 iteration counts: 2,206 to 2,210 against float64's 2,190 to 2,192, as on the M6 and the Threadripper.
- **Against the M6's 2026-11-02 run** (`../../qe-mini1/2026-11-02.jsonl`), each device's fastest stack at each sweep's largest size, in float64, in seconds:

    | Sweep, largest size | M5 Pro, warm | M6, warm | M5 Pro, cold | M6, cold |
    |---|---|---|---|---|
    | P1 `vfi`, grid of 1,500 | 4.99, `jax-cpu` | 7.71, `jax-cpu` | 5.02, `jax-cpu` | 7.70, `jax-cpu` |
    | P3 `egm`, grid of 50,000 | 34.6, `numba-parallel` | 33.4, `numba-parallel` | 36.0, `numba-parallel` | 34.3, `numba-parallel` |
    | P4 `wealth`, 10,000,000 households | 5.41, `numba-parallel` | 10.3, `numba-parallel` | 5.11, `numba-parallel` | 10.0, `numba-parallel` |
    | P7 `kalman`, 10,000 filters | 1.57, `numba-parallel` | 2.28, `numba-parallel` | 2.54, `jax-cpu` | 2.62, `jax-cpu` |
    | P7 `kalman`, one filter of 200 states | 0.117, `numba-parallel` | 0.107, `numba` | 0.128, `numpy` | 0.110, `numpy` |

    - **So the M5 Pro is 1.45× to 1.90× as fast,** warm, on P1, P4 and P7's batch, where the work spreads across cores.
    - **It's about as fast on P3, and a little slower on one small filter,** where most of the work falls on one core. There the M6's core is 11% to 12% faster: serial Numba takes 41.2 s against 46.1 s on P3, and 0.107 s against 0.119 s on the filter.
