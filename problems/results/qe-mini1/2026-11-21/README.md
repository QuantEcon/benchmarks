# Phase 2's four problem types on qe-mini1's M6, 2026-11-21

For [#5](https://github.com/QuantEcon/benchmarks/issues/5)'s Phase 2: P5, P2, P6 and P8 with its three companions, every size in both precisions, on the four CPU stacks. It is the first measured run of these types. The same set ran the same day on [`qe-wharton2`'s M5 Pro](../../qe-wharton2/2026-11-21/README.md) and on [`qe-wharton1-gpu`](../../qe-wharton1-gpu/2026-11-21/README.md). The rows are in [`../2026-11-21.jsonl`](../2026-11-21.jsonl), unlabelled, and `summary.md` tables them.

| | |
|---|---|
| Host | `qe-mini1`: Apple M6, 12 cores (2 Super, 4 Performance and 6 Efficiency), 32 GB, macOS 27.2.1 (26B131), on AC power, Low Power Mode off |
| Software | The runs checkout at `11c1c59`, #82's merge, with its venv: NumPy 2.5.4 (Accelerate), SciPy 1.18.1, Numba 0.68.1, JAX 0.12.0.1, Python 3.12.15 (`setup.txt`) |
| When | 2026-11-21, 02:45:14 to 04:01:30 UTC, in a quiet window agreed with the Mac's Claude sessions, which stayed running |
| Script | [`session/m6_p2_run.sh`](session/m6_p2_run.sh), under `caffeinate -i -s`. Everything it wrote went in a `.noindex` folder, so Spotlight left the output alone ([#59](https://github.com/QuantEcon/benchmarks/issues/59)) |

## What ran

One invocation of `problems/run_suite.py --problems stationary,hpi,time_iter,mle,mle_hand,mle_onepass,mle_blocked --host qe-mini1`, run `20261121T024514Z`, with each problem's own stacks, sweeps and sizes from `spec.py` and the default limit of 900 s a cell:

| Problem | Stacks | Cells | When (UTC) |
|---|---|---:|---|
| P5 `stationary` | `numpy`, `numba`, `numba-parallel`, `jax-cpu` | 88 | 02:45:14 to 02:59:38 |
| P2 `hpi` | the same, float64 only | 44 | 02:59:38 to 03:01:37 |
| P6 `time_iter` | the same | 80 | 03:01:37 to 03:44:07 |
| P8 `mle` | the same | 96 | 03:44:07 to 03:51:34 |
| P8 `mle_hand` | `jax-cpu` | 24 | 03:51:34 to 03:53:38 |
| P8 `mle_onepass` | `numba`, `numba-parallel` | 44 | 03:53:38 to 03:56:52 |
| P8 `mle_blocked` | `numba`, `numba-parallel` | 48 | 03:56:52 to 04:01:30 |

Each time is when the problem's first cell started, to when the next problem's did. `run.log` is the suite's output.

## Checks

- **All 424 cells ran,** status `ok`, and none came near its limit. The longest was P6's `numpy` at a grid of 50,000 in float32: 316 s for the process, with one warm run of 159 s.
- **Every answer agrees with NumPy's in float64** at the same size, within `spec.py`'s tolerances, P8's companions' with `mle`'s NumPy: `summary.md` lists no exceptions.
- **The iteration counts that differ, all in float32,** are listed under their tables in `summary.md`:
    - P5's `sparse` sweep takes 157 to 192 iterations against float64's 266 to 299, since its float32 tolerance is 1e-6 rather than 1e-10.
    - P8's NumPy takes one Newton step more in float32 than in float64 at 3,000,000 and 10,000,000 observations: 18 against 17, and 17 against 16. There `jax-cpu` takes 19 at 10,000,000 in `mle` and `mle_hand`, while the Numba stacks of `mle_onepass` and `mle_blocked` take float64's 17 and 16.

## The machine during the run

- **Other processes:** `cputimes.txt` has every process's CPU time at the start and the end. In the 4,576 s of the run, all the others together used 474 CPU-seconds, about a tenth of one core, or under 1% of the 12:
    - the 16 Claude sessions 299 s, about 0.4% of a core each;
    - `mediaanalysisd` 31 s and Spotlight (`mds_stores` and `mds`) 15 s;
    - `tailscaled` 30 s, most of it at 03:19 and 03:20, during the copy below, and `macmon`, a hardware monitor, 16 s.
- **`samples.txt`** has the six busiest processes each minute. No Claude session passed 16% of a core in any sample, `tailscaled` reached 23%, and Spotlight 9%. `mediaanalysisd` never showed among the six, so its 31 s came in spells shorter than a minute. Photos analysis and the console's animated wallpaper weren't busy at the start, so the script paused neither.
- **Also in the window:** a 255 MB copy onto the Mac by `rsync` over ssh, for another of QuantEcon's projects, from 03:18:23 to 03:20:34, and the start of the M5 Pro's run by ssh, from 03:22:01 to 03:22:30. The cells they overlapped were P6's `numpy` and `numba` at a grid of 20,000 in float32, and `jax-cpu` beside them, whose warm runs agree within 0.3%, 1% and 0.4%. At about 03:21 to 03:23 the same project made a small `rsync` and a 2 MB local copy, and a Python edit of a text file took under a second, beside P6's `numpy` at 50,000 in float32, which had one warm run. Through the window, sessions polled the other machines by ssh every 20 s to 2 min.
- **`state.txt`** has the busiest processes and the power state before and after. `pmset -g therm` recorded no thermal or performance warning.
