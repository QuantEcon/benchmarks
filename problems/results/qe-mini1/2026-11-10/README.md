# P7's Kalman filters by filters and states on qe-mini1's M6, 2026-11-10

For [#58](https://github.com/QuantEcon/benchmarks/issues/58): where a GPU overtakes a CPU on Kalman filters as filters and states grow together. This is the M6 half. [The TITAN RTX's half](../../qe-wharton1-gpu/2026-11-10/README.md) ran in the same hours. The rows are in [`../2026-11-10.jsonl`](../2026-11-10.jsonl), labelled `kalman-grid`, and `summary.md` tables them.

| | |
|---|---|
| Host | `qe-mini1`: Apple M6, 12 cores, 32 GB, macOS 27.2.1 (26B131), logged in at the screen |
| Software | The runs checkout at `62df721`, whose harness is #60's (`0222630`), with its venv: NumPy 2.5.4 (Accelerate), Numba 0.68.1, JAX 0.12.0.1 (`setup.txt`) |
| When | 2026-11-10, 09:01:02 to 10:50:11 UTC, in a quiet window agreed with the Mac's Claude sessions, which stayed running |
| Script | [`session/m6_kalman58.sh`](session/m6_kalman58.sh), in `tmux`. Everything it wrote went in a `.noindex` folder, so Spotlight and media analysis left the output alone ([why](../../../../precision/runs/2026-11-10/qe-mini1/noindex/README.md)) |

## What ran

Four invocations of `problems/run_suite.py --problems kalman --host qe-mini1`, in this order, in both precisions:

| Part | Sweeps | Filters, or states | Stacks | Cell limit | Run | When (UTC) |
|---|---|---|---|---|---|---|
| quick | `batch20`, `batch50`, `batch100` | 10 | all four | 900 s | a check; its rows are left out, and `run.log` keeps its lines | 09:01:02 to 09:01:53 |
| c | `batch`, `state` | as Phase 1: 1 to 10,000 filters; 2 to 200 states | all four | 900 s | `20261110T090153Z` | 09:01:53 to 09:11:11 |
| a | `batch20`, `batch50`, `batch100` | 1, 10, 100, 300 | all four | 1,800 s | `20261110T091112Z` | 09:11:11 to 09:21:28 |
| b | `batch20`, `batch50`, `batch100` | 1,000, 3,000, 10,000 | `numpy`, `numba-parallel`, `jax-cpu` | 1,800 s | `20261110T092128Z` | 09:21:28 to 10:50:11 |

The four stacks are `numpy`, `numba`, `numba-parallel` and `jax-cpu`. Serial Numba was left out from 1,000 filters, as #58 agreed: `numba-parallel` runs the same code, and was faster from 10 filters on at every state count.

## Checks

- **All 246 cells ran,** status `ok`, and none came near its limit. The longest was `jax-cpu` at 10,000 filters and 100 states in float64: 756 s for the process, with one warm run of 378 s.
- **Every answer agrees with NumPy's in float64** at the same size, within `spec.py`'s tolerances: `summary.md` lists no exceptions.
- **`batch` and `state` agree with 2026-11-02's** (#25, macOS 27.0.1). In 22 of the 24 cells, each size's fastest warm time is within 5% of that day's: 2.27 s against 2.28 at 10,000 filters, and 0.105 s against 0.107 at 200 states, in float64. The two others are faster now: 100 filters in float64 by 15% (0.0221 s against 0.0260), and 10 filters in float32 by 7%.

## The machine during the run

- **Background:** `samples.txt` has the media and Spotlight processes every 2 s, whenever at 1% or more of a core. Over the 1 h 49 min they used about 26 CPU-seconds in all, in four short spells:
    - `mediaanalysisd` at 61% to 92%, for a sample or two each, at 09:22:54, 09:27:10 and 10:23:08;
    - Spotlight (`mds_stores` up to 149%, `mds` up to 83%) from 09:46:33 to 09:46:44.
- **The cells they overlapped:**
    - three NumPy cells at 10,000 filters, whose warm runs agree within 1% (16.3–16.4 s, 17.1–17.2 s, 49.1–49.2 s);
    - `jax-cpu` at 10,000 filters and 100 states in float64, which had one warm run, so there's nothing to compare it with.
- **Also running:** the Claude sessions, a few per cent of a core each at the start, and `macmon`, a hardware monitor, at under 4%.
- **`state.txt`** has the busiest processes every 30 s. `pmset -g therm` recorded no thermal or performance warning.
