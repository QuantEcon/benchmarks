# P7's Kalman filters by filters and states on qe-wharton1-gpu's TITAN RTX, 2026-11-10

For [#58](https://github.com/QuantEcon/benchmarks/issues/58): where a GPU overtakes a CPU on Kalman filters as filters and states grow together. This is the TITAN RTX's half. [The M6's half](../../qe-mini1/2026-11-10/README.md) ran in the same hours. The rows are in [`../2026-11-10.jsonl`](../2026-11-10.jsonl), labelled `kalman-grid`, and `summary.md` tables them.

| | |
|---|---|
| Host | `qe-wharton1-gpu`: the TITAN RTX at `0b:00.0` (the suite's `--gpus 0`, in PCI order). The card at `43:00.0` drives the desktop and was idle throughout |
| Software | The runs checkout at `62df721`, whose harness is #60's (`0222630`), with its venv: JAX 0.12.0.1 with `jax-cuda13-plugin` 0.12.0.1, NVIDIA driver 595.99.02 (595.91.07 on 2026-11-02). `setup.txt` lists the CUDA libraries |
| When | 2026-11-10, 09:01:43 to 09:25:11 UTC, under project-econ-on-ai-hardware's machine lock, with both cards idle at the start |
| Script | [`session/titan_kalman58.sh`](session/titan_kalman58.sh), in `tmux` |

## What ran

`problems/run_suite.py --problems kalman --backends jax-gpu --gpus 0`, in both precisions:

| Part | Sweeps | Filters, or states | Cell limit | Run | When (UTC) |
|---|---|---|---|---|---|
| quick | `batch20`, `batch50`, `batch100` | 10 | 900 s | a check; its rows are left out, and `run.log` keeps its lines | 09:01:43 to 09:02:06 |
| full | `batch`, `state`, `batch20`, `batch50`, `batch100` | Phase 1's sizes for `batch` and `state`; 1 to 10,000 filters, with 300 and 3,000, for the new sweeps | 1,800 s | `20261110T090206Z` | 09:02:06 to 09:25:11 |

## Checks

- **All 66 cells ran,** status `ok`. The longest was 10,000 filters at 100 states in float64, 440 s for the process.
- **Every answer agrees with NumPy's in float64** at the same size, on the M6 in the same hours, within `spec.py`'s tolerances: `summary.md` lists no exceptions.
- **`batch` and `state` agree with 2026-11-02's** (#26) at the sizes where the work sets the time: 3.16 s against 3.15 at 10,000 filters in float64, and 0.524 s against 0.518 in float32. 14 of the 24 cells are within 5%.
- **The fixed cost of a step is up 5% to 10%.** The other ten are small cells, which take about 0.12 s to 0.17 s whatever their size: one filter at 2 to 200 states, and up to 100 filters. Those are 5% to 10% slower than on 2026-11-02, so the card's cost per step is about 0.14 ms now, against about 0.13. The driver is newer, and the desktop session had just restarted (GDM was stopped and started for an unrelated check at 08:45 to 08:49 UTC). Which of those matters isn't established.

## The machine during the run

- **The card's clock:** `gpu.csv` samples both cards every 10 s. `0b:00.0` ran at a median of 1,860 MHz while busy, at 81 °C at most, so it boosted as usual.
- **The other card:** `43:00.0`, which another check found held at its 1,350 MHz base clock under load, wasn't used.
- **CPU:** `state.txt` has the machine before and after. The Threadripper wasn't otherwise busy: the only processes above a few per cent were the desktop session's, starting when each `ssh` login did.
