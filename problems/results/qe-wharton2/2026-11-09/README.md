# One core's bandwidth and P1's serial Bellman steps on qe-wharton2's M5 Pro, 2026-11-09

For QuantEcon/project-benchmarks-jax#47: [`problems/checks/bw_probe.py`](../../../checks/bw_probe.py), as on [the M6](../../qe-mini1/2026-11-09/README.md), which says what each operation does.

| | |
|---|---|
| Host | `qe-wharton2`: an Apple M5 Pro, with 18 cores (6 Super and 12 Performance) and 64 GB, on macOS 26.7.1 (25G241), on AC power with Low Power Mode off. Another account was logged in at the screen |
| Software | The benchmarks clone at `4ed7c58`, whose `problems/vfi.py` is the same file as at `1874f1c`, with its venv: Python 3.12.15, NumPy 2.5.4, Numba 0.68.1 and llvmlite 0.50.0, as on the M6 |
| When | 2026-11-09, 04:39:36 to 04:39:39 UTC |
| Script | `session/wharton2_bw.sh`, run in `screen` under `caffeinate -i -s`, with every thread pool at one |

**Two edits for publication:** in `state.txt`, software the owner's institution manages is shown as `<managed agent>`, as in 2026-11-06's state files. In `bandwidth.jsonl`, `host` is set to `qe-wharton2` in place of the machine's own name. **One other process:** `XprotectService`, macOS's malware scan, took 21% of a core as the run started, scanning the newly copied probe.

## What it shows

| Operation | Best time | GB/s | The M6's GB/s |
|---|---:|---:|---:|
| `read` | 0.0240 s | 75.2 | 78.8 |
| `copy` | 0.0266 s | 135.4 | 121.8 |
| `numpy` | 0.1418 s | 38.1 | 41.8 |
| `numpy-out` | 0.1398 s | 38.6 | 41.5 |
| `numba` | 0.0949 s | 19.0 | 19.7 |
| `numba-evT` | 0.0946 s | 19.0 | 19.7 |

- **As on the M6,** the serial Numba step runs at a quarter of `read`'s rate, and allocation costs nothing.
- **The probe accounts for the suite's cells:** P1 at 1,500 takes 572 iterations in [2026-11-06's results](../2026-11-06.jsonl), whose warm medians are 85.06 s in `numpy` and 57.31 s in `numba`. The probe's steps are 95% of each iteration.
