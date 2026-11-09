# A STREAM-style triad on qe-wharton2's M5 Pro, 2026-11-09

For QuantEcon/project-benchmarks-jax#47: [`problems/checks/stream_triad.py`](../../../../checks/stream_triad.py), as on [the M6](../../../qe-mini1/2026-11-09/triad/README.md), which says what it times, at 1, 2, 4, 8, 16 and 18 threads.

| | |
|---|---|
| Host | `qe-wharton2`: an Apple M5 Pro, with 18 cores (6 Super and 12 Performance) and 64 GB, on macOS 26.7.1 (25G241), on AC power with Low Power Mode off. Another account was logged in at the screen |
| Software | The benchmarks clone at `4ed7c58`, with its venv: Python 3.12.15, NumPy 2.5.4, Numba 0.68.1 with its `workqueue` threading layer, and llvmlite 0.50.0, as on the M6 |
| When | 2026-11-09, 05:28:25 to 05:28:29 UTC, with econ-on-ai-hardware's OK |
| Script | `session/wharton2_triad.sh`, with `NUMBA_NUM_THREADS=18`, run in `screen` under `caffeinate -i -s` |

**Two edits for publication:** in `state.txt`, software the owner's institution manages is shown as `<managed agent>`, as in 2026-11-06's state files. In `triad.jsonl` and `check-pool1.jsonl`, `host` is set to `qe-wharton2` in place of the machine's own name. **`state.txt`** has only the before and after states, since the run took 4 s.

## What it shows

| Threads | Best time | GB/s |
|---:|---:|---:|
| 1 | 0.0493 s | 130.8 |
| 2 | 0.0292 s | 220.5 |
| 4 | 0.0252 s | 255.5 |
| 8 | 0.0243 s | 264.7 |
| 16 | 0.0254 s | 254.1 |
| 18 | 0.0249 s | 258.4 |

- **Four threads take all the bandwidth the M5 Pro's memory delivers,** 255 to 265 GB/s, about twice the M6's. One thread gets half of that, about as much as on the M6.
- **The check:** with `NUMBA_NUM_THREADS=1`, so that Numba's whole pool is one thread and that thread first touched the arrays too, the triad gave 131.3 GB/s (`check-pool1.jsonl`), the same as the one-thread row.
