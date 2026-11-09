# A STREAM-style triad on qe-mini1's M6, 2026-11-09

For QuantEcon/project-benchmarks-jax#47, at project-compute's request: what the M6's memory delivers to one thread and to all 12 cores, to set beside the problem types' GB/s. The script is [`problems/checks/stream_triad.py`](../../../../checks/stream_triad.py) (3,320 bytes, sha256 `e73525d3…`), the copy #47's AWS instances run. It times `a = b + 3c` over three float64 arrays of 2^28 elements, 2 GiB each, at 1, 2, 4, 8 and 12 threads, each once to warm up and then ten times. It counts 24 bytes per element, as STREAM does. It first touches the arrays once, on all of Numba's threads, before any thread count runs, so the docstring's "by the threads that use them" holds at the full count only. Each line of `triad.jsonl` is one thread count. The M5 Pro's run is [beside this one](../../../qe-wharton2/2026-11-09/triad/README.md).

| | |
|---|---|
| Host | `qe-mini1`: the Apple M6 Mac mini, 12 cores (2 Super, 4 Performance, 6 Efficiency) and 32 GB, on macOS 27.2 (26B123), logged in at the screen |
| Software | The runs clone at `5fbf033`, with its venv: Python 3.12.15, NumPy 2.5.4, Numba 0.68.1 with its `workqueue` threading layer, and llvmlite 0.50.0 |
| When | 2026-11-09, 05:27:00 to 05:27:03 UTC, then the check below to 05:27:33, in a quiet window agreed with the sessions that run on this Mac, which stayed running |
| Script | `session/m6_triad.sh`, with `NUMBA_NUM_THREADS=12` |

**`state.txt`** has only the before and after states, since the run took 3 s and the sampler waits 5 s.

## What it shows

| Threads | Best time | GB/s |
|---:|---:|---:|
| 1 | 0.0478 s | 134.8 |
| 2 | 0.0464 s | 138.8 |
| 4 | 0.0480 s | 134.3 |
| 8 | 0.0471 s | 136.7 |
| 12 | 0.0478 s | 134.9 |

- **One thread takes all the bandwidth the M6's memory delivers,** about 135 GB/s, and more threads add nothing.
- **The thread counts took effect.** With `NUMBA_NUM_THREADS=1`, so that Numba's whole pool is one thread, the triad gave 135.1 GB/s (`check-pool1.jsonl`). There, one thread first touched the arrays as well as using them, so first touch on all 12 threads made no difference to the one-thread row. And `session/check_cpu_wall.py` found `set_num_threads(1)` using 1.0 CPU-second per second, against 10.5 at 12 (`check-pool1.log`).
