# One core's bandwidth and P1's serial Bellman steps on qe-mini1's M6, 2026-11-09

For QuantEcon/project-benchmarks-jax#47, whose AWS instances ran the same probe: whether P1's serial cells (`numpy` and `numba`) are limited by one core's memory bandwidth. The probe is [`problems/checks/bw_probe.py`](../../../checks/bw_probe.py) (4,615 bytes, sha256 `3ab1785b…`). It times six operations on P1's own stored rewards at a wealth grid of 1,500 (1,500 × 100 × 1,500 float64, 1.8 GB), each once to warm up and then three times. Each line of `bandwidth.jsonl` is one operation: its best and median time, the bytes it must move at least, and the GB/s that implies. The M5 Pro's run is [beside this one](../../qe-wharton2/2026-11-09/README.md).

| | |
|---|---|
| Host | `qe-mini1`: the Apple M6 Mac mini, on macOS 27.2 (26B123), logged in at the screen |
| Software | The runs clone at `5fbf033`, whose `problems/vfi.py` is the same file as at `1874f1c`, with its venv: Python 3.12.15, NumPy 2.5.4, Numba 0.68.1 and llvmlite 0.50.0, the pins of `problems/constraints.txt` at `1874f1c` |
| When | 2026-11-09, 04:36:41 to 04:36:44 UTC, in a quiet window agreed with the sessions that run on this Mac, which stayed running |
| Script | `session/m6_bw.sh`, with every thread pool at one. macOS has no `taskset` |

**`state.txt`** has only the before and after states, since the run took 3 s and the sampler waits 5 s. The load was 0.9 before.

## What it shows

| Operation | Bytes counted | Best time | GB/s |
|---|---|---:|---:|
| `read`: `u.sum()` | 1× | 0.0228 s | 78.8 |
| `copy` into a buffer made once | 2× | 0.0296 s | 121.8 |
| `numpy`: `np.max(u + b, axis=2)`, a fresh temporary each time | 3× | 0.1291 s | 41.8 |
| `numpy-out`: the same into a buffer made once | 3× | 0.1302 s | 41.5 |
| `numba`: `vfi.py`'s serial Bellman step | 1× | 0.0913 s | 19.7 |
| `numba-evT`: the same with EV transposed | 1× | 0.0912 s | 19.7 |

- **The serial Numba step isn't limited by bandwidth here:** it runs at a quarter of `read`'s rate, about 2.5 billion elements a second. That fits a limit from its running maximum, each step waiting on the last.
- **Allocation costs nothing:** `numpy` and `numpy-out` take the same time.
- **The probe accounts for the suite's cells:** P1 at 1,500 takes 572 iterations in [2026-11-02's results](../2026-11-02.jsonl), whose warm medians are 77.95 s in `numpy` and 54.55 s in `numba`, or 0.136 s and 0.095 s an iteration. The probe's steps are 95% and 96% of that.
