# OpenBLAS's threads and parallel Numba on qe-wharton1-gpu's Threadripper, 2026-11-09

For QuantEcon/project-benchmarks-jax#47: whether OpenBLAS's thread pool holds back parallel Numba on x86. The suite caps no thread pool, so on x86, where NumPy and SciPy use scipy-openblas, the matrix products and solves Numba calls can start OpenBLAS's threads, which spin after each call. In #47's runs, parallel Numba was 4.1× to 4.6× as fast as serial Numba on P1 at 1,500 on the three AWS instances and on this Threadripper. On P7's batch of 10,000 filters it was 0.33× to 1.22×. The Macs, whose NumPy and SciPy use Accelerate, got 5.3× to 8.8× and 6.2× to 9.4×. Here the same cells ran with OpenBLAS as the suite leaves it, labelled `blas-default-…`, and with `OPENBLAS_NUM_THREADS=1`, labelled `blas1-…`. The rows are in [`../2026-11-09.jsonl`](../2026-11-09.jsonl).

| | |
|---|---|
| Host | `qe-wharton1-gpu`: a Threadripper 2990WX, 32 cores and 64 threads, with boost off, in four NUMA nodes, of which 0 and 2 have the memory, 64 GB each (`p1/setup.txt`) |
| Software | The run checkout at `4d09ef9`, whose timed code and runner (`problems/vfi.py`, `kalman.py`, `common.py`, `run_cell.py`, `run_suite.py` and `spec.py`) are the same files as at `f427574`, with its venv: NumPy 2.5.4, SciPy 1.18.1, Numba 0.68.1, and scipy-openblas |
| When | 2026-11-09: P1 from 06:48:04 to 07:12:35 UTC, then P7 from 07:13:42 to 07:31:24, under project-econ-on-ai-hardware's machine lock, with both cards idle |
| Scripts | `session/blas47.sh` (P1) and `session/blas47_p7.sh` (P7), each run in `tmux`. Beside each, [`problems/checks/clock_sampler.py`](../../../checks/clock_sampler.py) (sha256 `05591ad6…`) sampled every CPU every 2 s into `p1/clock.jsonl` and `p7/clock.jsonl` |

"CPUs busy" below is the sum over all 64 CPUs of each CPU's busy share, averaged over the cell's warm runs, from those samples.

## P1 at 1,500, float64

Parallel Numba ran twice in each setting, alternating, at the default placement and on node 0, whose 16 CPUs (8 cores) have their own memory, as a 16-CPU slice does. Each figure is one run's warm median.

| Cell | OpenBLAS as left, warm | `OPENBLAS_NUM_THREADS=1`, warm | Gain | CPUs busy |
|---|---|---|---:|---|
| `numba-parallel`, all 64 CPUs | 32.3 and 33.6 s (cold 36.6, 39.5) | 22.4 and 22.5 s (cold 25.5, 28.7) | 1.47× | 62 → 44 |
| `numba-parallel`, node 0 | 48.0 and 48.4 s (cold 51.9, 51.7) | 34.9 and 34.9 s (cold 38.6, 38.6) | 1.38× | 16 → 15 |
| `numba`, serial | 136.6 s (cold 138.8) | 134.1 s (cold 136.9) | 1.02× | 20.4 → 1.0 |

- **OpenBLAS's threads cost parallel Numba 1.4× to 1.5× here.** Capped, it runs 6.0× as fast as serial Numba on all 64 CPUs, against 4.1× uncapped, and 3.8× on node 0, against 2.8×.
- **The serial cell shows them:** uncapped, it kept about 20 CPUs busy for one core's work. Those are OpenBLAS's threads, spinning between each iteration's `v @ Q.T`. Capped, it kept one. So the cap reaches the OpenBLAS that Numba's `@` calls, through SciPy.

## P7's batch of 10,000 filters, float64

Each filter calls `@`, `np.linalg.solve` and `slogdet` inside the `prange` loop. Each setting ran once, at the default placement.

| Cell | OpenBLAS as left, warm | `OPENBLAS_NUM_THREADS=1`, warm | Gain | CPUs busy |
|---|---|---|---:|---|
| `numba-parallel` | 196.1 s (cold 213.8) | 173.1 s (cold 190.1) | 1.13× | 63 → 63 |
| `numba`, serial | 65.9 s (cold 78.5) | 62.7 s (cold 74.7) | 1.05× | 1.0 → 1.0 |

- **Here OpenBLAS's threads explain little.** Capped, parallel Numba is still 2.8× slower than serial Numba, with every CPU busy.
- **The serial cell kept one CPU busy either way,** so OpenBLAS starts no threads for products this small. What holds P7's parallel Numba back on x86 isn't established here.
