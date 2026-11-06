# `qe-wharton1-gpu`, 2026-11-06: the CPU stacks under four NUMA placements

For QuantEcon/project-benchmarks-jax#46, which asks why this machine's Threadripper 2990WX runs the CPU stacks slowly. Four of the suite's cells ran in float64 under three `numactl` placements, and then the `numpy` and `numba` stacks ran again on the CPUs of a node without memory. The suite is QuantEcon/benchmarks' `main` at `4d09ef9`, with nothing changed, run through [`numa46.sh`](numa46.sh).

| | |
|---|---|
| Host | `qe-wharton1-gpu`: an AMD Ryzen Threadripper 2990WX, with 32 cores and 64 threads, and 125 GiB, on Linux 7.0.0-34 (Ubuntu 24.04). Core boost is off (`cpufreq` boost 0), so the cores run at 2.2 to 3.2 GHz (`layout.log`) |
| Memory | Four NUMA nodes of eight cores each. Nodes 0 and 2 have the memory, 64 GB each; nodes 1 and 3 have none, so their cores reach memory only across the fabric. `numactl` puts another node at a distance of 16, against 10 for a node's own. `numa_balancing` is on, transparent huge pages are `madvise`, and the governor is `schedutil`. The BIOS's memory interleave setting can't be read without root |
| Software | The run checkout's `.venv`: Python 3.12.3, JAX 0.12.0.1, Numba 0.68.1, and NumPy 2.5.4 with OpenBLAS 0.3.34 (`environment.txt`) |
| Threads | None is set. OpenBLAS, Numba and XLA each size their pools to the CPUs a placement allows |
| When | 2026-11-06, 07:08:45 to 10:29:34 UTC, under the machine's lock, in tmux |

**The placements,** each under its own `--label`:

| Label | `numactl` | CPUs | Memory |
|---|---|---|---|
| `numa-default` | none, as on 2026-11-02 | all 64: 32 cores | where the kernel puts it, on node 0 or 2 |
| `numa-one-node` | `--cpunodebind=0 --membind=0` | node 0's 16: 8 cores | node 0 |
| `numa-memory-nodes` | `--cpunodebind=0,2 --membind=0,2` | nodes 0 and 2's 32: 16 cores | nodes 0 and 2 |
| `numa-remote-memory` | `--physcpubind=16-23,48-55 --membind=0` | node 1's 16: 8 cores | node 0, across the fabric |

`numactl` won't take `--cpunodebind` for a node without memory, so `numa-remote-memory` names node 1's CPUs. `layout.log` ends with each placement's `numactl --show`.

**The cells,** in float64, each with `--timeout 1800`:

| Problem | Size | Under the first three placements | Under `numa-remote-memory` |
|---|---:|---|---|
| P1 `vfi`, grid sweep | 1,500 | `numba`, `numba-parallel`, `jax-cpu` | `numba` |
| P7 `kalman`, batch sweep | 10,000 | `numpy`, `numba`, `numba-parallel`, `jax-cpu` | `numpy` |
| P4 `wealth`, households sweep | 10,000,000 | `numpy`, `numba-parallel`, `jax-cpu` | `numpy` |
| P3 `egm`, grid sweep | 50,000 | `numpy`, `numba`, `numba-parallel` | `numpy` |

- **Why these:** at these sizes P1, P3 and P7 are where the M6 was furthest ahead on 2026-11-02, and P4 is where the Threadripper won.
- **Left out, to fit an evening:** float32, and three stacks that were the slowest on the Threadripper and never its fastest: P1's `numpy`, P4's `numba` and P3's `jax-cpu`.

**The runs:**
- **A first start, at 07:07:11 UTC, was stopped** within two minutes, when `numactl` refused `--cpunodebind=1`. Nothing from it reached `problems/results/`.
- **The check,** 07:08:46 to 07:13:31: each cell's `--quick` sizes under each of the first three placements, into the run's own folder. All 39 rows were ok (`logs/check.log`), so the run went on. Its rows are left out.
- **The run,** 07:13:31 to 10:06:37: cell by cell, each under the three placements in turn, so that any drift would fall on all three alike. 39 rows.
- **On a node without memory,** 10:06:37 to 10:29:34: 4 rows.
- **Every step exited 0,** and all 43 rows are ok. `state.txt` logs each step's start, end and exit status, and samples the machine every 10 minutes.
- **Afterwards, an optional EGM run was done for project-econ-on-ai-hardware,** which reports it. `state.txt` stops before it, and this copy of `numa46.sh` has that step cut. Otherwise it's the script as it ran.

**The machine was quiet:** in `state.txt`'s samples no process outside the run took more than 1.3% of a core, and both cards stayed at 0%. Another user's name in its process lists is replaced by `(other)`.

**The files:**
- `numa46.sh` and `state.txt`;
- `layout.log`: `numactl --hardware`, `lscpu`, `free`, the kernel's NUMA, huge-page and frequency settings, and each placement's `numactl --show`;
- `environment.txt`: the `.venv`'s packages;
- `summary.md`: the tables `python problems/summarise.py problems/results/` prints for the four labels;
- `logs/`: each step's output.

## What they show

Each cell's warm median in seconds, with its ratio to `numa-default` in brackets:

| Cell | Backend | `numa-default` | `numa-one-node` | `numa-memory-nodes` | `numa-remote-memory` |
|---|---|---:|---:|---:|---:|
| P1 `vfi`, 1,500 | `numba` | 145.0 | 137.9 (0.95) | 138.6 (0.96) | 162.6 (1.12) |
| | `numba-parallel` | 33.5 | 48.3 (1.44) | 42.2 (1.26) | |
| | `jax-cpu` | 35.4 | 34.4 (0.97) | 34.1 (0.97) | |
| P7 `kalman`, 10,000 | `numpy` | 47.8 | 41.4 (0.87) | 43.6 (0.91) | 45.6 (0.95) |
| | `numba` | 66.5 | 63.1 (0.95) | 63.9 (0.96) | |
| | `numba-parallel` | 197.3 | 89.0 (0.45) | 157.0 (0.80) | |
| | `jax-cpu` | 63.8 | 42.6 (0.67) | 54.3 (0.85) | |
| P4 `wealth`, 10,000,000 | `numpy` | 127.6 | 128.1 (1.00) | 128.0 (1.00) | 141.7 (1.11) |
| | `numba-parallel` | 5.15 | 20.6 (4.01) | 10.3 (1.99) | |
| | `jax-cpu` | 85.4 | 93.1 (1.09) | 82.7 (0.97) | |
| P3 `egm`, 50,000 | `numpy` | 289.0 | 266.5 (0.92) | 273.3 (0.95) | 312.2 (1.08) |
| | `numba` | 306.5 | 287.9 (0.94) | 297.8 (0.97) | |
| | `numba-parallel` | 301.0 | 257.5 (0.86) | 273.9 (0.91) | |

- **Pinned to the two nodes with memory,** the largest gain in any cell was 20%, for P7's `numba-parallel`. The `numpy` and `numba` stacks gained 0% to 9%.
- **On one node's 8 cores,** the cells that don't scale ran faster than on all 32: P7's `numba-parallel` in 0.45 of the time, its `jax-cpu` in 0.67 and its `numpy` in 0.87, and P3's three stacks in 0.86 to 0.94.
- **The cells that scale ran slower there:** P4's `numba-parallel` took 4.0 times as long on 8 cores and 2.0 on 16, in proportion to the cores, and P1's took 1.44 and 1.26 times.
- **Memory across the fabric** cost the `numpy` and `numba` stacks 10% to 18%, against the same number of CPUs on node 0 with its own memory: `numa-remote-memory` against `numa-one-node`.
- **`numa-default` reproduces 2026-11-02's times** (`../2026-11-02.jsonl`, at `5fbf033`): 12 of its 13 cells within 5%, and P1's `jax-cpu` 11% slower.
- **The timings' spread:** most cells had one or two warm calls in the suite's time budget. Where a cell had two or more, they agree within 2%, except three:
    - P1's `jax-cpu`: 31.9 and 38.8 s under the default, and 36.8 and 31.5 s under the memory nodes;
    - P1's `numba-parallel` under the memory nodes: 44.0 and 40.4 s;
    - P4's `numba-parallel` under the default, whose first warm call took 5.9 s, against 5.1 to 5.3 s.

    The cold times, each from a fresh process, show the same pattern.
- **Every answer agrees** with NumPy's float64 answer, and every cell took the same number of iterations under every placement (`summary.md` flags none). Across placements each backend's answer is the same to 1.5e-15, except P4's `numba-parallel`, whose answer moves with the thread count, by 3.1e-4 at most, inside the suite's test.
