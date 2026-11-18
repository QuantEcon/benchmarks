# Two jobs on one TITAN RTX, qe-wharton1-gpu, 2026-11-18

For QuantEcon/project-compute#148, by way of QuantEcon/project-benchmarks-jax: how JAX jobs fare when two share one card without MIG, as two users of one workstation card would. It ran on the TITAN RTX at `0b:00.0` (the suite's `--gpus 0`), from QuantEcon/benchmarks at `f085c6d` with `jax[cuda13]` 0.12.0.1, from 01:32 to 02:12 UTC. It ran under econ-on-ai-hardware's machine lock, with nothing else computing on the card. Xorg, the display server, held 4 MiB on it, as `card-q.txt`'s process list in the sustained run shows. The same window then ran the card's sustained load, in [`precision/runs/2026-11-18/qe-wharton1-gpu/`](../../../../precision/runs/2026-11-18/qe-wharton1-gpu/README.md).

Every JAX process in steps 1 to 4 had 45% of the card's memory (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.45`, and `--memory-fraction 0.45` so that each job admits the same cells). The card at `43:00.0` drives the desktop and wasn't used. The script is `session/titan_shared65.sh`. The plan its header mentions isn't published.

| Step | Label in `../2026-11-18.jsonl` | What ran |
|---|---|---|
| 0 | none (`trap/`) | Two processes with JAX's defaults: the first preallocates 75% of the card, then a second starts |
| 1 | `shared-solo` | The 50 `jax-gpu` cells, one job, 01:34:06–01:39:46 |
| 2 | `shared-a`, `shared-b` | Two such jobs started together, 01:40:16–01:49:43 |
| 3 | `shared-solo2` | One job again, 01:50:13–01:56:01, to bound drift |
| 4 | `neighbour-busy` | The 9 largest cells, while `precision/sustained.py` held the card with `matmul` at 8,192 in float32 (`neighbour/`) |

All 209 rows are ok. `gpu.csv` samples both cards every 5 s, `state.txt` records the machine before and after, and each step's log is beside them.

## What they show

- **A second process with JAX's defaults starts, but quietly gets less.** With the first holding 18,190 MiB, the second's 17.60 GiB preallocation failed with `CUDA_ERROR_OUT_OF_MEMORY`. XLA retried at 0.9 of the size each time, logging each failure, until a pool fit. It then ran a 4 GiB array (`trap/b-default.txt`). With `XLA_PYTHON_CLIENT_PREALLOCATE=false` it started cleanly (`trap/b-nopreallocate.txt`). So two users of one card should each set that, or a memory fraction.
- **When the two jobs run the same cells together, most small cells don't notice.** Jobs A and B ran the same cells in step: each cell started within 1.86 s of its twin, with a median gap of 0.65 s. Of the 28 cells taking under 0.2 s warm alone, the median took 1.02× as long. Beside a job that keeps the card busy, small cells do slow, as step 4 shows: `wealth` at 10,000,000 in float32 took 2.29× as long, and `kalman`'s state sweep at 200 in float32 1.57×. The exception is `vfi`: at grids of 100 to 400 it took 2.4–3.1× as long, the most of any cell. It runs hundreds of short iterations, and each probably waits its turn on the card.
- **Large cells about double.** The 9 cells taking over 1 s took 1.95–2.34× as long, with a median of 2.26× over both jobs. For example:

  | Cell | Alone, warm s | Shared: job A, job B | Beside the `matmul` |
  |---|---:|---:|---:|
  | `vfi` grid 1,500, float64 | 3.28 | 7.44, 7.40 | 9.42 |
  | `egm` grid 50,000, float64 | 7.45 | 16.51, 16.51 | 18.49 |
  | `wealth` 10,000,000, float64 | 2.40 | 5.47, 5.47 | 5.90 |
  | `kalman` batch 10,000, float64 | 3.21 | 7.09, 7.09 | 7.73 |

- **Together the two jobs got 1.21× the work of one done.** Both finished in 567 s, against 340 and 348 s for one job alone, so each job took 1.65× as long. The two solo passes agree within a median of 0.8%.
- **A job beside a busy neighbour fares worse still,** taking 2.4–3.1× as long in its large cells. The neighbour slowed too: its step took 82.8 ms alone, 155.1 ms while the cells ran, and 86.5 ms after (`neighbour/`).

On one card, CUDA runs the two processes' kernels by turns rather than together (MPS wasn't used), so a cell that keeps the card busy slows by about half or more. The TITAN is Turing; a newer card shares by the same turns, but these ratios are for this card and these jobs.
