# A 1g.24gb MIG slice of an RTX PRO 6000, on RunPod, 2026-11-18

For QuantEcon/project-benchmarks-jax#63: the problem types and the precision suite on one MIG slice of an RTX PRO 6000 Blackwell Server Edition, rented on RunPod's Secure Cloud. A slice is how one card serves several users with their own memory and SMs. It ran by hand from a laptop session, from 01:22:59 UTC.

**The harness:** QuantEcon/benchmarks at `8d777d6`, the head of QuantEcon/benchmarks#66, with a clean tree. On a slice, `nvidia-smi --query-gpu=memory.total` reads "[Insufficient Permissions]". At `f085c6d` that stopped `env_info.nvidia_gpus()` before the first cell, and `card/attempt1/` keeps that attempt. #66 reads the slice's memory from `nvidia-smi -q -x` instead, and records it under `mig_slices`.

| File | What it holds |
|---|---|
| `../2026-11-18.jsonl` | The problem types: 50 cells, and 22 float32 cells again under `highest`, all ok |
| `../../../../results/runpod-rtxpro6000-mig1g24/2026-11-18.jsonl` | The precision matrix: 43 rows ok, 2 out of memory and 1 skipped, and 12 `highest` rows, all ok |
| `../../../../lectures/results/runpod-rtxpro6000-mig1g24/2026-11-18/` | `precision/run_by_hand.sh`'s five lectures and two scale scripts |
| `run.log`, `run-highest.log`, `environment.txt`, `machine.txt`, `summary.md` | The problem types' logs, packages, machine and tables |
| `card/` | The slice's record: `nvidia-smi -L`, `-q` and `-q -x`, `card.csv`, JAX's view of it (`jax-view.txt`), the host, the commit, the session's script (`mig.sh`), its log and timeline (`mig.log`, `status.txt`). The card's serial number and the GPU and MIG UUIDs are redacted |
| `card/attempt1/` | The first attempt's record and log, which stopped on the harness bug |

**The slice:** MIG 1g.24gb of an NVIDIA RTX PRO 6000 Blackwell Server Edition. It has 46 of the card's 188 SMs and 24,192 MiB, and MIG mode is enabled. The card's power limit is 600 W, settable from 300 to 600 W. Driver 595.112.07 with CUDA 13.2, and JAX 0.12.0.1 with `jax[cuda13]`. JAX sees it as one device, `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb`, with an allocator limit of 17.72 GiB by default and 22.45 GiB at a memory fraction of 0.95. The host reports 256 CPUs, as `host.txt` records, and Ubuntu 24.04.4 with kernel 6.8.0-138.

**Left out of the `.jsonl` files:** the `--quick` checks run first: 9 problem cells, and 18 precision rows, 4 of them `highest`. They're in `card/mig.log`.

## What they show

These are compared with the whole card's rows on `aws-g7e.2xlarge`: the problem types of 2026-11-05 and the precision matrix of 2026-10-26, on driver 595.91.07.

- **The large cells run about 4× slower, a quarter of the card's speed.** At each sweep's largest size the slice takes 2.2–4.3× as long, warm. `bellman_stored` reads 367 GB/s at 40,000 in float32, against 1,471, and `matmul` at 8,192 gives 427 GFLOPS in float64 against 1,757.
- **The small cells don't slow down.** Latency-bound cells, such as the Kalman filter's state sweep, take 0.73–1.1× as long, and the median over all 50 problem cells is 1.07.
- **Two cells ran out of memory.** `bellman_stored` at 40,000 in float64 failed allocating 11.94 GiB, and at 60,000 in float32 allocating 13.43 GiB. At 60,000 in float64 it was skipped, needing about 26.8 GiB against a planned 18.9. In this harness, the step's peak memory is about twice its stored array: at 40,000 in float32, 12.9 GB at the peak for a 6.4 GB array, by the row's `memory_stats`. So 40,000 in float64 needs about 25.6 GB, against the slice's 24.1 GB limit at 0.95.
- **Some compiles are slow:** `bellman_recompute` at 10,000 in float64 took 326 s to compile, and `matmul` at 8,192 in float64 took 259 s, by the rows' `compile_s`. Why isn't established here.
