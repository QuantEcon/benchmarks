# A 1g.24gb MIG slice of an RTX PRO 6000, on RunPod, 2026-11-18

For QuantEcon/project-benchmarks-jax#63: the problem types and the precision suite on one MIG slice of an RTX PRO 6000 Blackwell Server Edition, rented on RunPod's Secure Cloud. A slice is how one card serves several users with their own memory and SMs. It ran by hand from a laptop session, from 01:22:59 UTC.

**The harness:** QuantEcon/benchmarks at `8d777d6`, the head of QuantEcon/benchmarks#66, with a clean tree. On a slice, `nvidia-smi --query-gpu=memory.total` reads "[Insufficient Permissions]". At `f085c6d` that stopped `env_info.nvidia_gpus()` before the first cell, and `card/attempt1/` keeps that attempt. #66 reads the slice's memory from `nvidia-smi -q -x` instead, and records it under `mig_slices`.

| File | What it holds |
|---|---|
| `../2026-11-18.jsonl` | The problem types: 50 cells, and 22 float32 cells again under `highest`, all ok |
| `../../../../results/runpod-rtxpro6000-mig1g24/2026-11-18.jsonl` | The precision matrix: 43 rows ok, 2 out of memory and 1 skipped, and 12 `highest` rows, all ok |
| `../../../../lectures/results/runpod-rtxpro6000-mig1g24/2026-11-18/` | `precision/run_by_hand.sh`'s five lectures and two scale scripts |
| `run.log`, `run-highest.log`, `environment.txt`, `machine.txt`, `summary.md` | The problem types' logs, packages, machine and tables |
| `card/` | The slice's record: `nvidia-smi -L`, `-q` and `-q -x`, `card.csv`, JAX's view of it (`jax-view.txt`), the host, the commit, the session's script (`mig.sh`), its log and timeline (`mig.log`, `status.txt`). The card's serial number and the GPU and MIG UUIDs are redacted, as `REDACTED` in the XML so that it still parses |
| `card/attempt1/` | The first attempt's log, which stopped on the harness bug, and the slice's record. `nvidia-smi-q.xml` there is stamped 01:21:33 UTC, after that attempt stopped at 01:18:38 and before the restart at 01:22:59, so it was taken between the attempts, not by the first attempt's script |

**The slice:** MIG 1g.24gb of an NVIDIA RTX PRO 6000 Blackwell Server Edition. It has 46 of the card's 188 SMs and 24,192 MiB, and MIG mode is enabled. The card's power limit is 600 W, settable from 300 to 600 W. Driver 595.112.07 with CUDA 13.2, and JAX 0.12.0.1 with `jax[cuda13]`. JAX sees it as one device, `NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb`, with an allocator limit of 17.72 GiB by default and 22.45 GiB at a memory fraction of 0.95. The host reports 256 CPUs, as `host.txt` records, and Ubuntu 24.04.4 with kernel 6.8.0-138.

**Left out of the `.jsonl` files:** the `--quick` checks run first: 9 problem cells, and 18 precision rows, 4 of them `highest`. They're in `card/mig.log`.

## What they show

These are compared with the whole card's rows on `aws-g7e.2xlarge`: the problem types of 2026-11-05, on driver 595.112.07 as here, and the precision matrix of 2026-10-26, on driver 595.91.07.

- **The large cells run about 4× slower, a quarter of the card's speed.** At the largest size of 8 of the 9 problem sweeps, the slice takes 2.2–4.3× as long, warm. The ninth, P7's state sweep at 200 in float32, takes 1.13×. `bellman_stored` reads 367 GB/s at 40,000 in float32, against 1,471, and `matmul` at 8,192 gives 427 GFLOPS in float64 against 1,757.
- **The small cells don't slow down.** Latency-bound cells, such as the Kalman filter's state sweep, take 0.73–1.1× as long, and the median over all 50 problem cells is 1.07.
- **Two cells ran out of memory while compiling, as on Phase 2's A10.** `bellman_stored` at 40,000 in float64 failed allocating 11.94 GiB, and at 60,000 in float32 allocating 13.43 GiB, the same sizes as on the A10. At 60,000 in float64 it was skipped, needing about 26.8 GiB against a planned 18.9. Both tracebacks fail inside `backend_compile_and_load`. That's consistent with XLA's autotuner, which times candidate kernels on buffers of the real size, so compiling the step needs a second copy of its array. [The 2026-11-01 run notes](../../../../precision/runs/2026-11-01/README.md) established that on the A10 by switching the autotuner off. It wasn't tested on the slice. The rows' `memory_stats` show the same here: at 40,000 in float32 the peak is 12.9 GB for a 6.4 GB array. So 40,000 in float64 needs about 25.6 GB against the slice's 24.1 GB limit at 0.95.
- **Some compiles are much slower than on the whole card, and why isn't established.** By the rows' `compile_s`, `bellman_recompute` at 10,000 took 86 s in float32 and 326 s in float64, against 0.3 and 0.8 s on the whole card on 2026-10-26. At 20,000 and 40,000 in float64 it took about 82 s, against 19, and `matmul` at 8,192 in float64 took 259 s, against 66. Phase 2's long float64 compiles on Ampere and later cards were the autotuner, by the 2026-11-01 notes, but the slice's 86 s in float32 at 10,000 isn't explained by that.
- **P3's float32 answers show TF32, as on the whole card.** At XLA's default precision they stop 11–12% from NumPy's float64 answer, after fewer iterations. Under `highest` they agree (`summary.md`).
