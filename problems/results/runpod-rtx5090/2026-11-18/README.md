# A GeForce RTX 5090 on RunPod, 2026-11-18

For QuantEcon/project-benchmarks-jax#60: the problem types, the precision matrix and the five lectures on one GeForce RTX 5090, rented on RunPod's Secure Cloud. The card's hour of sustained load and cuBLAS's float64 emulation on it are in [`precision/runs/2026-11-18/runpod-rtx5090/`](../../../../precision/runs/2026-11-18/runpod-rtx5090/README.md). It ran by hand from a laptop session, from QuantEcon/benchmarks at `f085c6d`, from 01:00 UTC.

| File | What it holds |
|---|---|
| `../2026-11-18.jsonl` | The problem types: 50 cells, and 22 float32 cells again under `highest`, all ok |
| `../../../../results/runpod-rtx5090/2026-11-18.jsonl` | The precision matrix: 45 rows ok and 1 skipped, and 12 `highest` rows, all ok |
| `../../../../lectures/results/runpod-rtx5090/2026-11-18/` | `precision/run_by_hand.sh`'s five lectures and two scale scripts |
| `run.log`, `run-highest.log`, `environment.txt`, `machine.txt`, `summary.md` | The problem types' logs, packages, machine and tables |
| `card/` | The card's record (`card.csv`, `nvidia-smi-q.txt`), the host, the commit, and the power-limit test (`pl-test.txt`) |
| `session/` | The session's two scripts and their logs, and its timeline (`status.txt`). The timeline leaves out a later step that ran other code, not part of this suite |

**The card:** an NVIDIA GeForce RTX 5090, made by Gigabyte (sub-device `0x41991458`), with 32,607 MiB and VBIOS 98.02.2E.00.C9, on PCIe 5.0 x16. Driver 595.112.07 with CUDA 13.2, and JAX 0.12.0.1 with `jax[cuda13]`. Its power limit was 600 W, which is also its default; it can be set from 400 to 600 W. The pod couldn't set it: `nvidia-smi -pl` was refused for want of permissions (`pl-test.txt`). The host is an AMD Ryzen 9 9950X, with 32 CPUs visible to the pod, on Ubuntu 24.04.4 with kernel 6.8.0-149.

**Left out of the `.jsonl` files:** the `--quick` checks run first: 9 problem cells, and 18 precision rows, 4 of them `highest`. They're in `session/phaseA.log`.

## What they show

These are compared with the RTX PRO 6000 Blackwell Server Edition on `aws-g7e.2xlarge` at its 600 W: the problem types of 2026-11-05 and the precision matrix of 2026-10-26. Both cards are GB202; the 5090 has 170 of its SMs enabled against the Server Edition's 188.

- **The problem types run faster on the 5090.** It takes 0.80–0.89× as long at each sweep's largest size, warm, and the median over all 50 cells is 0.70×. The small cells gain most, which suggests the host as much as the card.
- **Bandwidth and float64 are a little ahead.** `bellman_stored` at 40,000 reads 1,629 GB/s in float32 and 1,059 in float64, against 1,471 and 919. `matmul` at 8,192 in float64 gives 2,054 GFLOPS against 1,757.
- **Float32 matrix products are behind.** At JAX's default precision, which is TF32 on both cards, `matmul` at 8,192 gives 125 TFLOPS, against 186 on the Server Edition.
- **The 5090 runs float32 products in TF32 by default too.** P3's float32 answers stop 11–12% from NumPy's float64 answer at the default precision, as on the Server Edition, and agree under `highest` (`summary.md`).
- **Only the largest stored step didn't fit.** `bellman_stored` at 60,000 in float64 was skipped, needing about 26.8 GiB against a planned 25.5. Every other cell ran, 40,000 in float64 included: its 25.8 GB peak, twice the array while XLA's autotuner compiles it ([2026-11-01](../../../../precision/runs/2026-11-01/README.md)), fits the card's 32 GB.
