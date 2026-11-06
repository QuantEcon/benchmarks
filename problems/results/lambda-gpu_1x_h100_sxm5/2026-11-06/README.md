# `lambda-gpu_1x_h100_sxm5`, 2026-11-06: the problem-type suite on an H100

This is the suite's `jax-gpu` backend on one H100 SXM 80 GB. It was run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh), for QuantEcon/project-benchmarks-jax#37. The same instance type ran the precision set on 2026-10-26 (`results/lambda-gpu_1x_h100_sxm5/`).

| | |
|---|---|
| Instance | Lambda's `gpu_1x_h100_sxm5`, on demand, in us-south-2, after a launch in us-southeast-1 was refused for capacity. It had 26 vCPUs and 221 GiB, on Ubuntu 22.04.5 with kernel 6.8.0-60-generic, and was launched from a laptop and terminated once the results were copied off. DigitalOcean, the first choice, listed no one-GPU H100 that day |
| Card | NVIDIA H100 80GB HBM3, power limit 700 W, driver 570.148.08 with CUDA 12.8 (`machine.txt`) |
| Software | JAX 0.12.0.1 with `jax[cuda12]`, since the driver is older than 580, so the rows are labelled `cuda12`, on Python 3.12.16 (`environment.txt`) |
| Harness | `5cd9d3d` |
| On the machine (UTC) | 2026-11-05 23:59 to 2026-11-06 00:38, launch request to termination |

## The runs

- **The `--quick` check,** 00:31 to 00:32. Its nine rows (`run_id` 20261106T003123Z) are left out of the `.jsonl`, since the full run repeats each of their cells. Its output is in `session/problems-quick.log`.
- **The full run,** 00:34 to 00:36, `run_id` 20261106T003420Z: every size in both precisions, 50 cells, all `ok` (`session/chain37.sh`, `run.log`).
- **Every float32 sweep again with `JAX_DEFAULT_MATMUL_PRECISION=highest`,** 00:36 to 00:38, `run_id` 20261106T003637Z: 22 cells, all `ok` (`session/highest.sh`, `run-highest.log`). It's labelled `cuda12-highest`, the script's label with `-highest` after it, as `precision/run_by_hand.sh` labels its own pass.

`summary.md` is this host's tables, from `python problems/summarise.py problems/results/` once the `--quick` rows were left out. It adds the `highest` tables to what the run wrote.

## What they show

- **P3 in float32, at XLA's default precision, stops early with a wrong answer.**
    - It took 809 iterations at 200 points, and 1,005 to 1,381 at 1,000 to 50,000, against NumPy float64's 2,190 to 2,192.
    - Its answers differ from NumPy's float64 answer by a relative 0.11 to 0.12, where the suite allows 1e-3.
    - On this card, too, XLA runs float32 matrix products in TF32 by default. The iteration counts are the RTX PRO 6000's, size by size.
- **With `highest` it took 2,208 to 2,214 iterations,** again the RTX PRO 6000's counts, and every answer agreed. Its warm times were 0.28 s to 0.97 s, against 0.10 s to 0.59 s at the default.
- **Everything else agrees with NumPy:** P1, which runs in float64 only, P3 in float64, and P4 and P7 in both precisions.
- **Compiles are quick:** no cell's cold time passed 2.5 s, and P1's were 0.8 s to 1.8 s.
