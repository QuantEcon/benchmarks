# `aws-g7e.2xlarge`, 2026-11-05: the problem-type suite on an RTX PRO 6000

The suite's `jax-gpu` backend on one RTX PRO 6000 Blackwell Server Edition, run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh), for QuantEcon/project-benchmarks-jax#32. The same instance type ran the precision suite on 2026-10-26 (`results/aws-g7e.2xlarge/`).

| | |
|---|---|
| Instance | EC2 `g7e.2xlarge`, on demand, in us-east-2b after us-east-2a refused it for capacity. It was launched from a laptop's AWS CLI, with a key pair and an SSH-only security group made for it and deleted afterwards |
| Image | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04), 20261103 (`ami-0f9eff726a7654780`) |
| Card | RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, power limit 600 W, driver 595.112.07 (`machine.txt`) |
| Software | JAX 0.12.0.1 with `jax[cuda13]`, Python 3.12.3 (`environment.txt`) |
| Harness | `6b42b04` |
| On the machine (UTC) | 06:48 to 06:59, launch to termination |

## The runs

- **The `--quick` check,** 06:50 to 06:51 (`session/quick.sh`). Its nine rows (`run_id` 20261105T065044Z) are left out of the `.jsonl`, since the full run repeats each of their cells. Its output is in `run.log`.
- **The full run,** 06:52 to 06:56, `run_id` 20261105T065217Z: every size in both precisions, 50 cells, all `ok` (`session/full.sh`). It sets `UV_VENV_CLEAR=1`, since at `6b42b04` the script makes its environment afresh each time and uv won't replace one unasked (#35).
- **Every float32 sweep again with `JAX_DEFAULT_MATMUL_PRECISION=highest`,** 06:56 to 06:58, `run_id` 20261105T065614Z, labelled `highest` as in `precision/`: 22 cells, all `ok` (`session/highest.sh`, `run-highest.log`). On this card XLA runs float32 matrix products in TF32 by default, and the check had found P3's float32 answer off.

`summary.md` is this host's tables, from `python problems/summarise.py problems/results/` once the `--quick` rows were left out.

## What they show

- **P3 in float32, at XLA's default precision, stops early with a wrong answer.** It took 809 iterations at 200 points and 1,005 to 1,381 at 1,000 to 50,000, against NumPy float64's 2,190 to 2,192. Its answers differ from NumPy's float64 answer by a relative 0.11 to 0.12, where the suite allows 1e-3.
- **With `highest` it took 2,208 to 2,214 iterations,** as the TITAN RTX did, and every answer agreed. Its warm times were 0.27 s to 0.77 s, against 0.10 s to 0.46 s at the default. Per iteration the two take about the same time, so the default's speed comes from stopping early.
- **Everything else agrees with NumPy:** P1, which runs in float64 only, P3 in float64, and P4 and P7 in both precisions. P4, which has no matrix product, and P7 are the same under `highest`.
