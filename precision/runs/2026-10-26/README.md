# Phase 2 of #1, 2026-10-26

The precision matrix on four rented cards, each with matmul again with TF32 off (labelled `highest`), five lectures as written three times on the card, and the `aiyagari_jax` and `ifp_egm_jax` scale scripts, all by [`precision/run_by_hand.sh`](../../run_by_hand.sh). Each machine ran nothing else and was terminated once its results were copied off. The results are in `results/<host>/2026-10-26.jsonl` and `lectures/results/<host>/2026-10-26/`, with each machine's `machine.txt` (`nvidia-smi`, its topology, `lscpu`, memory) and the run's log, `run.log`.

| Host | Card | Stands in for | CPU, memory | Driver, JAX build | Rows | Time on the run (UTC) | Commit |
|---|---|---|---|---|---|---|---|
| `lambda-gpu_1x_a100_sxm4` | A100 SXM4 40 GB | The A800 40GB Active | EPYC 7J13, 30 vCPUs, 216 GiB | 570.148.08, `jax[cuda12]` | 104, one error | 00:23–00:51 | `6faf2df` |
| `aws-g7e.2xlarge` | RTX PRO 6000 Blackwell Server Edition, 96 GB | The RTX PRO 6000 Max-Q | Xeon Platinum 8559C, 8 vCPUs, 62 GiB | 595.91.07, `jax[cuda13]` | 104 | 01:40–01:53 | `03da872` |
| `lambda-gpu_1x_h100_sxm5` | H100 SXM5 80 GB | A float64 ceiling | Xeon Platinum 8480+, 26 vCPUs, 221 GiB | 570.148.08, `jax[cuda12]` | 104 | 00:19–00:33 | `6faf2df` |
| `lambda-gpu_1x_a10` | A10 24 GB | The A10G in RunsOn's `g5` | Xeon Platinum 8358, 30 vCPUs, 222 GiB | 570.148.08, `jax[cuda12]` | 104: two errors, one timeout, one skipped | 00:20–01:07 | `6faf2df` |

All ran JAX 0.12.0.1 (`precision/constraints.txt`) with Python 3.12. `6faf2df` and `03da872` hold the same harness; the second is the first with `main` merged in. On Lambda the driver is too old for CUDA 13, so the script installed the CUDA 12 build and labelled its rows `cuda12`.

## Where they ran

- **Lambda,** at USD 1.99 (A100), 4.29 (H100) and 1.29 (A10) an hour, through [`precision/lambda.sh`](../../lambda.sh). A 1× A100 showed free at 11:16 AEDT, and a launch a minute later was refused for capacity; a loop launched one at 11:18.
- **The Blackwell card on EC2, outside RunsOn.** RunsOn's subnets are in us-west-2a and 2b only, and on 2026-10-26 its launches of `g7e.2xlarge` there failed for capacity again (run [40253580690](https://github.com/QuantEcon/benchmarks/actions/runs/40253580690), cancelled before it launched), as on 2026-10-25 (#9). AWS scored `g7e` spot capacity at 1 of 10 in every us-west-2 zone and 9 in one us-east-2 zone. An on-demand instance (USD 3.36 an hour) from AWS's Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04, 20261016) was refused in us-east-2b and launched in us-east-2a, from the laptop's AWS CLI, with a key pair and an SSH-only security group made for it and deleted after.
- **About USD 6 in all:** roughly 1.9 (H100), 1.4 (A10), 1.3 (A100) and 1.6 (Blackwell), launch to termination.

## What didn't run

- **The stored Bellman step ran out of memory while compiling** on the A10 at 40k in float64 (an 11.92 GiB array) and 60k in float32 (13.41 GiB), and on the A100 at 60k in float64 (26.82 GiB). The planner had admitted each cell, and XLA's memory analysis, here as on every card, counts only the array and about 0.06 GiB of temporaries. The error comes from `compile()`, which tries to allocate a second buffer as large as the array: 11.94 GiB on the A10 beside the 11.92 GiB already on the card. The TITAN RTX (24 GB) ran 40k in float64 on Turing. So on these cards, compiling the step needs about twice the array; the cause isn't diagnosed. XLA's GPU autotuner, which runs candidate kernels on buffers of the real size, is one candidate; `XLA_FLAGS=--xla_gpu_autotune_level=0` would test it.
- **On the A10, the recomputing step at 10k in float32 timed out,** and 60k in float64 was skipped for memory (26.8 GiB per device).

## Compile times

On the A10, the A100 and the H100, compiling the recomputing Bellman step in float64 took 155 s to 460 s at 40k and 60k, and 23 s to 416 s at 20k on the A10 and A100. In float32 the same cells compiled in 1 s to 41 s. On the Blackwell card, in the CUDA 13 build, they took 19 s at 20k and 40k and 92 s at 60k in float64, and 5 s to 12 s in float32. On the TITAN RTX and the T4, in both CUDA builds, they compile in under 1 s. So the long compiles come with the newer cards, not with one build. These times are in each row's `compile_s`; the warm times exclude them.
