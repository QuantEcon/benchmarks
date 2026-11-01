# Phase 2 follow-ups, 2026-11-01

Two short rented sessions after Phase 2 of #1 ([2026-10-26](../2026-10-26/README.md)), each started from a laptop, with every machine terminated once its results were copied off.

| Host | Card | Driver, JAX | Asks | Time on the run (UTC) | Script |
|---|---|---|---|---|---|
| `lambda-gpu_1x_a10` | A10 24 GB, Lambda, us-east-1 | 570.148.08, `jax[cuda12]` 0.12.0.1 | Whether XLA's autotuner makes compiling the stored step need twice its array | 06:40–06:51 | [`a10_autotune.sh`](a10_autotune.sh), at `40f8062` |
| `aws-g7e.2xlarge` | RTX PRO 6000 Blackwell Server Edition 96 GB, EC2, us-east-2a | 595.91.07, JAX 0.12.0.1 and 0.11.2 in both CUDA builds | Whether jax-ml/jax#41888's float64 scatter bug reaches `sm_120`; whether JAX's float64 matrix product reaches cuBLAS's float64 emulation; and whether the autotuner is what makes float64 compile slowly | 06:40–07:31 | [`blackwell.sh`](blackwell.sh), then [`blackwell3.sh`](blackwell3.sh) to [`blackwell6.sh`](blackwell6.sh) |

The Blackwell session's plan changed as it ran. `blackwell.sh` ran the scatter check and the first matrix-product setting, then stopped, because XLA's autotuner made each float64 product's first call take minutes; each later script ran after the one before, on what the last had shown. The machine came from the same AMI as Phase 2 (Deep Learning Base OSS Nvidia Driver GPU AMI, Ubuntu 24.04, 20261016), on demand at USD 3.36 an hour, with a key pair and an SSH-only security group made for it and deleted after. The two sessions cost about USD 3.10: 0.25 for the A10 and 2.85 for the Blackwell card.

## The autotuner and the stored step's memory

Phase 2 found that on the A10 and the A100, compiling the stored Bellman step ran out of memory allocating a second buffer as large as the reward array. The same cells ran here twice, each in a fresh process: with XLA's defaults (label `cuda12`), and with `XLA_FLAGS=--xla_gpu_autotune_level=0` (label `cuda12-autotune0`). The rows are in [`results/lambda-gpu_1x_a10/2026-11-01.jsonl`](../../../results/lambda-gpu_1x_a10/2026-11-01.jsonl), with the log, the machine and the environment in [`lambda-gpu_1x_a10/`](lambda-gpu_1x_a10/).

| Cell | Array, GiB | Autotuner on: compile, warm | Autotuner off: compile, warm |
|---|---|---|---|
| 20k, float64 | 2.98 | 1.17 s, 13.2 ms | 0.05 s, 13.2 ms |
| 20k, float32 | 1.49 | 0.18 s, 3.50 ms | 0.04 s, 3.49 ms |
| 40k, float64 | 11.92 | Out of memory allocating 11.94 GiB | 0.05 s, 51.2 ms |
| 40k, float32 | 5.96 | 0.38 s, 13.2 ms | 0.04 s, 13.3 ms |
| 60k, float32 | 13.41 | Out of memory allocating 13.43 GiB | 0.04 s, 29.1 ms |

60k in float64 needs 26.8 GiB and was skipped both times.

**The autotuner is the cause.** With it off, both cells that ran out of memory compile in 0.05 s and run, and every cell that ran both ways runs at the same speed. XLA's autotuner times candidate kernels on buffers of the real size, so on a card with room for one copy of the array and not two, compiling fails before the step can run. The TITAN RTX ran 40k in float64 in Phase 1 with the autotuner on. XLA's Triton kernels need an Ampere card or later, so on Turing there are probably fewer candidates to time; that isn't established.

## The autotuner and float64 compile times

Phase 2 found float64 compiles of 19 s to 460 s on the A10, the A100, the H100 and the Blackwell card, against under 1 s on Turing. The recomputing step ran again on the Blackwell card at 20k and 40k, with XLA's defaults (no label, as in Phase 2) and with `--xla_gpu_autotune_level=0` (label `autotune0`). The rows are in [`results/aws-g7e.2xlarge/2026-11-01.jsonl`](../../../results/aws-g7e.2xlarge/2026-11-01.jsonl).

| Cell | Autotuner on: compile, warm | Autotuner off: compile, warm |
|---|---|---|
| 20k, float64 | 18.99 s, 21.01 ms | 0.11 s, 21.01 ms |
| 40k, float64 | 18.96 s, 91.39 ms | 0.19 s, 91.43 ms |
| 20k, float32 | 5.41 s, 0.76 ms | 0.10 s, 0.74 ms |
| 40k, float32 | 12.06 s, 2.34 ms | 0.12 s, 2.35 ms |

**The long compiles are the autotuner, and here it buys nothing:** the warm times agree to within 3%. The float64 matrix product below shows the same, more sharply: at n = 16,384 its first call takes 509 s with the autotuner on and 5.5 s with it off, and both run at 1.71 TFLOPS.

## jax-ml/jax#41888 on `sm_120`

The issue's reproducer, unchanged, in [`scatter_41888.py`](scatter_41888.py), under two JAX releases and both CUDA builds, each in its own environment. The output is in [`aws-g7e.2xlarge/scatter-41888.txt`](aws-g7e.2xlarge/scatter-41888.txt), with each environment's packages beside it.

| jax | Wheels | 4 × 237501, 237500 indices | 16 × 200001, 6641 indices | With hints |
|---|---|---|---|---|
| 0.12.0.1 | `jax[cuda13]` (CUDA runtime 13.4.92) | 0 | 0 | 0 0 |
| 0.12.0.1 | `jax[cuda12]` (CUDA runtime 12.9.79) | 0 | 0 | 0 0 |
| 0.11.2 | `jax[cuda13]` | **237500** | 0 | 0 0 |
| 0.11.2 | `jax[cuda12]` | 0 | 0 | 0 0 |

This is the H100's pattern in the issue: the first case fails only under 0.11.2 with the CUDA 13 wheels, and the second passes. The second has failed only on the GB10 (`sm_121`).

## JAX's float64 matrix product and cuBLAS's emulation

QuantEcon/project-econ-on-ai-hardware#4's first question, which QuantEcon/project-compute#102 tracks. cuBLAS from CUDA 13.0 Update 2 emulates float64 matrix products on integer tensor cores, and a program that doesn't ask for it in its code can turn it on with `CUBLAS_EMULATE_DOUBLE_PRECISION=1`. [`dot64.py`](dot64.py) times a float64 product C = AB in JAX or CuPy, one setting per fresh process, on three kinds of input: uniform on [0, 1), standard normal, and lognormal with σ = 5, whose entries' logarithms have a standard deviation of 5. It measures the error on an 8 × 256 corner against an 80-bit long double product, in units of float64's roundoff u, scaled by |A| |B|. All 124 rows are in [`aws-g7e.2xlarge/dot64.jsonl`](aws-g7e.2xlarge/dot64.jsonl), with a table in [`dot64-summary.txt`](aws-g7e.2xlarge/dot64-summary.txt). The script gained its last three options as the session went on; the rows it wrote earlier measure the same things.

At n = 16,384, uniform and normal inputs:

| Route | Setting | TFLOPS | Error / u: uniform, normal | First call, s |
|---|---|---|---|---|
| JAX, XLA's defaults: a Triton kernel | — | 1.72 | 112.8, 2.77 | 509 |
| JAX, XLA's defaults | autotuner off | 1.71 | —, 2.77 | 5.5 |
| JAX, Triton's GEMMs off: cuBLASLt | — | 1.51 | 112.8, 2.77 | 53 |
| JAX, Triton's GEMMs off | `CUBLAS_EMULATE_DOUBLE_PRECISION=1`, either strategy | 1.71 | 112.8, 2.77 | 49 |
| JAX, Triton's GEMMs and cuBLASLt off | the same | 1.71 | 112.8, 2.77 | 49 |
| CuPy: `cublasGemmEx` | — | 1.51 | 112.8, 2.77 | 5.9 |
| CuPy | `CUBLAS_EMULATE_DOUBLE_PRECISION=1`, or math mode 8, either strategy | **31.7, 30.6** | **1.83, 0.06** | 0.3 |

**cuBLAS emulates float64 on this card, and JAX doesn't reach it.**

- **By default XLA doesn't call cuBLAS for a float64 dot.** It compiles its own Triton kernel (`__triton_nested_gemm_fusion` in the compiled HLO, [`hlo-default.txt`](aws-g7e.2xlarge/hlo-default.txt)), and its autotuner spends minutes choosing among Triton candidates at float64 speed.
- **With `--xla_gpu_enable_triton_gemm=false`, XLA calls `cublasLtMatmul` with `COMPUTE_64F`** ([`cublaslt-notriton.log`](aws-g7e.2xlarge/cublaslt-notriton.log)). `--xla_gpu_enable_cublaslt=false` doesn't change that. The variable changes the algorithms cuBLASLt's heuristic offers, from `[22,22,22,22,22,22]` to `[22,10,59,16]` ([`cublaslt-notriton-emulate.log`](aws-g7e.2xlarge/cublaslt-notriton-emulate.log)), and XLA's autotuner picks a faster one. But at 8,192 and 16,384 its errors equal native's to every digit shown, so it runs native arithmetic.
- **CuPy calls `cublasGemmEx` with `CUBLAS_COMPUTE_64F`** ([`cublas-cupy-mathmode.log`](aws-g7e.2xlarge/cublas-cupy-mathmode.log)), and with the variable or the math mode cuBLAS emulates: 18× to 19× native at 8,192 and 20× to 21× at 16,384, with errors 40 to 60 times smaller than native's. The two strategies, performant and eager, gave the same results.
- **CuPy's environment had cuBLAS 13.8.0.4,** against 13.8.2.12 in JAX's. The routes differ in the API called, so the version is unlikely to matter, but it wasn't held fixed.

**Emulation needs the inputs' exponents to stay in a range.** On lognormal inputs with σ = 5, cuBLAS falls back to native float64 (1.51 TFLOPS). A sweep over σ with CuPy at n = 8,192:

| σ of the lognormal inputs | 0.5 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Emulated, TFLOPS | 27.95 | 26.16 | 23.83 | 21.30 | 19.74 | 1.50 |
| Emulated, error / u | 1.51 | 1.42 | 2.13 | 2.11 | 2.06 | 212 |
| Native, error / u | 85.4 | 85.2 | 96.2 | 119.1 | 115.9 | 243 |

So the emulation pays, and is more accurate than native, while the logarithms of the entries have a standard deviation of up to 4, and falls back somewhere between 4 and 5. Its speed falls gradually before that, from 28 to 20 TFLOPS.
