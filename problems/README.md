# Problem types

Which stack to use for which kind of problem, at what size, and on what hardware: the problem types QuantEcon's lectures use, each written in NumPy, Numba and JAX and timed as used, on CPUs and GPUs, across problem sizes. The design and the questions are in [#5](https://github.com/QuantEcon/benchmarks/issues/5), for [meta#264](https://github.com/QuantEcon/meta/issues/264).

[`precision/`](../precision/) measures JAX alone, by kernel type. This suite asks the question a lecture's author faces: for this problem, at this size, is NumPy, Numba or JAX faster, and on which device?

## Problems

Phase 1 has four of #5's eight problem types. Each takes its model and parameters from a lecture, and each stack's version is written as a careful author in that stack would write it, rather than copied from the lecture.

| | Problem | Model, from | Sweep | Sizes | Precision |
|---|---|---|---|---|---|
| P1 | [`vfi`](vfi.py): value function iteration on a dense grid | lecture-jax's `opt_savings_1`: savings, with income on a 100-state chain | the wealth grid | 100 to 1,500 | float64 |
| P3 | [`egm`](egm.py): the endogenous grid method | lecture-jax's `ifp_egm_jax`: an income fluctuation problem, with income on a 25-state chain | the savings grid | 200 to 50,000 | both |
| P4 | [`wealth`](wealth.py): Monte Carlo, parallel across paths and sequential in time | lecture-jax's `wealth_dynamics_jax`: a cross-section of households' wealth, 200 periods | households | 1,000 to 10,000,000 | both |
| P7 | [`kalman`](kalman.py): a sequential recursion, bound by each step's latency | precision's `kalman`: a log-likelihood over 1,000 periods with 3 observations | independent filters at 10 states (`batch`); states, for one filter (`state`) | 1 to 10,000; 2 to 200 | both |

- **P1 and P3 solve to the lecture's tolerance,** so a cell times the whole solve, and records the iterations it took, from which a time per iteration follows.
- **P1 stores its rewards,** computed once inside the timed solve, and forms the expectation with one matrix product an iteration. The lecture recomputes both each time. P1 runs in float64 only, because a tolerance of 1e-5 on values near −50 is within a few units of float32's roundoff.
- **P3 interpolates once per (savings, next income) pair** an iteration, since next period's assets don't depend on today's income. Numba interpolates each income state's column in one `np.interp` call, as NumPy does: a scalar call per point searches the grid afresh each time, and ran 7× slower at 20,000 points.
- **P4's aggregate path is drawn once, by NumPy, and shared,** so given it the households are independent. Each stack draws the households' shocks with its own generator, inside the timed solve.
- **P7 sweeps the work in each step,** not the series' length, which only scales every stack's time in proportion. Every GPU measured in `precision/` pays about 90 to 160 µs a Kalman step, against about 1.4 µs on the M6, so the question is how much work a step needs before a GPU wins.
- **P7 has three optional batch sweeps,** for [#58](https://github.com/QuantEcon/benchmarks/issues/58): `batch20`, `batch50` and `batch100` run 1 to 10,000 filters, with 300 and 3,000 between, at 20, 50 and 100 states. They run only when `--sweeps` names them, so a default run is Phase 1's. `spec.py`'s `KALMAN_BATCH_STATES` gives each batch sweep's states.

## Backends

| Backend | Stack and device |
|---|---|
| `numpy` | NumPy, on the CPU. Its linear algebra runs on the BLAS it was built with, recorded as `env.numpy_blas` |
| `numba` | Numba, `njit` with `parallel=False`, on the CPU |
| `numba-parallel` | The same Numba source with `parallel=True`, so its `prange` loops use every core |
| `jax-cpu` | JAX on the CPU |
| `jax-gpu` | JAX on one GPU: NVIDIA's under CUDA, or AMD's under ROCm |

Serial Numba is kept beside parallel because a parallel region costs microseconds to start, which a small problem doesn't repay.

## Results

The suite has run on two machines, both on 2026-11-02 at `5fbf033`: `qe-mini1`'s M6 on the four CPU backends (#25), and `qe-wharton1-gpu` on all five, with the Threadripper 2990WX for the CPU backends and the TITAN RTX at `0b:00.0` for `jax-gpu` (#26). Every cell ran, and every answer agreed with NumPy's float64 answer at the same size. The only flags are P3's iteration counts in float32, 14 to 26 more than float64's 2,190 to 2,192.

On 2026-11-06 the four CPU backends also ran on `qe-wharton2`'s Apple M5 Pro, every size in both precisions ([notes](results/qe-wharton2/2026-11-06/README.md)), and aren't in the tables below yet. Every cell ran, and every answer agreed, with P3's float32 iteration counts as on the other two machines. At each sweep's largest size in float64 it was 1.45× to 1.90× as fast as the M6 on P1, P4 and P7's batch, warm, and about as fast on P3.

On 2026-11-05 `jax-gpu` also ran on an RTX PRO 6000 Blackwell Server Edition, EC2's `g7e.2xlarge` ([notes](results/aws-g7e.2xlarge/2026-11-05/README.md)), and isn't in the tables below yet. There P3's float32 answers were 11% to 12% off, after 809 to 1,381 iterations, because on that card XLA runs float32 matrix products in TF32 by default. With `JAX_DEFAULT_MATMUL_PRECISION=highest`, they agreed. Every other answer agreed at XLA's default. Rounding P3's product to TF32 in NumPy reproduces how far off the card's answers were at every size, and its iteration counts to within 1% up to 20,000 points ([`checks/tf32_egm.py`](checks/tf32_egm.py), with its output in [`checks/tf32_egm.txt`](checks/tf32_egm.txt)): the iteration stalls on TF32's rounding, and the convergence test passes on a change of exactly zero.

On 2026-11-05 `jax-gpu` also ran on an AMD Instinct MI350X under ROCm 10.1, DigitalOcean's `gpu-mi350x1-288gb-spot` ([notes](results/digitalocean-gpu-mi350x1-288gb-spot/2026-11-05/README.md)), and isn't in the tables below yet. Every answer agreed with NumPy's float64 answer at JAX's default precision, and P3's float32 iteration counts were the same under `highest`: on that card a float32 matrix product is full float32 by default.

On 2026-11-06 `jax-gpu` also ran on an H100 SXM 80 GB, Lambda's `gpu_1x_h100_sxm5` ([notes](results/lambda-gpu_1x_h100_sxm5/2026-11-06/README.md)), and isn't in the tables below yet. TF32 shows the same way there: at XLA's default, P3's float32 answers were 11% to 12% off, after the same iteration counts as on the RTX PRO 6000, and with `highest` they agreed. Every other answer agreed at the default.

On 2026-11-06 `jax-gpu` also ran on an H200, DigitalOcean's `gpu-h200x1-141gb` ([notes](results/digitalocean-gpu-h200x1-141gb/2026-11-06/README.md)), and isn't in the tables below yet. TF32 shows there as on the H100, with the same iteration counts, and every other answer agreed.

On 2026-11-06 `jax-gpu` also ran on an AMD Instinct MI300X under ROCm 10.1, DigitalOcean's `gpu-mi300x1-192gb` ([notes](results/digitalocean-gpu-mi300x1-192gb/2026-11-06/README.md)), and isn't in the tables below yet. There P3's float32 answers were 25% to 26% off at JAX's default, after 494 to 901 iterations, because on that card float32 matrix products are rounded to TF32's precision by truncation. With `highest` they agreed, and every other answer agreed at the default.

On 2026-11-06 four CPU cells also ran on `qe-wharton1-gpu` under four NUMA placements, for QuantEcon/project-benchmarks-jax#46 ([notes](results/qe-wharton1-gpu/2026-11-06/README.md)): P1, P3, P4 and P7 at one size each, in float64. Those rows are labelled `numa-default`, `numa-one-node`, `numa-memory-nodes` and `numa-remote-memory`, and aren't in the tables below. Pinned to the two nodes with memory, no cell gained more than 20%. On one node's 8 cores, P7's parallel stacks ran faster than on all 32, while P4's parallel Numba took four times as long. Every answer agreed with NumPy's.

On 2026-11-09 [`checks/bw_probe.py`](checks/bw_probe.py), one core's bandwidth and P1's two serial Bellman steps at a wealth grid of 1,500, ran on `qe-mini1`'s M6 and `qe-wharton2`'s M5 Pro, for QuantEcon/project-benchmarks-jax#47 ([M6](results/qe-mini1/2026-11-09/README.md), [M5 Pro](results/qe-wharton2/2026-11-09/README.md)). On both, the serial Numba step runs at about a quarter of one core's read rate, and the probe's steps are 95% to 96% of P1's suite time per iteration. The same day [`checks/stream_triad.py`](checks/stream_triad.py), a STREAM-style triad from one thread to all cores, ran on both ([M6](results/qe-mini1/2026-11-09/triad/README.md), [M5 Pro](results/qe-wharton2/2026-11-09/triad/README.md)). One thread takes all of the M6's 135 GB/s, while the M5 Pro needs four to reach its 255 to 265.

On 2026-11-09 the four CPU stacks and the precision suite's CPU column also ran on 16 cores, one thread each, of three EC2 instances, through `run_by_hand.sh --cpu`, for the same issue: AMD's EPYC 9R45 in `c8a.4xlarge` ([notes](results/aws-c8a.4xlarge/2026-11-09/README.md)), Intel's Xeon 6975P-C in `c8i.8xlarge` ([notes](results/aws-c8i.8xlarge/2026-11-09/README.md)) and Intel's Xeon Platinum 8488C in `c7i.8xlarge` ([notes](results/aws-c7i.8xlarge/2026-11-09/README.md)). They aren't in the tables below yet. Every answer agreed with NumPy's in float64. On the c7i, P1's `numpy` cell at 1,500 passed the 900 s limit for a cell, and its rerun with 1,800 s is labelled `timeout-1800`. Each instance also ran `bw_probe.py` and `stream_triad.py`, and recorded each CPU's clock every 2 s with [`checks/clock_sampler.py`](checks/clock_sampler.py).

On 2026-11-09 P1's and P7's Numba cells also ran on `qe-wharton1-gpu`'s Threadripper with OpenBLAS as the suite leaves it and with `OPENBLAS_NUM_THREADS=1`, for the same issue ([notes](results/qe-wharton1-gpu/2026-11-09/README.md)). Those rows are labelled `blas-default-…` and `blas1-…`, and aren't in the tables below. Capped, parallel Numba ran P1 at 1,500 1.4× to 1.5× as fast, but P7's batch of 10,000 filters only 1.13× as fast, still slower than serial Numba.

The tables give each device's fastest stack at each size in float64, in seconds. Warm is the median of the warm runs. Cold is what a fresh process pays from its inputs to its first answer, compile included. The fastest device in each row is in bold, for warm and for cold separately. The TITAN RTX's columns are `jax-gpu`, the only stack on it. Every number here is in a table that `python problems/summarise.py problems/results/` prints, with every backend at every size and in both precisions.

**P1 `vfi`, by grid size:**

| Size | M6, warm | Threadripper, warm | TITAN RTX, warm | M6, cold | Threadripper, cold | TITAN RTX, cold |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | **0.08** `numba-parallel` | 0.227 `jax-cpu` | 0.109 | **0.11** `jax-cpu` | 0.539 `jax-cpu` | 1.08 |
| 200 | 0.174 `jax-cpu` | 0.72 `jax-cpu` | **0.145** | **0.2** `jax-cpu` | 1.17 `jax-cpu` | 1.15 |
| 400 | 0.579 `jax-cpu` | 2.1 `numba-parallel` | **0.287** | **0.609** `jax-cpu` | 3.41 `jax-cpu` | 1.27 |
| 700 | 1.7 `jax-cpu` | 6.8 `numba-parallel` | **0.852** | **1.72** `jax-cpu` | 10.3 `jax-cpu` | 1.87 |
| 1,000 | 3.37 `jax-cpu` | 15 `numba-parallel` | **1.5** | 3.38 `jax-cpu` | 13.2 `jax-cpu` | **2.58** |
| 1,500 | 7.71 `jax-cpu` | 31.8 `jax-cpu` | **3.25** | 7.7 `jax-cpu` | 36.4 `jax-cpu` | **4.32** |

**P3 `egm`, by savings grid size:**

| Size | M6, warm | Threadripper, warm | TITAN RTX, warm | M6, cold | Threadripper, cold | TITAN RTX, cold |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | **0.186** `numba` | 0.908 `numba` | 0.548 | **0.302** `numpy` | 1.13 `numpy` | 1.64 |
| 1,000 | 0.786 `numba` | 4.6 `numba` | **0.634** | **0.821** `numpy` | 4.53 `numpy` | 1.73 |
| 5,000 | 3.42 `numba-parallel` | 24.1 `numpy` | **1.06** | 3.84 `numpy` | 25 `numpy` | **2.24** |
| 20,000 | 13.1 `numba-parallel` | 109 `numpy` | **2.78** | 14.2 `numba-parallel` | 112 `numpy` | **4** |
| 50,000 | 33.4 `numba-parallel` | 282 `numpy` | **7.4** | 34.3 `numba-parallel` | 286 `numpy` | **8.59** |

**P4 `wealth`, by households:**

| Size | M6, warm | Threadripper, warm | TITAN RTX, warm | M6, cold | Threadripper, cold | TITAN RTX, cold |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | **0.00112** `numba-parallel` | 0.00348 `numba-parallel` | 0.00473 | **0.00325** `numpy` | 0.0153 `numpy` | 1.4 |
| 10,000 | 0.0113 `numba-parallel` | 0.012 `numba-parallel` | **0.0092** | **0.0268** `numpy` | 0.109 `numpy` | 1.5 |
| 100,000 | 0.112 `numba-parallel` | 0.0764 `numba-parallel` | **0.0405** | **0.254** `numpy` | 1.09 `numpy` | 2.09 |
| 1,000,000 | 1.04 `numba-parallel` | 0.534 `numba-parallel` | **0.244** | **1.34** `numba-parallel` | 2.29 `numba-parallel` | 2.18 |
| 10,000,000 | 10.3 `numba-parallel` | 5.28 `numba-parallel` | **2.34** | 10 `numba-parallel` | 7.01 `numba-parallel` | **4.43** |

**P7 `kalman`, by independent filters:**

| Size | M6, warm | Threadripper, warm | TITAN RTX, warm | M6, cold | Threadripper, cold | TITAN RTX, cold |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | **0.00139** `numba` | 0.00661 `numba` | 0.156 | **0.0124** `numpy` | 0.081 `numpy` | 1.29 |
| 10 | **0.00284** `numba-parallel` | 0.043 `jax-cpu` | 0.134 | **0.0229** `numpy` | 0.127 `numpy` | 1.29 |
| 100 | **0.0259** `numba-parallel` | 0.508 `numpy` | 0.127 | **0.123** `numpy` | 0.508 `numpy` | 1.29 |
| 1,000 | **0.22** `numba-parallel` | 4.47 `jax-cpu` | 0.372 | **0.492** `jax-cpu` | 5.2 `numpy` | 1.65 |
| 10,000 | **2.28** `numba-parallel` | 47.8 `numpy` | 3.15 | **2.62** `jax-cpu` | 46.9 `numpy` | 4.38 |

**P7 `kalman`, by states, for one filter:**

| Size | M6, warm | Threadripper, warm | TITAN RTX, warm | M6, cold | Threadripper, cold | TITAN RTX, cold |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | **0.000594** `jax-cpu` | 0.00444 `jax-cpu` | 0.129 | **0.0104** `numpy` | 0.075 `numpy` | 1.2 |
| 5 | **0.00116** `numba` | 0.00548 `numba` | 0.134 | **0.0106** `numpy` | 0.0771 `numpy` | 1.24 |
| 10 | **0.00142** `numba` | 0.00674 `numba` | 0.155 | **0.0116** `numpy` | 0.0801 `numpy` | 1.3 |
| 20 | **0.00262** `jax-cpu` | 0.0116 `numba` | 0.148 | **0.0135** `numpy` | 0.0847 `numpy` | 1.34 |
| 50 | **0.00804** `numba` | 0.0549 `numba` | 0.213 | **0.0184** `numpy` | 0.129 `numpy` | 1.38 |
| 100 | **0.0267** `numba` | 0.387 `jax-cpu` | 0.32 | **0.034** `numpy` | 0.431 `numpy` | 1.49 |
| 200 | **0.107** `numba` | 1.11 `numpy` | 0.494 | **0.11** `numpy` | 1.12 `numpy` | 1.7 |

float32, on the three problems that allow it, changes each device's fastest warm time at the largest size like this, in seconds, with the stack that set it; the TITAN RTX's is `jax-gpu`. The fastest stack can change with the precision: on the M6, P7's state sweep goes from `numba` to `numpy`, and on the Threadripper P3 goes from `numpy` to `numba-parallel`.

| Problem, at its largest size | M6 | Threadripper | TITAN RTX |
|---|---|---|---|
| P3 `egm`, 50,000 | 33.4 → 20.4 (1.6×), `numba-parallel` | 282 `numpy` → 200 `numba-parallel` (1.4×) | 7.4 → 5.11 (1.4×) |
| P4 `wealth`, 10,000,000 | 10.3 → 8.31 (1.2×), `numba-parallel` | 5.28 → 4.83 (1.1×), `numba-parallel` | 2.34 → 0.0542 (43×) |
| P7 `kalman`, batch 10,000 | 2.28 → 2.37 (0.96×), `numba-parallel` | 47.8 → 41.3 (1.2×), `numpy` | 3.15 → 0.518 (6.1×) |
| P7 `kalman`, state 200 | 0.107 `numba` → 0.0577 `numpy` (1.8×) | 1.11 → 0.712 (1.6×), `numpy` | 0.494 → 0.115 (4.3×) |

What the tables show:

- **The TITAN RTX is fastest warm on P1, P3 and P4 from small sizes on:** from a grid of 200 on P1, of 1,000 on P3, and from 10,000 households on P4. At the largest sizes it is 2.4× to 4.5× ahead of the M6.
- **On P7, in float64, the card doesn't catch the M6.** At 1 to 100 filters, its 1,000 Kalman steps take 0.127 s to 0.156 s, about 0.13 ms a step. Below 100 filters or 100 states, it loses to both CPUs. From 100 on it beats the Threadripper, but the M6 runs 10,000 filters in 2.28 s against its 3.15 s, and 200 states in 0.107 s against its 0.494 s.
- **Cold, the card's start decides the small problems.** A fresh process on the card takes 1.08 s at the least, while the M6 answers every problem's smallest size in 0.31 s or less. The card is fastest cold only from P1's grid of 1,000, P3's 5,000 and P4's 10,000,000 households, and never on P7.
- **On the M6 no one CPU stack is best.**
    - Warm, JAX is fastest on P1 from a grid of 200, and parallel Numba is fastest on P4 at every size, on P7's batch from 10 filters, and on P3 from 5,000. Serial Numba is fastest on 5 of P7's 7 state sizes.
    - Cold, NumPy is fastest in 16 of the 28 cells, since it compiles nothing, while Numba's first call, which compiles, takes 0.2 s to 3.2 s.
- **The Threadripper is slower than the M6 on all but one problem, despite its 32 cores:** 4.1× on P1 at 1,500 (31.8 s against 7.71 s, both JAX), 8.4× on P3 at 50,000, and 21× on P7's batch at 10,000. The exception is P4. There its parallel Numba is faster from 100,000 households, and takes 5.28 s against 10.3 s at 10,000,000.
- **Parallel Numba scales on the Threadripper only on P1 and P4.**
    - There, at the largest sizes, it is 4.2× and 59× faster than serial Numba. On P3 it is 1.05× faster, and on P7's batch it is 3.0× slower (199 s against 65.9 s).
    - On the M6 the same code is 1.2× to 7.0× faster than serial on those four sweeps. These runs don't say why. [#1](https://github.com/QuantEcon/benchmarks/issues/1) lists the machine's NUMA layout among its quirks, and `--prefix 'numactl --interleave=all'` would test it.
- **Numba's first call takes 4.4× to 5.6× as long on the Threadripper as on the M6,** at each sweep's smallest size: 12.4 s against 2.25 s on P7's batch, for example.
- **float32 matters on the card and little on the CPUs.** On the TITAN RTX it takes P4's 10,000,000 households from 2.34 s to 0.054 s, and P7's 10,000 filters from 3.15 s to 0.52 s, ahead of the M6's 2.37 s. On the CPUs it gains up to 1.8×, and on the M6's P7 batch it is slightly slower (2.37 s against 2.28 s). [`precision/`](../precision/) measures float64's cost on the card by kernel type.

## Running it

Install the releases pinned in [`constraints.txt`](constraints.txt), the same on every platform: `pip install "jax[cuda13]" numba scipy -c problems/constraints.txt` on NVIDIA, `pip install jax numba scipy -c problems/constraints.txt` elsewhere. AMD's cards are the exception, as in [`precision/`](../precision/): there is no ROCm build of that JAX release, so [`constraints-rocm.txt`](constraints-rocm.txt) pins 0.12.0 with the same other releases. Install `pip install jax jax-rocm7-plugin jax-rocm7-pjrt numba scipy -c problems/constraints-rocm.txt` on ROCm 7, or the `rocm10` packages on ROCm 10. Then from the repository root:

```bash
python problems/run_suite.py --quick                       # a minute or two; checks the setup
python problems/run_suite.py                               # the CPU backends, every size
python problems/run_suite.py --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu
python problems/summarise.py problems/results/             # markdown tables
```

`--problems`, `--sweeps`, `--backends`, `--dtypes` and `--sizes` narrow the suite; a problem runs only in the precisions it allows. `--gpus` picks the card for `jax-gpu` by index, in PCI bus order on NVIDIA and in `amd-smi`'s order on AMD; the card is NVIDIA's, run under CUDA, if `nvidia-smi` lists any, and otherwise AMD's, run under ROCm. `--host` names the machine in the output, and `--dry-run` shows what would run and what would be skipped for memory. `--prefix` wraps each cell's process, for example in `numactl --interleave=all`. `--timeout` (900 s) bounds a cell, `--reps` (5) and `--budget` (60 s) its warm runs.

On `qe-wharton1-gpu`, inside `tmux`, with the GitHub runner paused, the card at `0b:00.0` is `--gpus 0`.

On a rented machine, NVIDIA's or AMD's, [`run_by_hand.sh`](run_by_hand.sh) records the machine, installs the pinned releases and runs `jax-gpu` alone, every size in both precisions, after a `--quick` check. With `--cpu` it runs the four CPU backends instead, on a machine with no GPU, and records the CPU in full: `lscpu`, whether SMT is on, the NUMA layout, `/proc/meminfo`, and the threads each stack will start with. Its header says how to run it, and how to add the precision matrix's CPU column in the same environment.

## Method

- **One fresh process per cell** (problem × size × backend × precision). The driver sets `JAX_ENABLE_X64` and `JAX_PLATFORMS` before the process starts, as a lecture's settings would be, and for the GPU `CUDA_VISIBLE_DEVICES` (`HIP_VISIBLE_DEVICES` on AMD) and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, as in `precision/`. A cell imports only its own stack, so a NumPy cell never loads JAX or Numba.
- **Each size runs every backend before the next size starts,** so a slow spell on the machine lands on one size rather than on one backend. Once a backend times out, its larger sizes in that sweep are recorded as skipped.
- **Cold and warm are kept apart.** The cold time is what a fresh process pays from its inputs to its first answer: for JAX, starting the backend, compiling and the first run; for Numba, the first call, which compiles; for NumPy, the first call. Each stack's import is timed separately as `import_s`. Then up to five warm runs, stopping once they have taken 60 s, summarised by their median.
- **The result dtype is asserted.** Numba promotes a float32 array to float64 when it meets a Python float, so each Numba version takes its constants in the cell's precision.
- **Every answer is checked against NumPy's in float64** at the same size. P1, P3 and P7 keep values at fixed positions in the answer and must agree to within a relative 1e-8 in float64 and 1e-3 in float32. P4 keeps the share of households above seven wealth levels, from the 10th to the 99th percentile, and each share must be within five binomial standard errors of NumPy's. The summary also flags a cell that took a different number of iterations.
- **Memory:** a cell that would need more than 80% of the device's memory (`--memory-fraction`) is recorded as skipped. XLA's own buffer accounting is kept as `memory_analysis`. On a card, so is the allocator's, as `memory_stats`: its `peak_bytes_in_use` is the most the cell held at once. Both it and the GPU telemetry are read after timing, as in `precision/`.

## Output

Each cell appends one JSON line to `problems/results/<host>/<date>.jsonl`, with its timings, iterations, sample, and the `env` block of [`precision/env_info.py`](../precision/env_info.py), which records Numba's version and `NUMBA_` settings too. Numba cells also record their thread count and, when parallel, the threading layer.
