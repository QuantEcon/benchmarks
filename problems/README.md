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

## Backends

| Backend | Stack and device |
|---|---|
| `numpy` | NumPy, on the CPU. Its linear algebra runs on the BLAS it was built with, recorded as `env.numpy_blas` |
| `numba` | Numba, `njit` with `parallel=False`, on the CPU |
| `numba-parallel` | The same Numba source with `parallel=True`, so its `prange` loops use every core |
| `jax-cpu` | JAX on the CPU |
| `jax-gpu` | JAX on one GPU |

Serial Numba is kept beside parallel because a parallel region costs microseconds to start, which a small problem doesn't repay.

## Running it

Install the releases pinned in [`constraints.txt`](constraints.txt), the same on every platform: `pip install "jax[cuda13]" numba scipy -c problems/constraints.txt` on NVIDIA, `pip install jax numba scipy -c problems/constraints.txt` elsewhere. Then from the repository root:

```bash
python problems/run_suite.py --quick                       # a minute or two; checks the setup
python problems/run_suite.py                               # the CPU backends, every size
python problems/run_suite.py --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu
python problems/summarise.py problems/results/             # markdown tables
```

`--problems`, `--sweeps`, `--backends`, `--dtypes` and `--sizes` narrow the suite; a problem runs only in the precisions it allows. `--gpus` picks the card for `jax-gpu` by index in PCI bus order, `--host` names the machine in the output, and `--dry-run` shows what would run and what would be skipped for memory. `--prefix` wraps each cell's process, for example in `numactl --interleave=all`. `--timeout` (900 s) bounds a cell, `--reps` (5) and `--budget` (60 s) its warm runs.

On `qe-wharton1-gpu`, inside `tmux`, with the GitHub runner paused, the card at `0b:00.0` is `--gpus 0`.

## Method

- **One fresh process per cell** (problem × size × backend × precision). The driver sets `JAX_ENABLE_X64` and `JAX_PLATFORMS` before the process starts, as a lecture's settings would be, and for the GPU `CUDA_VISIBLE_DEVICES` and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, as in `precision/`. A cell imports only its own stack, so a NumPy cell never loads JAX or Numba.
- **Each size runs every backend before the next size starts,** so a slow spell on the machine lands on one size rather than on one backend. Once a backend times out, its larger sizes in that sweep are recorded as skipped.
- **Cold and warm are kept apart.** The cold time is what a fresh process pays from its inputs to its first answer: for JAX, starting the backend, compiling and the first run; for Numba, the first call, which compiles; for NumPy, the first call. Each stack's import is timed separately as `import_s`. Then up to five warm runs, stopping once they have taken 60 s, summarised by their median.
- **The result dtype is asserted.** Numba promotes a float32 array to float64 when it meets a Python float, so each Numba version takes its constants in the cell's precision.
- **Every answer is checked against NumPy's in float64** at the same size. P1, P3 and P7 keep values at fixed positions in the answer and must agree to within a relative 1e-8 in float64 and 1e-3 in float32. P4 keeps the share of households above seven wealth levels, from the 10th to the 99th percentile, and each share must be within five binomial standard errors of NumPy's. The summary also flags a cell that took a different number of iterations.
- **Memory:** a cell that would need more than 80% of the device's memory (`--memory-fraction`) is recorded as skipped. XLA's own buffer accounting is kept as `memory_analysis`, and GPU telemetry is read after timing, as in `precision/`.

## Output

Each cell appends one JSON line to `problems/results/<host>/<date>.jsonl`, with its timings, iterations, sample, and the `env` block of [`precision/env_info.py`](../precision/env_info.py), which records Numba's version and `NUMBA_` settings too. Numba cells also record their thread count and, when parallel, the threading layer.
