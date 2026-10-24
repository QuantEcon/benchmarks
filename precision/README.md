# Precision Benchmarks

What float64 costs against float32 on each kind of device we build on or own, by kernel type, so that "float64 belongs on the CPU" can be a measured rule rather than a guess. The design and the questions are in [#1](https://github.com/QuantEcon/benchmarks/issues/1).

The hypothesis is that the penalty follows arithmetic intensity: weak FP64 (1/32 of FP32 on Turing, less on later consumer and workstation cards) hurts compute-bound kernels and barely touches memory-bound ones. So each workload stands for one kernel type.

## Workloads

| Workload | Bound by | What it is | Sizes | Rate |
|---|---|---|---|---|
| `matmul` | compute | Dense n × n matrix multiply | n = 256 to 8192 | GFLOPS |
| `bellman_stored` | memory | One Bellman step, `max(R + βV)` over choices, with the m × m reward array stored | m = 1k to 60k | GB/s of rewards read |
| `bellman_recompute` | compute (transcendental) | The same step, recomputing log-utility rewards from the grid in blocks of 512 rows, which is how the stored step is made to fit a small card | m = 1k to 60k | Gcells/s |
| `kalman` | launch latency | A 10-state Kalman filter log-likelihood under `lax.scan` | T = 1k to 100k | ksteps/s |

At m = 60k the stored rewards are 14.4 GB in float32 and 28.8 GB in float64, so on a 24 GB card the capacity line falls between the two precisions. The two Bellman workloads compute the same answer, and so do their sharded versions; that is checked by the fidelity samples.

Still to come: the `sargent_surico` NUTS cell from [lecture-python.myst#1022](https://github.com/QuantEcon/lecture-python.myst/issues/1022), run as is.

## Running it

Install JAX for the platform (`pip install -U "jax[cuda13]"` on NVIDIA, `pip install -U jax` elsewhere), then from the repository root:

```bash
python precision/run_matrix.py --quick                    # a few minutes; checks the setup
python precision/run_matrix.py --devices cpu,gpu          # the full matrix, GPU and host CPU
python precision/summarise.py results/                    # markdown tables
```

`--devices` takes `cpu`, `gpu` (one card) and `gpu:N` (N cards, sharded by rows; the Bellman workloads only). `--gpus` picks cards by index in PCI bus order. `--workloads`, `--dtypes` and `--sizes` narrow the matrix, `--host` names the machine in the output, and `--dry-run` shows what would run and what would be skipped for memory. `--memory-fraction` (default 0.8) sets how much of a device's memory a cell may need; lower it on a machine that is in use, since a cell that swaps measures the disk. `cpu:N` forces N host devices, which tests the sharded path on a laptop but measures nothing useful.

On `qe-wharton1-gpu`, inside `tmux`, with the GitHub runner paused:

```bash
python precision/run_matrix.py --host qe-wharton1-gpu --devices cpu,gpu,gpu:2 --gpus 0,1
python precision/run_matrix.py --host qe-wharton1-gpu --devices cpu \
    --prefix "numactl --interleave=all" --label interleave
```

Card 0 in PCI order is `0b:00.0`; card 1, `43:00.0`, drives the desktop, so `gpu` uses card 0. The second run repeats the CPU leg with memory interleaved across the 2990WX's NUMA nodes, two of which have no local memory.

The workflow in `.github/workflows/benchmark.yml` runs the matrix on the RunsOn GPU instance and its host CPU when dispatched with `run_precision`.

## Method

- **One fresh process per cell** (workload × size × precision × device). The driver sets `JAX_ENABLE_X64`, `JAX_PLATFORMS` and `CUDA_VISIBLE_DEVICES` before the process starts, as a lecture's settings would be, and always `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- **Cold and warm are kept apart.** `compile_s` and `first_run_s` are what a lecture pays on first use; `warm_s` holds five timed runs after that, summarised by their median. `process_s` is the whole process, imports included.
- **The result dtype is asserted**, so a silent float32 cannot pass as float64.
- **Fidelity**: each cell keeps 64 values from fixed positions in its result, and the summary reports the largest relative difference between float32 and float64.
- **Memory**: cells that would need more than `--memory-fraction` of a device's memory (80% by default) are recorded as skipped rather than run. XLA's own buffer accounting is kept as `memory_analysis`.
- **GPU telemetry**: clocks, temperature, power and clock-event reasons are read just after timing, since stacked cards can throttle.

## Output

Each cell appends one JSON line to `results/<host>/<date>.jsonl`, carrying the cell's timings and an `env` block: CPU, cores (with Apple's performance levels), memory, GPUs, driver, CUDA, package versions, flags, the commit and `numactl --hardware`. The `env` block is a superset of the `system` block in `hardware/benchmark_hardware.py`.

## Caveats

- **JAX's CPU backend does not use Apple's matrix unit.** On an M1 Pro, NumPy's float64 matmul (through Accelerate) runs at 441 GFLOPS against JAX's 251, so a Mac running JAX gets less from its CPU than its specifications suggest.
- **float32 matmul on Ampere and later may use TF32 by default**, which flatters float32 against float64. Turing has no TF32.
- **Written as one broadcast, the recomputing Bellman step saves no memory on CPU**: XLA materialises the full m × m array. That is why it works in blocks.
