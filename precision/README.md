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
| `ssy_operator` | compute (fused) | One application of the Koopmans operator of the Schorfheide–Song–Yaron long-run risk model, from [sdfs_via_autodiff](https://github.com/jstac/sdfs_via_autodiff): a sum over every pair of states on a grid of n points on each of four axes, as the authors wrote it | n = 4 to 20 (n⁴ states) | Gcells/s |
| `ssy_jvp` | compute (fused) | The Jacobian–vector product of that operator, by forward-mode autodiff: what the authors' Newton–Kantorovich iteration repeats inside its linear solve | n = 4 to 20 | Gcells/s |

At m = 60k the stored rewards are 14.4 GB in float32 and 28.8 GB in float64, so on a 24 GB card the capacity line falls between the two precisions. The two Bellman workloads compute the same answer, and so do their sharded versions; that is checked by the fidelity samples.

The `ssy` workloads are the two kernels of that repository's method for the model's wealth-consumption ratio, the fixed point of the operator. A cell is one term of the operator's sum, and there are n⁸ of them. [`ssy.py`](ssy.py) lists what changed from the authors' code, and [`runs/2026-10-28/checks/`](runs/2026-10-28/checks/) compares it with theirs: the operator agrees to 2e-16, and Newton's method on it reaches their solution.

Still to come: the `sargent_surico` NUTS cell from [lecture-python.myst#1022](https://github.com/QuantEcon/lecture-python.myst/issues/1022), run as is.

## Running it

Install JAX at the release pinned in [`constraints.txt`](constraints.txt), the same on every platform so that results compare: `pip install "jax[cuda13]" -c precision/constraints.txt` on NVIDIA, `pip install jax -c precision/constraints.txt` elsewhere. Then from the repository root:

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

To compare JAX's two CUDA builds, install `jax[cuda12]` the same way in a second environment and run with `--label cuda12`. Each row's `env` block records which plugin ran.

The workflow in `.github/workflows/benchmark.yml` runs the matrix on a RunsOn GPU instance when dispatched with `run_precision`. Its inputs:

| Input | What it sets |
|---|---|
| `precision_instance` | The card: `g4dn` T4, the lectures' CI instance; `g5` A10G; `g6` L4; `g7e` RTX PRO 6000 Blackwell; `p4d` eight A100 40 GB; `p5` one H100 80 GB |
| `precision_run` | `smoke` describes the GPU, its driver and JAX's allocator limit, and runs nothing; `quick` the small sizes; `full` the matrix |
| `precision_devices` | `cpu,gpu` by default; `gpu,gpu:2` adds the Bellman workloads sharded over two cards |
| `precision_cuda` | The CUDA build |
| `precision_extras` | A second matmul pass with `JAX_DEFAULT_MATMUL_PRECISION=highest`, labelled `highest`, and the `aiyagari_jax` and `ifp_egm_jax` scale scripts on the card, from lecture-jax at `4b939ec` |

The job stops after 60 minutes, which caps a `p4d` run at about USD 22. On a `p4d` it first installs and starts NVIDIA's fabric manager, which CUDA needs on a machine whose cards are joined by NVSwitch.

## Method

- **One fresh process per cell** (workload × size × precision × device). The driver sets `JAX_ENABLE_X64`, `JAX_PLATFORMS` and `CUDA_VISIBLE_DEVICES` before the process starts, as a lecture's settings would be, and always `XLA_PYTHON_CLIENT_PREALLOCATE=false`. GPU cells also get `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`: even without preallocation, JAX's allocator otherwise stops at 75% of the card, below the 80% that `--memory-fraction` allows (a T4 showed 10.92 GiB of 14.6 usable, against 13.84 GiB at 0.95). The T4's `bellman_stored` cell at m = 40,000 in float64, which needs 11.92 GiB, ran out of memory in both CUDA builds for that reason (`results/runson-g4dn.2xlarge/2026-10-25.jsonl`, before the change).
- **Cold and warm are kept apart.** `backend_init_s`, `compile_s` and `first_run_s` are what a fresh process pays before its first result, and the summary's cold time is their sum. A lecture starts the backend once per process, so the cold time charges every kernel for it; it is kept as its own field so that it can be taken out. `warm_s` holds five timed runs after that, summarised by their median. `process_s` is the whole process, imports included.
- **The result dtype is asserted**, so a silent float32 cannot pass as float64.
- **Fidelity**: each cell keeps 64 values from fixed positions in its result, and the summary reports the largest relative difference between float32 and float64.
- **Memory**: cells that would need more than `--memory-fraction` of a device's memory (80% by default) are recorded as skipped rather than run. XLA's own buffer accounting is kept as `memory_analysis`.
- **GPU telemetry**: clocks, temperature, power and clock-event reasons are read just after timing, since stacked cards can throttle.

## Output

Each cell appends one JSON line to `results/<host>/<date>.jsonl`, carrying the cell's timings and an `env` block: CPU, cores (with Apple's performance levels), memory, GPUs, driver, CUDA, package versions, flags, the commit and `numactl --hardware`. The `env` block is a superset of the `system` block in `hardware/benchmark_hardware.py`.

## Caveats

- **JAX's CPU backend makes only partial use of Apple's matrix unit.**
    - On an M1 Pro, NumPy's float64 matmul (through Accelerate) runs at 441 GFLOPS, against JAX's 251.
    - On the M6 with JAX 0.12.0.1, float32 matmul goes through XLA's YNNPACK fusion. It reaches about 2 TFLOPS at n = 1,024, which suggests the matrix unit is in use there, and falls to about 900 GFLOPS from n = 4,096.
    - Float64 stays near 400 GFLOPS throughout, against 545 from NumPy with Accelerate at n = 4,096.
    - So a Mac running JAX gets less from its CPU than its specifications suggest.
- **On Apple Silicon, each small float32 matrix product costs about 1 µs.** XLA's YNNPACK fusion takes every product from 8 × 8 up. On the M6 a call costs about 1.1 µs in float32, against 0.16 µs in float64, so a recursion of small products, such as the `kalman` workload, runs about 3.5× slower in float32. `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=` turns the fusion off: the Kalman filter then runs at the same speed in both precisions, but large float32 matmuls lose up to 2.4×. The results here use the default, as a lecture would. The diagnosis is in [`runs/2026-10-25/checks/`](runs/2026-10-25/checks/).
- **float32 matmul on Ampere and later may use TF32 by default**, which flatters float32 against float64. Turing has no TF32; the A10G, L4, A100, H100 and Blackwell cards do. The default is what a lecture gets; to measure full float32 as well, repeat the run with `JAX_DEFAULT_MATMUL_PRECISION=highest` and a `--label`, as `precision_extras` does.
- **On the CPU, the stored Bellman step slows 10× to 30× once its float64 array passes about 2 GiB.** On four CPUs (M1 Pro, M6, Threadripper, Xeon) its throughput drops from 30–100 GB/s to about 3–12 GB/s. In float64 that happens between 16k (2.05 GB) and 18k (2.59 GB). In float32 it happens later: after 31k (3.84 GB) on the M1 Pro, and by 40k (5.96 GiB) on the others. XLA reports no temporary buffer, so the reduction stays fused. So the CPU cells at 20k in float64 measure this cliff, not the cost of precision. The evidence is in [`runs/2026-10-25/checks/cpu_cliff.txt`](runs/2026-10-25/checks/cpu_cliff.txt).
- **Written as one broadcast, the recomputing Bellman step saves no memory on CPU**: XLA materialises the full m × m array. That is why it works in blocks.
- **float32 can't solve the SSY model as the authors wrote it.** The solution lies between 584 and 924, and the operator raises w to θ = −16.02. 800^θ is 3e-47, below float32's smallest normal number (1.2e-38), so the operator returns inf at every state ([`runs/2026-10-28/checks/ssy.txt`](runs/2026-10-28/checks/ssy.txt)). The `ssy` workloads apply it at a w near 1, where both precisions work and the operator agrees between them to 1.3e-7.
- **On the CPU, XLA's default fusion stores the N × N array once it passes about 3 GiB, and its memory analysis doesn't show it.** On the M1, in a fresh process, `ssy_operator` in float64 peaked at 0.34 GB resident at n = 11 (1.6 GiB of cells) and 3.9 GB at n = 12 (3.2 GiB). Float32 did the same between n = 13 and 14, reaching 4.1 GB and 3.4 ns a cell against 0.23. `ssy_jvp` needed 5.3 GB at n = 12 in float64. The CPU memory model in `spec.py` allows for this, so the M1's larger cells are skipped; a GPU fuses them without storing anything.
- **With YNNPACK fusion off, the CPU stores nothing and the operator is faster, but not steadily.** Under `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=` the M1 peaked at 0.18 GB at n = 20, and at n = 12 `ssy_operator` ran 2.6× faster in float64 (290 ms to 110 ms) and 3.4× in float32 (100 ms to 29 ms). At n = 10 float32 was slower, 30 ms against 22 ms, and with the fusion off its rate is 3.3 and 4.3 Gcells/s at n = 10 and 20 against 14.7 and 17.6 at n = 12 and 16, in repeated fresh processes. Why isn't diagnosed. These rows are labelled `noynn` and ran with `--memory-fraction 1000`, which lifts the planner's limit, after the memory was checked by hand; the unlabelled rows use the defaults, as a lecture would ([`runs/2026-10-28/`](runs/2026-10-28/)).
