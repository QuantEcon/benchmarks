# Float64 on Apple Silicon's CPU: Accelerate against XLA

A probe for QuantEcon/project-benchmarks-jax#48. JAX's CPU backend doesn't call Accelerate: at jaxlib 0.12.0.1's defaults, XLA hands float64 dots, and the fused row reduction of a Bellman step, to YNNPACK. This measures what that leaves unused on a Mac, beside NumPy, which calls Accelerate.

## What it runs

1. **The chip** (`features.txt`): its name, macOS, its cores, the SME features macOS reports (`FEAT_SME`, `FEAT_SME2` and `FEAT_SME_F64F64` among them), NumPy's BLAS, and the frameworks' versions.
2. **Float64 matrix products, three ways,** at n = 8, 64, 512, 1,024 and 4,096 (`matmul.jsonl`).
    - **The three:** NumPy's `a @ b`, which calls Accelerate; `jnp.dot` at JAX's defaults, which XLA hands to YNNPACK; and `jnp.dot` with `--xla_cpu_experimental_ynn_fusion_type=-individual_dot`, which leaves it to XLA's own dot.
    - **Each row** has GFLOPS, the largest difference from NumPy's answer, relative to its largest magnitude, and which code ran.
    - **At n = 8 and 64,** each variant also runs the recurrence from QuantEcon/project-benchmarks-jax#4, one product a step for 10,000 steps, timed per product. NumPy's time includes Python's dispatch, so its bare `np.dot` call is timed as well.
3. **The cliff** of QuantEcon/project-benchmarks-jax#7, at m = 16,000 and 18,000 (`cliff.jsonl`).
    - **What:** `max(R + 0.96 v)` over the rows of a float64 m × m array, at the defaults and with YNNPACK's reduce fusion off (`--xla_cpu_experimental_ynn_fusion_type=-reduce`), then its two halves: the add alone and the row max alone.
    - **Which ops YNNPACK took:** each row says, and an `--xla_dump_to` dump keeps each variant's optimised modules (`xla-dump/`).
4. **MLX as a float64 engine** (`step.jsonl`): the stored Bellman step at m = 2,000, 8,000 and 14,000, and a dense `P @ V` at n = 1,000 and 4,000, in MLX on its CPU device under `mx.compile`, against JAX. Each must agree with NumPy's answer within 1e-12.

## Method

- **Each variant runs in a fresh process.**
- **Compile is timed apart:** a cell's first call is timed on its own, compile included.
- **A cell's time** is the median of its warm calls: at least three, and up to 20 within ten seconds, or five for the cliff.
- **Thread counts** are left at their defaults.

## Running it

From the root of a checkout, with the benchmarks venv (built from `precision/constraints.txt`) and a separate MLX venv built from `precision/probes/mac_float64/requirements-mlx.txt`:

```
PROBE_HOST=<host> MLXPY=<the MLX venv>/bin/python bash precision/probes/mac_float64/run.sh precision/runs/<date>/<host>/mac-float64
```

- **`--quick`** runs small sizes, as a check that everything runs, not a measurement.
- **A step that fails** is logged in `run.log` and the run goes on to the others; it ends with a nonzero exit, and `run.log`'s last line names the failed steps.
- **`summary.md`** tabulates a run.
- **These are CPU timings,** so the machine should be quiet.
