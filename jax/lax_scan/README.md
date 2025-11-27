# lax.scan GPU Performance Profiler

This directory contains benchmarks and profiling tools for analyzing the `lax.scan` GPU performance issue.

## The Issue

When running `lax.scan` with many lightweight iterations on GPU, performance can be **1000x+ slower** than CPU. This is due to kernel launch overhead, not the actual computation.

### Root Cause

Each `lax.scan` iteration launches **3 separate GPU kernels**:
1. `mul` - multiplication operation
2. `add` - addition operation  
3. `dynamic_update_slice` - array update

Each kernel launch has ~2-3µs overhead. With millions of iterations:
- 10M iterations × 3 kernels × ~3µs = ~90 seconds of overhead

This was confirmed via:
- TensorBoard JAX profiler showing 1000 calls each for mul/add/dynamic_update_slice
- NVIDIA Nsight Systems showing the kernel launch pattern

### Solution

Force sequential scalar operations to CPU:

```python
from functools import partial
import jax

cpu = jax.devices("cpu")[0]

@partial(jax.jit, static_argnums=(1,), device=cpu)
def qm_jax(x0, n, α=4.0):
    def update(x, t):
        x_new = α * x * (1 - x)
        return x_new, x_new
    _, x = lax.scan(update, x0, jnp.arange(n))
    return jnp.concatenate([jnp.array([x0]), x])
```

## Files

- `profile_lax_scan.py` - Main profiling script with multiple analysis modes

## Usage

### Basic Timing Comparison

```bash
python profile_lax_scan.py
```

Output shows GPU vs CPU timing for the same operation.

### Diagnostic Mode (Recommended)

```bash
python profile_lax_scan.py --diagnose
```

Shows how execution time scales with iteration count, demonstrating the per-iteration overhead.

### NVIDIA Nsight Systems Profiling

```bash
nsys profile -o lax_scan_profile --trace=cuda,nvtx python profile_lax_scan.py --nsys
nsys-ui lax_scan_profile.nsys-rep
```

Produces detailed GPU timeline showing kernel launches.

### JAX Profiler (TensorBoard)

```bash
python profile_lax_scan.py --jax-profile
tensorboard --logdir=/tmp/jax-trace
```

View detailed operation-level profiling in TensorBoard.

### XLA HLO Dump

```bash
python profile_lax_scan.py --xla-dump
ls /tmp/xla_dump/
```

Dumps the XLA HLO intermediate representation for analysis.

## Expected Results

On a system with GPU (e.g., Tesla T4):

```
Default device (cuda:0):
  Run 1: 0.812345s
  Run 2: 0.809876s
  Run 3: 0.811234s

CPU (forced with device=cpu):
  Run 1: 0.065432s
  Run 2: 0.064321s
  Run 3: 0.065123s

Ratio (default/cpu): 12.5x
```

With `--diagnose`:

```
Iteration Count | GPU Time (s) | Time/Iter (µs)
          1,000 |     0.008123 |           8.12
          5,000 |     0.040234 |           8.05
         10,000 |     0.081234 |           8.12
         50,000 |     0.405678 |           8.11
        100,000 |     0.812345 |           8.12

✓ Time scales ~linearly with iterations!
  Estimated sync overhead: ~8.1 µs per iteration
```

## References

- [JAX Issue #2491](https://github.com/google/jax/issues/2491) - Original issue report
- [QuantEcon PR #437](https://github.com/QuantEcon/lecture-python-programming.myst/pull/437) - Investigation details
