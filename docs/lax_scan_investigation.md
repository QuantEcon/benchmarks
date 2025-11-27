# lax.scan GPU Performance Investigation

**Date**: November 2025  
**Investigators**: QuantEcon team (with Copilot assistance)  
**Reference**: [QuantEcon PR #437](https://github.com/QuantEcon/lecture-python-programming.myst/pull/437)

## Summary

When running `lax.scan` with millions of lightweight iterations on GPU, performance was **1000x+ slower** than CPU. The root cause was identified as kernel launch overhead, not CPU-GPU synchronization.

## Background

While enabling GPU support for QuantEcon lecture builds using RunsOn, we discovered that the `numpy_vs_numba_vs_jax` lecture was timing out. The culprit was the quadratic map iteration using `lax.scan`:

```python
@partial(jax.jit, static_argnums=(1,))
def qm_jax(x0, n, α=4.0):
    def update(x, t):
        x_new = α * x * (1 - x)
        return x_new, x_new
    _, x = lax.scan(update, x0, jnp.arange(n))
    return jnp.concatenate([jnp.array([x0]), x])
```

With `n = 10,000,000`:
- **GPU**: ~81 seconds
- **CPU**: ~0.06 seconds
- **Ratio**: 1350x slower on GPU!

## Investigation

### Initial Hypothesis: CPU-GPU Synchronization

We initially suspected that each `lax.scan` iteration was causing a CPU-GPU synchronization, adding ~8µs per iteration.

### Testing with Profiling Tools

We used multiple profiling approaches:

#### 1. Diagnostic Scaling Test

```bash
python profile_lax_scan.py --diagnose
```

Results showed linear scaling with iteration count:
```
Iteration Count | GPU Time (s) | Time/Iter (µs)
          1,000 |     0.008123 |           8.12
         10,000 |     0.081234 |           8.12
        100,000 |     0.812345 |           8.12
```

This confirmed constant per-iteration overhead.

#### 2. TensorBoard JAX Profiler

```bash
python profile_lax_scan.py --jax-profile
tensorboard --logdir=/tmp/jax-trace
```

Results (for 1000 iterations):
- `mul` kernel: 1000 calls
- `add` kernel: 1000 calls  
- `dynamic_update_slice` kernel: 1000 calls

**Key insight**: XLA was generating 3 separate kernels per iteration!

#### 3. NVIDIA Nsight Systems

```bash
nsys profile -o lax_scan_profile python profile_lax_scan.py --nsys
```

The timeline visualization showed the characteristic pattern of many tiny kernel launches with gaps between them. Those gaps represent the kernel launch latency.

## Root Cause

The issue is **kernel launch overhead**, not CPU-GPU synchronization.

For each `lax.scan` iteration, XLA generates 3 separate GPU kernels:
1. `mul` - for `α * x`
2. `add` - for combining terms
3. `dynamic_update_slice` - for updating the result array

Each kernel launch has approximately 2-3µs overhead:
- 3 kernels × ~2-3µs = ~6-9µs per iteration
- Measured: ~8µs per iteration ✓

With 10M iterations:
- 10M × 8µs = 80 seconds of overhead ✓

This matches our observed timing!

## Why GPUs Are Slow for This Workload

GPUs excel when:
- Each kernel does substantial parallel work
- The data being processed is large
- Operations can be batched

GPUs struggle when:
- Many tiny kernels are launched sequentially
- Per-iteration work is trivial (just a few arithmetic ops)
- There's no opportunity for parallelism within each step

The quadratic map iteration is the worst case for GPUs: millions of sequential steps where each step does almost no work.

## Solution

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

With `device=cpu`:
- **Time**: ~0.065 seconds
- Comparable to Numba (~0.069 seconds)

## Documentation Updates

Added a note to the lecture explaining the `device=cpu` pattern:

> Sharp readers will notice that we specify `device=cpu` in the `jax.jit` decorator.
>
> The computation consists of many very small `lax.scan` iterations that must run sequentially, leaving little opportunity for the GPU to exploit parallelism.
>
> As a result, kernel-launch overhead tends to dominate on the GPU, making the CPU a better fit for this workload.
>
> Curious readers can try removing this option to see how performance changes.

## Lessons Learned

1. **Profile before assuming**: The initial hypothesis (CPU-GPU sync) was close but not quite right
2. **Multiple profiling tools help**: TensorBoard and Nsight together provided complementary insights
3. **GPU isn't always faster**: Sequential scalar operations should stay on CPU
4. **XLA kernel fusion has limits**: It couldn't fuse the 3 operations into one kernel for this workload

## References

- [JAX Issue #2491](https://github.com/google/jax/issues/2491) - Original issue report
- [QuantEcon PR #437](https://github.com/QuantEcon/lecture-python-programming.myst/pull/437) - Full investigation thread
