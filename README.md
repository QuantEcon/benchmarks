# QuantEcon Benchmarks

A collection of benchmarks and diagnostic scripts for profiling numerical computing performance across different hardware configurations.

## Overview

This repository contains benchmarks and diagnostic tools developed during QuantEcon's work on GPU-accelerated lecture builds. These scripts help identify performance characteristics and potential issues when running numerical code on different hardware (CPU vs GPU).

## Repository Structure

```
benchmarks/
├── jax/                    # JAX-specific benchmarks
│   ├── lax_scan/          # lax.scan performance analysis
│   └── matmul/            # Matrix multiplication benchmarks
├── hardware/              # Hardware detection and general benchmarks
├── notebooks/             # Jupyter notebook benchmarks
├── lectures/              # Whole lectures run as used, timed per cell, in either precision
└── docs/                  # Documentation and findings
```

## Categories

### JAX Benchmarks (`jax/`)

Benchmarks specific to JAX and its interaction with GPUs.

- **lax.scan**: Profiles the known issue where `lax.scan` with many lightweight iterations performs poorly on GPU due to kernel launch overhead ([JAX Issue #2491](https://github.com/google/jax/issues/2491))

### Hardware Benchmarks (`hardware/`)

General hardware detection and cross-platform benchmarks comparing:
- Pure Python performance
- NumPy (CPU)
- Numba (CPU, with parallelization)
- JAX (CPU and GPU)

### Notebook Benchmarks (`notebooks/`)

Benchmarks that test performance through different execution pathways:
- Direct Python execution
- Jupyter notebook execution (nbconvert)
- Jupyter Book execution

### Lectures, run as used (`lectures/`)

Whole QuantEcon lectures run cell by cell in one fresh process, with compile time counted, in float64 as written or switched to float32. Each lecture's folder holds the scripts for its questions, and `lectures/results/` holds the runs. See [`lectures/README.md`](lectures/README.md).

## Key Findings

### lax.scan GPU Performance Issue

When running `lax.scan` with millions of lightweight iterations on GPU, performance can be **1000x+ slower** than CPU due to kernel launch overhead:

- Each iteration launches 3 separate GPU kernels (mul, add, dynamic_update_slice)
- Each kernel launch has ~2-3µs overhead
- With 10M iterations: 3 kernels × 10M × ~3µs ≈ 90 seconds of overhead

**Solution**: Use `device=cpu` for sequential scalar operations:

```python
from functools import partial
import jax

cpu = jax.devices("cpu")[0]

@partial(jax.jit, static_argnums=(1,), device=cpu)
def sequential_operation(x0, n):
    # ... lax.scan code ...
```

## Usage

### Running lax.scan Profiler

```bash
# Basic timing comparison
python jax/lax_scan/profile_lax_scan.py

# With diagnostic output showing per-iteration overhead
python jax/lax_scan/profile_lax_scan.py --diagnose

# With NVIDIA Nsight Systems profiling
nsys profile -o lax_scan_profile python jax/lax_scan/profile_lax_scan.py --nsys

# With JAX profiler (view with TensorBoard)
python jax/lax_scan/profile_lax_scan.py --jax-profile
tensorboard --logdir=/tmp/jax-trace
```

### Running Hardware Benchmarks

```bash
python hardware/benchmark_hardware.py
```

## Requirements

- Python 3.10+
- JAX (with CUDA support for GPU benchmarks)
- NumPy
- Numba (optional, for Numba benchmarks)

For GPU profiling:
- NVIDIA Nsight Systems
- TensorBoard with profile plugin

## Contributing

When adding new benchmarks:

1. Place them in the appropriate category directory
2. Include clear documentation of what the benchmark measures
3. Add usage instructions to the script's docstring
4. Update this README with any significant findings

## References

- [JAX Issue #2491](https://github.com/google/jax/issues/2491) - lax.scan GPU performance
- [QuantEcon PR #437](https://github.com/QuantEcon/lecture-python-programming.myst/pull/437) - Original investigation

## License

BSD-3-Clause (same as QuantEcon)
