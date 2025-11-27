# Hardware Benchmarks

General hardware detection and cross-platform performance benchmarks.

## Overview

These benchmarks provide a consistent way to measure performance across different CI runners and hardware configurations. They compare:

- Pure Python performance
- NumPy (CPU, with implicit multithreading)
- Numba (CPU, with explicit parallelization)
- JAX (CPU and GPU)

## Files

- `benchmark_hardware.py` - Comprehensive hardware benchmark script
- `test_jax_install.py` - Quick JAX installation verification

## Usage

### Full Hardware Benchmark

```bash
python benchmark_hardware.py
```

This produces:
1. System information (CPU, GPU detection)
2. Pure Python benchmarks
3. NumPy benchmarks
4. Numba benchmarks (if available)
5. JAX benchmarks (GPU if available, else CPU)
6. JSON output file with all results

### Quick JAX Test

```bash
python test_jax_install.py
```

Simple verification that JAX is installed and working.

## Output

Results are saved to `benchmark_results_bare_metal.json`:

```json
{
  "pathway": "bare_metal",
  "timestamp": "2025-11-28T10:00:00",
  "system": {
    "platform": "Linux-6.8.0-...",
    "processor": "x86_64",
    "python": "3.11.0",
    "cpu_count": 8,
    "cpu_model": "Intel Xeon ...",
    "gpu": "Tesla T4, 15360 MiB"
  },
  "benchmarks": {
    "pure_python": {...},
    "numpy": {...},
    "numba": {...},
    "jax": {...}
  }
}
```

## Interpreting Results

### Expected Performance Hierarchy

For **vectorized operations** (e.g., matrix multiply):
```
JAX (GPU) > JAX (CPU) ≈ NumPy > Numba (non-parallel) > Pure Python
```

For **sequential operations** (e.g., iterative algorithms):
```
Numba ≈ JAX (CPU) > NumPy > Pure Python
JAX (GPU) may be slower due to kernel launch overhead
```

### GPU vs CPU Guidelines

Use GPU when:
- Large matrix operations (>1000x1000)
- Embarrassingly parallel workloads
- Batch processing of many independent computations

Use CPU when:
- Sequential/iterative algorithms with small per-iteration work
- Memory-bound operations on small data
- Operations with many small kernel launches

## CI Integration

These benchmarks can be integrated into CI workflows:

```yaml
- name: Run Hardware Benchmarks
  run: |
    python hardware/benchmark_hardware.py
    
- name: Upload Benchmark Results
  uses: actions/upload-artifact@v4
  with:
    name: benchmark-results
    path: benchmark_results_bare_metal.json
```
