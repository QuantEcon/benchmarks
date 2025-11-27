# Notebook Execution Benchmarks

Benchmarks for testing performance through different Jupyter notebook execution pathways.

## Overview

When building documentation with Jupyter Book, notebooks can be executed through different pathways that may have different performance characteristics:

1. **Direct Python** - Running `.py` scripts directly
2. **nbconvert** - Executing notebooks via `jupyter nbconvert --execute`
3. **Jupyter Book** - Notebooks executed as part of `jupyter-book build`

These benchmarks help identify if there are performance differences between pathways.

## Files

- `benchmark_jupyterbook.md` - MyST Markdown notebook for Jupyter Book execution
- `benchmark_jupyter.ipynb` - Standard Jupyter notebook for nbconvert execution

## Usage

### Jupyter Book Execution

```bash
# Build as part of a Jupyter Book
jb build . --path-output ./

# Or execute standalone with jupytext
jupytext --execute benchmark_jupyterbook.md
```

### nbconvert Execution

```bash
jupyter nbconvert --to notebook --execute benchmark_jupyter.ipynb --output executed.ipynb
```

### Direct Python Comparison

Use the hardware benchmarks for direct Python comparison:
```bash
python ../hardware/benchmark_hardware.py
```

## Output

Each benchmark produces a JSON results file that can be compared across execution pathways:

```json
{
  "pathway": "jupyter_book",
  "timestamp": "2025-11-28T10:00:00",
  "system": {...},
  "benchmarks": {
    "matmul_1000x1000_warmup": 0.234,
    "matmul_1000x1000_compiled": 0.012,
    ...
  }
}
```

## Comparing Results

To compare results across pathways:

```python
import json

with open('benchmark_results_bare_metal.json') as f:
    bare_metal = json.load(f)

with open('benchmark_results_jupyterbook.json') as f:
    jupyterbook = json.load(f)

# Compare specific benchmarks
bm_key = 'matmul_3000x3000_compiled'
print(f"Bare metal: {bare_metal['benchmarks']['jax'][bm_key]:.3f}s")
print(f"JupyterBook: {jupyterbook['benchmarks'][bm_key]:.3f}s")
```

## Known Findings

In our testing, all three execution pathways showed **identical performance** for JAX operations, indicating that the slowdowns observed in lecture builds were not due to the execution pathway itself, but rather to the underlying `lax.scan` GPU issue.
