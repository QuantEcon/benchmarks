"""
Time one cell of the problem-type suite, in a fresh process.

The driver (run_suite.py) sets the precision and the device through the
environment before this process starts, so JAX sees them at import, as a
lecture would. Only the cell's own stack is imported: a NumPy cell never
loads JAX or Numba. Prints one JSON object on stdout.
"""
import time

T_START = time.perf_counter()

import argparse
import importlib
import importlib.util
import json
import statistics
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def precision_helpers():
    """precision/run_cell.py, for its GPU telemetry and XLA's memory analysis."""
    spec = importlib.util.spec_from_file_location(
        "precision_run_cell", HERE.parent / "precision" / "run_cell.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--budget", type=float, default=60,
                        help="stop the warm runs once they have taken this many seconds")
    args = parser.parse_args()

    stack, _, variant = args.backend.partition("-")
    row = {"status": "error"}
    try:
        problem = importlib.import_module(args.problem)
        dtype = np.dtype(args.dtype)
        model = problem.model(args.sweep, args.size, dtype)

        # The stack's own import, which a lecture pays once
        t = time.perf_counter()
        if stack == "jax":
            import jax
        elif stack == "numba":
            import numba
        row["import_s"] = time.perf_counter() - t

        backend_init_s = compile_s = 0.0
        if stack == "jax":
            # Starting the backend is part of first use too, and CUDA's is slow
            t = time.perf_counter()
            devices = jax.devices()
            row["backend_init_s"] = backend_init_s = time.perf_counter() - t
            row.update(jax=jax.__version__, platform=devices[0].platform,
                       device_kind=devices[0].device_kind,
                       default_float=jax.numpy.zeros(()).dtype.name)

        t = time.perf_counter()
        if stack == "jax":
            fn, inputs = problem.BUILDERS["jax"](model, devices)
            jax.block_until_ready(inputs)
        elif stack == "numba":
            fn, inputs = problem.BUILDERS["numba"](model, parallel=variant == "parallel")
        else:
            fn, inputs = problem.BUILDERS["numpy"](model)
        row["setup_s"] = time.perf_counter() - t

        if stack == "jax":
            t = time.perf_counter()
            compiled = fn.lower(*inputs).compile()
            row["compile_s"] = compile_s = time.perf_counter() - t
            row["memory_analysis"] = precision_helpers().memory_analysis(compiled)
            run = lambda: jax.block_until_ready(compiled(*inputs))
        else:
            run = lambda: fn(*inputs)

        # The first run is what a lecture pays on first use; for Numba it
        # includes compiling, which happens on the first call
        t = time.perf_counter()
        answer, iterations = run()
        row["first_run_s"] = time.perf_counter() - t
        row["cold_s"] = backend_init_s + compile_s + row["first_run_s"]
        answer = np.asarray(answer)
        row["iterations"] = int(iterations)

        # A silent float64 must not pass as float32, nor the other way
        row["result_dtype"] = answer.dtype.name
        if answer.dtype != dtype:
            raise RuntimeError(f"result is {answer.dtype.name}, expected {dtype.name}")

        warm = []
        while len(warm) < args.reps and (not warm or sum(warm) < args.budget):
            t = time.perf_counter()
            run()
            warm.append(time.perf_counter() - t)
        row.update(warm_s=warm, warm_median_s=statistics.median(warm), warm_min_s=min(warm))

        if args.backend == "jax-gpu":
            row["gpu_telemetry"] = precision_helpers().gpu_telemetry()
        if stack == "numba":
            row["numba_threads"] = numba.config.NUMBA_NUM_THREADS
            if variant == "parallel":
                row["numba_threading_layer"] = numba.threading_layer()
        row["sample"] = problem.sample(answer)
        row["status"] = "ok"
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:2000]
        row["traceback"] = traceback.format_exc()[-4000:]
    print(json.dumps(row))


if __name__ == "__main__":
    main()
