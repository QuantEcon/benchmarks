"""
Time one cell of the precision benchmark, in a fresh process.

The driver (run_matrix.py) sets the precision and the device through the
environment before this process starts, so JAX sees them at import, as a
lecture would. Prints one JSON object on stdout.
"""
import time

T_START = time.perf_counter()

import argparse
import json
import os
import statistics
import subprocess
import traceback

import numpy as np

SAMPLES = 64
TELEMETRY = "index,pci.bus_id,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"


def gpu_telemetry():
    """Clocks, temperature and power of the cards in use, read just after timing."""
    ids = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not ids:
        return None
    for fields in (TELEMETRY + ",clocks_event_reasons.active", TELEMETRY):
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--id={ids}", f"--query-gpu={fields}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode == 0:
            names = fields.split(",")
            return [dict(zip(names, (v.strip() for v in line.split(","))))
                    for line in out.stdout.strip().splitlines()]
    return None


def memory_analysis(compiled):
    """XLA's own account of the buffers a compiled kernel needs."""
    try:
        stats = compiled.memory_analysis()
        return {k: getattr(stats, k) for k in
                ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")}
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    row = {"status": "error"}
    try:
        import jax
        import jax.numpy as jnp
        import kernels

        row["import_s"] = time.perf_counter() - T_START

        # Starting the backend is part of first use too, and CUDA's is slow
        t = time.perf_counter()
        devices = jax.devices()
        row["backend_init_s"] = time.perf_counter() - t
        row.update(
            jax=jax.__version__,
            platform=devices[0].platform,
            device_kind=devices[0].device_kind,
            n_devices=len(devices),
            default_float=jnp.zeros(()).dtype.name,
        )
        dtype = jnp.dtype(args.dtype)

        t = time.perf_counter()
        fn, inputs = kernels.BUILDERS[args.workload](args.size, dtype, devices)
        jax.block_until_ready(inputs)
        row["setup_s"] = time.perf_counter() - t

        # Compile and the first run are what a lecture pays on first use
        t = time.perf_counter()
        compiled = fn.lower(*inputs).compile()
        row["compile_s"] = time.perf_counter() - t
        row["memory_analysis"] = memory_analysis(compiled)

        t = time.perf_counter()
        out = jax.block_until_ready(compiled(*inputs))
        row["first_run_s"] = time.perf_counter() - t

        # A silent float32 must not pass as float64
        row["result_dtype"] = out.dtype.name
        if out.dtype != dtype:
            raise RuntimeError(f"result is {out.dtype.name}, expected {dtype.name}")

        warm = []
        for _ in range(args.reps):
            t = time.perf_counter()
            jax.block_until_ready(compiled(*inputs))
            warm.append(time.perf_counter() - t)
        row["gpu_telemetry"] = gpu_telemetry()
        row.update(warm_s=warm, warm_median_s=statistics.median(warm), warm_min_s=min(warm))

        # Fixed positions in the result, for comparing float32 with float64
        idx = np.linspace(0, out.size - 1, min(out.size, SAMPLES)).round().astype(int)
        row["sample"] = np.asarray(jnp.ravel(out)[idx]).astype(float).tolist()
        row["status"] = "ok"
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:2000]
        row["traceback"] = traceback.format_exc()[-4000:]
    print(json.dumps(row))


if __name__ == "__main__":
    main()
