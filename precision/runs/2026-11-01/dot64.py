"""Does JAX's float64 matrix product reach cuBLAS's float64 emulation, and how fast and how accurate is it?

    python dot64.py <label> [--sizes 2048,8192,16384] [--reps 5] [--lib jax|cupy [--math-mode N]]

QuantEcon/project-econ-on-ai-hardware#4, question 1. cuBLAS 13.0 Update 2 and later emulate float64
matrix products with integer tensor cores (the Ozaki scheme). A program that doesn't ask for it in
code can turn it on with CUBLAS_EMULATE_DOUBLE_PRECISION=1, and CUBLAS_EMULATION_STRATEGY=eager uses
it even where cuBLAS judges it slower. cuBLAS reads these when it starts, so each setting needs a
fresh process: the caller sets them, and <label> names the setting.

Three kinds of input, since the emulation picks its mantissa bits from the inputs' exponent range:
uniform on [0, 1), as a Markov operator's entries; standard normal, with cancellation in the sums;
and lognormal with sigma 5 (lognormalS sets sigma S), whose wide range of exponents is what
exp(R / sigma) has in the taste-shock Bellman step. The error is measured on an 8 x 256 corner
against an 80-bit long double product, as a multiple of float64's unit roundoff scaled by |A| |B|,
so native float64's own error is visible beside it.
Prints one JSON line per (input, size).
"""
import argparse
import json
import os
import re
import statistics
import time
from importlib import metadata

import numpy as np

ROWS, COLS = 8, 256
U = 2.0 ** -53


def make(kind, n, rng):
    if kind == "uniform":
        return rng.random((n, n)), rng.random((n, n))
    if kind == "normal":
        return rng.standard_normal((n, n)), rng.standard_normal((n, n))
    sigma = float(kind[len("lognormal"):] or 5)  # lognormal2 has sigma 2
    return np.exp(sigma * rng.standard_normal((n, n))), np.exp(sigma * rng.standard_normal((n, n)))


def error(a, b, corner):
    """Largest |c - ab| / (|a| |b|) over the ROWS x COLS corner, in units of float64's roundoff."""
    a8, b8 = a[:ROWS].astype(np.longdouble), b[:, :COLS].astype(np.longdouble)
    exact = a8 @ b8
    scale = np.abs(a8) @ np.abs(b8)
    return float(np.max(np.abs(corner.astype(np.longdouble) - exact) / scale) / U)


def backend(lib, math_mode=None):
    if lib == "jax":
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        f = jax.jit(jnp.matmul)
        put = jax.device_put
        sync = lambda x: x.block_until_ready()
        host = np.asarray

        def targets(a, b):
            # cuBLAS appears as a custom call, a Triton kernel as a fusion's backend kind
            hlo = f.lower(a, b).compile().as_text()
            found = re.findall(r'custom_call_target="([^"]+)"', hlo) + re.findall(r'"kind":"(__[^"]+)"', hlo)
            return sorted(set(found)) or ["(neither)"]
        info = {"device": jax.devices()[0].device_kind}
        return f, put, sync, host, targets, info
    import cupy as cp
    f = cp.matmul
    sync = lambda x: (cp.cuda.Device().synchronize(), x)[1]
    info = {"device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
    if math_mode is not None:  # 8 is CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH in cublas_api.h
        cp.cuda.cublas.setMathMode(cp.cuda.device.get_cublas_handle(), math_mode)
        info["math_mode"] = cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle())
    try:
        info["cublas"] = cp.cuda.cublas.getVersion(cp.cuda.device.get_cublas_handle())
    except Exception as e:
        info["cublas"] = repr(e)
    return f, cp.asarray, sync, cp.asnumpy, lambda a, b: ["cupy.matmul"], info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label")
    p.add_argument("--sizes", default="2048,8192,16384")
    p.add_argument("--kinds", default="uniform,normal,lognormal")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--lib", choices=["jax", "cupy"], default="jax")
    p.add_argument("--math-mode", type=int, help="CuPy only: cuBLAS's math mode for its handle")
    args = p.parse_args()

    f, put, sync, host, targets, info = backend(args.lib, args.math_mode)
    versions = {d.metadata["Name"]: d.version for d in metadata.distributions()
                if d.metadata["Name"].lower().startswith(("jax", "cupy", "nvidia-cublas", "nvidia-cuda-runtime"))}
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("CUBLAS", "XLA_FLAGS"))}
    rng = np.random.default_rng(0)
    for kind in args.kinds.split(","):
        for n in map(int, args.sizes.split(",")):
            a, b = make(kind, n, rng)
            da, db = put(a), put(b)
            t = time.perf_counter()
            c = sync(f(da, db))
            first = time.perf_counter() - t
            times = []
            for _ in range(args.reps):
                t = time.perf_counter()
                sync(f(da, db))
                times.append(time.perf_counter() - t)
            warm = statistics.median(times)
            row = dict(label=args.label, lib=args.lib, kind=kind, n=n, first_s=first, warm_median_s=warm,
                       tflops=2 * n ** 3 / warm / 1e12, error_u=error(a, b, host(c[:ROWS, :COLS])),
                       targets=targets(da, db), env=env, versions=versions, **info)
            print(json.dumps(row), flush=True)
            del da, db, c


if __name__ == "__main__":
    main()
