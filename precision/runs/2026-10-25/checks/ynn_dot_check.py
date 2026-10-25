"""
Small matrix products on the CPU, by precision: per-step time inside lax.scan, and what XLA lowers the dot to.

    python ynn_dot_check.py float32
"""
import os
import re
import statistics
import sys
import time

dtype = sys.argv[1]
os.environ["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

T = 10_000
print(f"{dtype}, jax {jax.__version__}, XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
for n in (2, 4, 8, 10, 16, 32, 64, 128):
    M = (jnp.eye(n) * 0.5 + 0.01).astype(dtype)

    @jax.jit
    def run(X0):
        return jax.lax.scan(lambda X, _: (M @ X * 0.9 + M, None), X0, None, length=T)[0]

    X0 = jnp.eye(n, dtype=dtype)
    jax.block_until_ready(run(X0))
    times = []
    for _ in range(5):
        t = time.perf_counter()
        jax.block_until_ready(run(X0))
        times.append(time.perf_counter() - t)
    hlo = run.lower(X0).compile().as_text()
    lowered = (sorted(set(re.findall(r'custom_call_target="([^"]+)"', hlo)))
               + sorted(set(m for m in re.findall(r"kind=k(\w+)", hlo) if m != "Loop")))
    xnn = sorted(set(re.findall(r"(xnn\w*|ynn\w*|onednn\w*|__\w+\$\w+)", hlo)))
    print(f"  n={n:4d}: {statistics.median(times) / T * 1e6:7.3f} µs per product   lowering: {lowered + xnn}")
