"""
Which operation makes the precision benchmark's Kalman filter slower in float32 than in float64?

    python kalman_check.py float32

Times each kind of operation in the filter's step on its own, inside a
lax.scan of 10,000 steps on the CPU, warm, and prints the LAPACK custom calls
the solves compile to.
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

T, k, p = 10_000, 10, 3
f32 = jnp.float32
A = (0.9 * jnp.eye(k, dtype=f32) + 0.01 * jax.random.normal(jax.random.key(1), (k, k), f32)).astype(dtype)
H = jax.random.normal(jax.random.key(2), (p, k), f32).astype(dtype)
Y = jax.random.normal(jax.random.key(3), (T, p), f32).astype(dtype)
S0 = H @ H.T + 0.5 * jnp.eye(p, dtype=dtype)
I_k, I_p = jnp.eye(k, dtype=dtype), jnp.eye(p, dtype=dtype)

# Each body maps (carry, y) to a carry of the same shape, kept bounded
BODIES = {
    "matmuls (A P A', H P H')": (I_k, lambda P, y: 0.5 * (A @ P @ A.T) + 0.1 * (H.T @ (H @ P @ H.T) @ H) / 10 + I_k),
    "solve, 3x3 with 10 right-hand sides": (H, lambda X, y: jnp.linalg.solve(S0, 0.5 * X + H)),
    "solve, 3x3 with one right-hand side": (y0 := jnp.zeros(p, dtype), lambda v, y: jnp.linalg.solve(S0, 0.5 * v + y)),
    "slogdet, 3x3": (jnp.zeros((), dtype), lambda s, y: 0.5 * s + jnp.linalg.slogdet(S0 + 1e-3 * jnp.tanh(s) * I_p)[1]),
    "cho_solve, 3x3 with 10 right-hand sides": (H, lambda X, y: jax.scipy.linalg.cho_solve(
        jax.scipy.linalg.cho_factor(S0), 0.5 * X + H)),
}


def scan_of(init, body):
    @jax.jit
    def run(Y):
        return jax.lax.scan(lambda c, y: (body(c, y), None), init, Y)[0]
    return run


print(f"{dtype} on {jax.devices()[0].device_kind}, jax {jax.__version__}, T = {T}")
for name, (init, body) in BODIES.items():
    f = scan_of(init, body)
    jax.block_until_ready(f(Y))
    times = []
    for _ in range(5):
        t = time.perf_counter()
        jax.block_until_ready(f(Y))
        times.append(time.perf_counter() - t)
    calls = sorted(set(re.findall(r'custom_call_target="([^"]+)"', f.lower(Y).compile().as_text())))
    print(f"  {name:42s} {statistics.median(times) / T * 1e6:7.2f} µs/step   {', '.join(calls) or '-'}")
