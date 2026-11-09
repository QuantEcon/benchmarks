import os, sys, time
os.environ["JAX_ENABLE_X64"] = "1" if sys.argv[1] == "float64" else "0"
import jax, jax.numpy as jnp

n, T = 10, 10_000
M = (jnp.eye(n) * 0.5 + 0.01).astype(sys.argv[1])

@jax.jit
def run(X0):
    return jax.lax.scan(lambda X, _: (M @ X * 0.9 + M, None), X0, None, length=T)[0]

X0 = jnp.eye(n, dtype=sys.argv[1])
run(X0).block_until_ready()
t = time.perf_counter(); run(X0).block_until_ready()
print(sys.argv[1], f"{(time.perf_counter() - t) / T * 1e6:.3f} µs per product")
