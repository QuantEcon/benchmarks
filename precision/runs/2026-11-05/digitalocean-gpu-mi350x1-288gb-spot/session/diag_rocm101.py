# diag_rocm101.py: which JAX operations fail on this card under ROCm 10.1, and whether float64
# matmul runs once hipBLASLt is turned off. Not part of the harness; run with XLA_FLAGS varied.
import os
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
print(f"jax {jax.__version__}  {jax.devices()[0].device_kind}  XLA_FLAGS={os.environ.get('XLA_FLAGS', '')!r}")
rng = np.random.default_rng(0)


def t(name, f, *args, ref=None):
    try:
        out = jax.block_until_ready(jax.jit(f)(*args))
        err = ""
        if ref is not None:
            got = np.asarray(out)
            err = f"  max rel err vs NumPy {np.max(np.abs(got - ref)) / np.max(np.abs(ref)):.1e}"
        print(f"ok    {name}{err}")
    except Exception as e:
        print(f"FAIL  {name}: {str(e).splitlines()[0][:160]}")


for dt in (np.float32, np.float64):
    d, n = np.dtype(dt).name, 256
    a = rng.standard_normal((n, n)).astype(dt)
    b = rng.standard_normal(n).astype(dt)
    spd = (a @ a.T + n * np.eye(n)).astype(dt)
    t(f"{d} matmul {n}", lambda x: x @ x, a, ref=a.astype(np.float64) @ a.astype(np.float64))
    t(f"{d} matvec {n}", lambda x, y: x @ y, a, b, ref=a.astype(np.float64) @ b.astype(np.float64))
    t(f"{d} batched matmul 64x8x8", lambda x: x @ x, a[:64 * 8 * 8 // n * 0 + 64 * 8].reshape(64, 8, 8) if False else rng.standard_normal((64, 8, 8)).astype(dt))
    t(f"{d} lu_factor", lambda x: jax.scipy.linalg.lu_factor(x), a)
    t(f"{d} solve", lambda x, y: jnp.linalg.solve(x, y), a, b, ref=np.linalg.solve(a.astype(np.float64), b.astype(np.float64)))
    t(f"{d} inv", jnp.linalg.inv, a)
    t(f"{d} cholesky", jnp.linalg.cholesky, spd)
    t(f"{d} solve_triangular", lambda x, y: jax.scipy.linalg.solve_triangular(x, y, lower=True), np.tril(spd), b)
