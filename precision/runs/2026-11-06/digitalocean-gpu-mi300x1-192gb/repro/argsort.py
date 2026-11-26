# argsort.py: ROCm/jax#895's check through jnp.argsort(x, stable=True), the two-operand sort
# (keys and an iota) that NumPy-style code reaches. A stable argsort is unique, so it must match
# NumPy's exactly.
import time
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
n, repeats = 3 * 2**20, 100
print(f"jax {jax.__version__}  {jax.devices()[0].device_kind}")
keys = np.random.default_rng(0).integers(0, 3 * 2**16, n).astype(np.int32)
expected = np.argsort(keys, kind="stable")
argsort = jax.jit(lambda k: jnp.argsort(k, stable=True))
k_d = jnp.asarray(keys)
bad, t0 = 0, time.time()
for _ in range(repeats):
    bad += not np.array_equal(np.asarray(argsort(k_d)), expected)
print(f"jnp.argsort(stable=True) of {n} keys: {(time.time() - t0) / repeats * 1e3:.0f} ms per call "
      f"incl. checks; wrong in {bad} of {repeats} calls")
