import time
t = time.time()
try:
    import jax, jax.numpy as jnp
    d = jax.devices()[0]
    print("B device:", d, "bytes_limit", d.memory_stats().get("bytes_limit"), flush=True)
    jnp.ones((1024, 1024)).block_until_ready(); print("B: 4 MiB ok", flush=True)
    jnp.ones((1024, 1024, 1024), jnp.float32).block_until_ready(); print("B: 4 GiB ok", flush=True)
except Exception as e:
    print("B failed:", type(e).__name__, str(e)[:800], flush=True)
print(f"B: {time.time() - t:.1f} s", flush=True)
