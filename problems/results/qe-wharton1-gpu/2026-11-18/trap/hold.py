import time, jax, jax.numpy as jnp
x = jnp.ones((1024, 1024)); x.block_until_ready()
print("A ready:", jax.devices()[0], "bytes_limit", jax.devices()[0].memory_stats().get("bytes_limit"), flush=True)
time.sleep(60)
print("A done", flush=True)
