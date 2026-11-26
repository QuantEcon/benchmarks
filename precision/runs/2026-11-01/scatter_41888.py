"""jax-ml/jax#41888's reproducer, unchanged, with the versions its H100 table records.

    python scatter_41888.py
"""
from importlib import metadata

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)


def wrong(rows, width, count, hint=False):
    """Count elements where out.at[:rows, idx].set(upd) on the device differs from NumPy."""
    out = np.zeros((rows, width))
    upd = np.arange(rows * count, dtype=np.float64).reshape(rows, count) + 1.0
    idx = np.arange(1, count + 1, dtype=np.int32)  # unique, sorted, in range

    def f(o, u, i):
        if hint:
            return o.at[:rows, i].set(u, unique_indices=True, indices_are_sorted=True)
        return o.at[:rows, i].set(u)

    got = np.asarray(jax.jit(f)(out, upd, idx))
    want = out.copy()
    want[:rows, idx] = upd
    return int(np.count_nonzero(got != want))


nvidia = sorted(f"{d.metadata['Name']} {d.version}" for d in metadata.distributions()
                if d.metadata["Name"].lower().startswith(("jax", "nvidia-cuda-runtime", "nvidia-cublas")))
print("; ".join(nvidia))
print(jax.__version__, jax.devices()[0].device_kind)
print("4 x 237501, 237500 indices:", wrong(4, 237501, 237500))
print("16 x 200001, 6641 indices: ", wrong(16, 200001, 6641))
print("same, with hints:          ", wrong(4, 237501, 237500, True), wrong(16, 200001, 6641, True))
