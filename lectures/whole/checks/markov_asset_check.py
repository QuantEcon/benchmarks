"""markov_asset_jax's stability check, cell #15: the 15-step power iteration from each precision's start vector.

    python markov_asset_check.py gen <64|32> <out.npy> <markov_asset_jax.md>    the lecture's random start vector
    python markov_asset_check.py run <64|32> <start.npy> <markov_asset_jax.md>  the estimate from that start

The lecture's own cells, from lecture-jax at 4b939ec, are executed unchanged; only the start vector is swapped in.
"""
import re
import sys

import numpy as np

mode, bits, path, md = sys.argv[1:5]
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", bits == "64")

cells = re.findall(r"^```\{code-cell\}[^\n]*\n(.*?)^```", open(md).read(), re.S | re.M)
src = lambda i: "\n".join(l for l in cells[i - 1].splitlines()
                          if not l.startswith((":", "!")) and "matplotlib" not in l and "jax_enable_x64" not in l)
ns = {}
for i in (3, 5, 7, 13, 14, 16):          # imports, power_iteration_sr, the SV model, compute_A, sv_model
    exec(src(i), ns)
A = ns["compute_A"](ns["sv_model"])
n = A.shape[1]

if mode == "gen":
    b = jax.random.normal(jax.random.key(1234), (n,))      # as power_iteration_sr draws it
    np.save(path, np.asarray(b))
    sys.exit()

b0 = jnp.asarray(np.load(path).astype(jnp.zeros(0).dtype))
A = jnp.asarray(A)
b_k, sr = b0, 0
for _ in range(15):                                       # power_iteration_sr's loop, unchanged
    b_k1 = jnp.dot(A, b_k)
    b_k1_norm = jnp.linalg.norm(b_k1)
    sr = jnp.sum(b_k1 * b_k) / jnp.sum(b_k * b_k)
    b_k = b_k1 / b_k1_norm
lecture = float(ns["power_iteration_sr"](A)) if path.endswith(f"{bits}.npy") else float("nan")
print(f"  A {A.dtype} {A.shape}: estimate {float(sr):.4f}" + (f" (the lecture's own function: {lecture:.4f})" if lecture == lecture else ""))
