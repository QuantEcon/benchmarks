"""hopenhayn's equilibrium price p*, with the Monte Carlo draws made in one precision and the model solved in another.

    python hopenhayn_check.py gen <64|32> <out.npz> <hopenhayn.md>     the lecture's A and E draws
    python hopenhayn_check.py solve <64|32> <draws.npz> <hopenhayn.md>  p* as the lecture computes it

The lecture's own cells, from lecture-jax at 4b939ec, are executed unchanged; only the draws are swapped in.
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
                          if not l.startswith(":") and "quantecon" not in l and "matplotlib" not in l
                          and "jax_enable_x64" not in l)
ns = {}
for i in [2] + list(range(4, 21)):           # imports, the model, and every function up to compute_p_star
    exec(src(i), ns)

model = ns["create_model"]()
parameters, grids = model
if mode == "gen":
    np.savez(path, A=np.asarray(grids.A_draws), E=np.asarray(grids.E_draws))
    print(f"draws {grids.E_draws.dtype}: mean E {float(grids.E_draws.mean()):.5f}, mean A {float(grids.A_draws.mean()):.5f}")
    sys.exit()

d = np.load(path)
dtype = jnp.zeros(0).dtype
grids = grids._replace(A_draws=jnp.asarray(d["A"].astype(dtype)), E_draws=jnp.asarray(d["E"].astype(dtype)))
p_star, v_bar = ns["compute_p_star"](parameters, grids)

# The lecture's VFI at p*, counting its iterations
T, tol, max_iter = ns["T"], 1e-6, 10_000
def cond(s): return jnp.logical_and(s[0] < max_iter, s[2] > tol)
def body(s):
    i, v, _ = s
    new_v = T(v, p_star, parameters, grids)
    return i + 1, new_v, jnp.max(jnp.abs(v - new_v))
i, v, err = jax.lax.while_loop(cond, body, (0, jnp.zeros_like(grids.φ_grid), tol + 1))
print(f"  solved in {v.dtype}: p* = {p_star:.6f}; VFI at p* from zero: {int(i)} iterations, last error {float(err):.2e}, max |v| {float(jnp.max(jnp.abs(v))):.1f}")
