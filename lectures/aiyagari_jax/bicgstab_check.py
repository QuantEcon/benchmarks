"""
Does aiyagari_jax's get_value run bicgstab to its iteration cap in float32?

    python bicgstab_check.py aiyagari_jax.md --dtype float32

At the lecture's grid, time the lecture's bicgstab solve warm with several
caps, and report the relative residual each reaches. A solve that converges
stops early, so its time flattens as the cap rises; one that cannot reach
`tol` runs every iteration it is allowed, so its time grows with the cap.
"""
import argparse
import re
import statistics
import time

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'

parser = argparse.ArgumentParser()
parser.add_argument("lecture")
parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
args = parser.parse_args()

cells = [m.group(1) for m in CELL.finditer(open(args.lecture).read())]
last = next(i for i, c in enumerate(cells) if "def howard_policy_iteration" in c)
cells = [c for c in cells[:last + 1] if "!nvidia-smi" not in c]
if args.dtype == "float32":
    cells = [c.replace(X64_ON, X64_OFF) for c in cells]
ns = {}
for c in cells:
    exec(compile(c, "<lecture>", "exec"), ns)

import jax
import jax.numpy as jnp

household, prices = ns["create_household"](), ns["create_prices"]()
σ = ns["howard_policy_iteration"](household, prices)
r_σ = ns["compute_r_σ"](σ, household, prices)
R = lambda v: ns["R_σ"](v, σ, household)
default_cap = 10 * r_σ.size


@jax.jit
def solve(r_σ, σ, maxiter):
    return jax.scipy.sparse.linalg.bicgstab(lambda v: ns["R_σ"](v, σ, household), r_σ,
                                            maxiter=maxiter)[0]


print(f"{args.dtype} on {jax.devices()[0].platform}, grid 200 x 2, default cap {default_cap}")
for cap in (10, 30, 100, 300, 1000, default_cap):
    v = jax.block_until_ready(solve(r_σ, σ, cap))
    times = []
    for _ in range(5):
        t = time.perf_counter()
        jax.block_until_ready(solve(r_σ, σ, cap))
        times.append(time.perf_counter() - t)
    resid = float(jnp.linalg.norm(R(v) - r_σ) / jnp.linalg.norm(r_σ))
    print(f"  cap {cap:5d}: {statistics.median(times) * 1e3:8.2f} ms   relative residual {resid:.1e}"
          f"   {'(meets tol 1e-5)' if resid <= 1e-5 else '(above tol 1e-5)'}")
