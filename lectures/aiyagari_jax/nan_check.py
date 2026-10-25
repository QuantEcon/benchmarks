"""
Where does aiyagari_jax's float32 policy iteration go NaN on the GPU at a 5,000-point grid?

    python nan_check.py aiyagari_jax.md --a-size 5000

Runs the lecture's own definitions in float32 and steps its Howard policy
iteration by hand, checking every intermediate for finite values. At the first
non-finite one it reruns that step with jax_debug_nans, which names the
operation that produced the NaN, and checks the bicgstab solve's inputs.
"""
import argparse
import re
import traceback

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'

parser = argparse.ArgumentParser()
parser.add_argument("lecture")
parser.add_argument("--a-size", type=int, default=5000)
args = parser.parse_args()

cells = [m.group(1) for m in CELL.finditer(open(args.lecture).read())]
last = next(i for i, c in enumerate(cells) if "def howard_policy_iteration" in c)
cells = [c.replace(X64_ON, X64_OFF) for c in cells[:last + 1] if "!nvidia-smi" not in c]
ns = {}
for c in cells:
    exec(compile(c, "<lecture>", "exec"), ns)

import jax
import jax.numpy as jnp

household, prices = ns["create_household"](a_size=args.a_size), ns["create_prices"]()
get_greedy, get_value, compute_r_σ, R_σ = ns["get_greedy"], ns["get_value"], ns["compute_r_σ"], ns["R_σ"]
print(f"float32 on {jax.devices()[0].device_kind}, a_size {args.a_size}, jax {jax.__version__}")


def finite(x):
    return bool(jnp.all(jnp.isfinite(x)))


σ = jnp.zeros((args.a_size, 2), dtype=int)
v = get_value(σ, household, prices)
print(f"  start: v finite {finite(v)}")
for i in range(1, 30):
    σ_new = get_greedy(v, household, prices)
    r_σ = compute_r_σ(σ_new, household, prices)
    v_new = get_value(σ_new, household, prices)
    error = jnp.max(jnp.abs(v_new - v))
    changed = int(jnp.sum(σ_new != σ))
    print(f"  loop {i}: policy points changed {changed}, r_σ finite {finite(r_σ)}, "
          f"v finite {finite(v_new)}, error {float(error):.3e}, error > 1e-4: {bool(error > 1e-4)}")
    if not finite(v_new):
        resid = R_σ(v, σ_new, household) - r_σ   # the solve's residual at its starting guess
        print(f"  non-finite values: {int(jnp.sum(~jnp.isfinite(v_new)))} of {v_new.size}; "
              f"r_σ range [{float(r_σ.min()):.3g}, {float(r_σ.max()):.3g}]; "
              f"initial residual norm {float(jnp.linalg.norm(resid)):.3g}")
        for cap in (5, 10, 20, 50, 100, 200):
            x = jax.scipy.sparse.linalg.bicgstab(lambda w: R_σ(w, σ_new, household), r_σ, maxiter=cap)[0]
            print(f"    bicgstab with maxiter {cap:3d}: finite {finite(x)}")
        jax.config.update("jax_debug_nans", True)
        try:
            get_value(σ_new, household, prices)
            print("  jax_debug_nans: no error raised")
        except FloatingPointError as e:
            print("  jax_debug_nans:", str(e).splitlines()[0][:300])
            tb = [l for l in traceback.format_exc().splitlines() if "linalg" in l or "bicgstab" in l]
            print("   ", "\n    ".join(tb[-4:]))
        break
    if not bool(error > 1e-4):
        print("  the loop's test stops here")
        break
    σ, v = σ_new, v_new
