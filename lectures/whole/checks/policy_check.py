"""opt_invest and opt_savings_2: the policies HPI, VFI and OPI return, in one precision.

    python policy_check.py <64|32> <lecture.md> <last function cell> <model cell> <out.npz>
    python policy_check.py compare <float64.npz> <float32.npz>

The lecture's own cells, from lecture-jax at 4b939ec, are executed unchanged.
"""
import contextlib
import io
import re
import sys

import numpy as np

if sys.argv[1] == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    print(f"  float64 HPI: {int(a['hpi_loops'])} loops; float32 HPI: {int(b['hpi_loops'])} loops")
    ref = a["hpi"]
    for name in ("hpi", "vfi", "opi"):
        for label, other in (("float64", a[name]), ("float32", b[name])):
            d = np.abs(other.astype(int) - ref.astype(int))
            print(f"  {label} {name}: differs from float64 HPI at {int((d > 0).sum())} of {d.size} states, by at most {int(d.max())} grid steps")
    v64, v32 = a["v_hpi"], b["v_hpi"]
    print(f"  value of each precision's HPI policy: max |v32 - v64| {np.abs(v32 - v64).max():.2e}, relative {np.abs(v32 - v64).max() / np.abs(v64).max():.2e}")
    sys.exit()

bits, md, last_fn, model_cell, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", bits == "64")

cells = re.findall(r"^```\{code-cell\}[^\n]*\n(.*?)^```", open(md).read(), re.S | re.M)
src = lambda i: "\n".join(l for l in cells[i - 1].splitlines()
                          if not l.startswith((":", "!")) and "matplotlib" not in l and "jax_enable_x64" not in l)
ns = {}
for i in [2] + list(range(5, last_fn + 1)) + [model_cell]:
    exec(src(i), ns)
model = ns["model"]
params, sizes, arrays = model

log = io.StringIO()
with contextlib.redirect_stdout(log):
    σ_hpi = ns["howard_policy_iteration"](model)
loops = log.getvalue().count("Concluded loop")
σ_vfi = ns["value_function_iteration"](model, tol=1e-5)
σ_opi = ns["optimistic_policy_iteration"](model, m=10, tol=1e-5)
v_hpi = ns["get_value"](σ_hpi, params, sizes, arrays)
np.savez(out, hpi=np.asarray(σ_hpi), vfi=np.asarray(σ_vfi), opi=np.asarray(σ_opi),
         v_hpi=np.asarray(v_hpi, dtype=np.float64), hpi_loops=loops)
print(f"  float{bits}: HPI {loops} loops; v dtype {v_hpi.dtype}")
