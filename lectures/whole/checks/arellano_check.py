"""arellano_jax's solve, cell #15, in one precision and at one tolerance; then the gap between two solutions.

    python arellano_check.py <64|32> <arellano_jax.md> <out.npz> [tol, default the lecture's 1e-8]
    python arellano_check.py compare <reference.npz> <other.npz>

The lecture's own cells, from lecture-jax at 4b939ec, are executed unchanged.
"""
import contextlib
import io
import re
import sys

import numpy as np

if sys.argv[1] == "compare":
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in ("v_c", "v_d", "q"):
        print(f"  {k}: largest gap {np.abs(b[k] - a[k]).max():.2e}, largest |value| {np.abs(a[k]).max():.3g}")
    dq = np.abs(b["q"] - a["q"])
    print(f"  q: gap above 0.01 at {int((dq > 0.01).sum())} of {dq.size} states, above 0.1 at {int((dq > 0.1).sum())}")
    d = np.abs(b["B_star"].astype(int) - a["B_star"].astype(int))
    print(f"  B_star: differs at {int((d > 0).sum())} of {d.size} states, by at most {int(d.max())} grid steps")
    da, db = a["default"], b["default"]
    print(f"  default region (v_d > v_c): differs at {int((da != db).sum())} of {da.size} states")
    sys.exit()

bits, md, out = sys.argv[1:4]
tol = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-8
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", bits == "64")

cells = re.findall(r"^```\{code-cell\}[^\n]*\n(.*?)^```", open(md).read(), re.S | re.M)
src = lambda i: "\n".join(l for l in cells[i - 1].splitlines()
                          if not l.startswith((":", "!")) and "matplotlib" not in l and "jax_enable_x64" not in l)
ns = {}
for i in [2] + list(range(5, 16)):           # imports, the model, and every function up to solve
    exec(src(i), ns)
with contextlib.redirect_stdout(io.StringIO()) as log:
    v_c, v_d, q, B_star = ns["solve"](ns["create_arellano"](), tol=tol)
print(f"  float{bits}, tol = {tol:g}: " + log.getvalue().strip().splitlines()[-1])
np.savez(out, v_c=np.asarray(v_c, np.float64), v_d=np.asarray(v_d, np.float64), q=np.asarray(q, np.float64),
         B_star=np.asarray(B_star), default=np.asarray(v_d[None, :] > v_c))
