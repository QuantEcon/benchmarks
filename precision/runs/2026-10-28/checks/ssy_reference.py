"""
The SSY wealth-consumption ratio from the authors' own code, run unmodified, for
ssy_check.py to compare precision/ssy.py with. Writes ssy_reference.csv.

    python ssy_reference.py <checkout of jstac/sdfs_via_autodiff at bfbdab2>

It needs the era's JAX, since the authors' code imports `jax.config.config`, which
later releases removed: jax and jaxlib 0.4.20 (Python 3.11), jaxopt 0.8.2, numba and
quantecon. One line is added to theirs: ssy_wc_ratio.py calls `rouwenhorst` without
importing it.

For each shape it saves T applied to a fixed w, then the solution by their Newton
solver and by their successive approximation, both from w = 800 everywhere.
"""
import os
import sys

import numpy as np
import quantecon
import jax

checkout = os.path.abspath(sys.argv[1])
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ssy_reference.csv")
os.chdir(os.path.join(checkout, "code", "ssy", "discrete"))
sys.path.insert(0, ".")
import ssy_wc_ratio as m

m.rouwenhorst = quantecon.rouwenhorst

rows = []
for shapes in [(2, 3, 4, 5), (4, 2, 5, 3)]:
    ssy = m.SSY()
    arrays = [jax.device_put(a) for a in m.discretize_ssy(ssy, shapes)]
    T = lambda w: m.T_ssy(w, shapes, ssy.params, arrays)
    name = "x".join(map(str, shapes))
    w = 600 + 300 * np.sin(np.arange(np.prod(shapes))).reshape(shapes)
    found = {"T": T(w)}
    for algorithm in ("newton", "successive_approx"):
        found[algorithm] = m.solver(T, np.full(shapes, 800.0), algorithm=algorithm, verbose=False)
    for quantity, values in found.items():
        rows += [(name, quantity, i, repr(float(v))) for i, v in enumerate(np.asarray(values).ravel())]

with open(out, "w") as f:
    f.write("shape,quantity,index,value\n")
    f.writelines(",".join(map(str, row)) + "\n" for row in rows)
print(f"wrote {len(rows)} rows to {out}")
