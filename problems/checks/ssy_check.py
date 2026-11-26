"""
Check problems/ssy.py against the authors' code, whose answers ssy_reference.csv holds
(made by ssy_reference.py). Run from anywhere, in the benchmark's JAX:

    python ssy_check.py
"""
import os
import sys
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"  # float64 for the checks; float32 arrays stay float32

import numpy as np
import jax

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ssy

reference = {}
for line in (HERE / "ssy_reference.csv").read_text().splitlines()[1:]:
    shape, quantity, _, value = line.split(",")
    reference.setdefault((shape, quantity), []).append(float(value))
SHAPES = [(2, 3, 4, 5), (4, 2, 5, 3)]  # no two axes alike, so a mixed-up axis cannot pass


def model(shapes, sweep, dtype=np.float64):
    return dict(arrays=tuple(a.astype(dtype) for a in ssy.discretize(shapes)),
                w0=np.full(shapes, ssy.START, dtype), sweep=sweep)


print("1. T w, against the authors' T_ssy on the same w")
for shapes in SHAPES:
    name = "x".join(map(str, shapes))
    w = 600 + 300 * np.sin(np.arange(np.prod(shapes))).reshape(shapes)
    got, want = ssy.T(model(shapes, "newton")["arrays"], w).ravel(), np.array(reference[name, "T"])
    print(f"   {name}: largest relative difference {np.max(np.abs(got - want) / want):.1e}")

print("2. The solution of w = T w from w = 800, on each stack, against the authors' two solvers")
shapes = SHAPES[0]
name = "x".join(map(str, shapes))
builders = {"numpy": lambda m: ssy.BUILDERS["numpy"](m),
            "numba": lambda m: ssy.BUILDERS["numba"](m, parallel=False),
            "numba-parallel": lambda m: ssy.BUILDERS["numba"](m, parallel=True),
            "jax": lambda m: ssy.BUILDERS["jax"](m, jax.devices())}
for sweep, solver in (("newton", "newton"), ("successive", "successive_approx")):
    print(f"   {name}, by {sweep}, against their {solver}:")
    for stack, build in builders.items():
        solve, args = build(model(shapes, sweep))
        w, iterations, extra = solve(*args)
        w = np.asarray(w).ravel()
        gap = np.abs(w - np.array(reference[name, solver]))
        print(f"      {stack:15s} {int(iterations)} iterations; w from {w.min():.2f} to {w.max():.2f}; "
              f"|T w - w| {float(extra['residual']):.1e}; "
              f"largest difference {gap.max():.1e} ({np.max(gap / w):.0e} relative)")

print("3. float32")
arrays32 = model(shapes, "newton", np.float32)["arrays"]
print(f"   theta = {ssy.THETA:.2f}; 800^theta = {np.float32(800) ** np.float32(ssy.THETA):.1e} in float32, "
      f"whose smallest normal number is {np.finfo(np.float32).tiny:.1e}")
w = np.full(shapes, 800, np.float32)
with np.errstate(divide="ignore"):
    out = ssy.T(arrays32, w)
print(f"   T w at w = 800, where the solution lies (584 to 924): {np.isinf(out).sum()} of {out.size} values are inf")
