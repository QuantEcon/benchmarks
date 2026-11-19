"""
Check precision/ssy.py against the authors' code, whose answers ssy_reference.csv holds
(made by ssy_reference.py). Run from anywhere, in the benchmark's JAX:

    python ssy_check.py
"""
import os
import sys
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"  # float64 for the checks; float32 arrays stay float32

import numpy as np
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import ssy

reference = {}
for line in (HERE / "ssy_reference.csv").read_text().splitlines()[1:]:
    shape, quantity, _, value = line.split(",")
    reference.setdefault((shape, quantity), []).append(float(value))
SHAPES = [(2, 3, 4, 5), (4, 2, 5, 3)]  # no two axes alike, so a mixed-up axis cannot pass


def operator(shapes, dtype=jnp.float64):
    arrays = tuple(jnp.asarray(a, dtype) for a in ssy.discretize(shapes))
    return lambda w: ssy.koopmans(w, arrays)


def newton(T, w, tol=1e-7, atol=1e-4):
    """The authors' newton_solver: x - J^-1 g(x) by bicgstab on products J v, until steps are under tol."""
    g = lambda x: T(x) - x

    @jax.jit
    def step(x):
        return x - jax.scipy.sparse.linalg.bicgstab(lambda v: jax.jvp(g, (x,), (v,))[1], g(x), atol=atol)[0]

    for k in range(1, 100):
        w, last = step(w), w
        if jnp.max(jnp.abs(w - last)) <= tol:
            return w, k


print("1. T w, against the authors' T_ssy on the same w")
for shapes in SHAPES:
    name = "x".join(map(str, shapes))
    w = 600 + 300 * np.sin(np.arange(np.prod(shapes))).reshape(shapes)
    got, want = np.asarray(operator(shapes)(jnp.asarray(w))).ravel(), np.array(reference[name, "T"])
    print(f"   {name}: largest relative difference {np.max(np.abs(got - want) / want):.1e}")

print("2. The solution of w = T w from w = 800, by Newton's method with autodiff, against the authors' two solvers")
for shapes in SHAPES[:1]:
    name = "x".join(map(str, shapes))
    T = operator(shapes)
    w, steps = newton(T, jnp.full(shapes, 800.0))
    w = np.asarray(w).ravel()
    print(f"   {name}: {steps} Newton steps; w from {w.min():.2f} to {w.max():.2f}; "
          f"largest residual |T w - w| {np.max(np.abs(np.asarray(T(w.reshape(shapes))).ravel() - w)):.1e}")
    for solver in ("newton", "successive_approx"):
        gap = np.abs(w - np.array(reference[name, solver]))
        print(f"      against their {solver}: largest difference {gap.max():.1e} ({np.max(gap / w):.0e} relative)")

print("3. The Jacobian-vector product (ssy_jvp), against the whole Jacobian and a central difference")
shapes = SHAPES[0]
T = operator(shapes)
w = jnp.asarray(1 + 0.25 * np.sin(np.arange(120)).reshape(shapes))
v = jnp.asarray(np.cos(np.arange(120)).reshape(shapes))
fast = jax.jvp(T, (w,), (v,))[1]
dense = jax.jacfwd(lambda x: T(x.reshape(shapes)).ravel())(w.ravel()) @ v.ravel()
eps = 1e-6
central = (T(w + eps * v) - T(w - eps * v)) / (2 * eps)
print(f"   {shapes}: relative difference {np.max(np.abs(fast.ravel() - dense) / np.abs(dense).max()):.1e} from jacfwd, "
      f"{np.max(np.abs(fast - central)) / np.abs(central).max():.1e} from the difference")

print("4. float32")
n = 6
arrays32 = tuple(jnp.asarray(a, jnp.float32) for a in ssy.discretize((n,) * 4))
T32 = lambda w: ssy.koopmans(w, arrays32)
print(f"   theta = {ssy.THETA:.2f}; 800^theta = {float(jnp.float32(800) ** ssy.THETA):.1e} in float32, "
      f"whose smallest normal number is {np.finfo(np.float32).tiny:.1e}")
at_800 = np.asarray(T32(jnp.full((n,) * 4, 800.0, jnp.float32)))
print(f"   T w at w = 800, where the solution lies (584 to 924): {np.isinf(at_800).sum()} of {at_800.size} values are inf")
w64 = ssy._inputs(n, jnp.float64)[0]
fn32, in32 = ssy.ssy_operator(n, jnp.float32, None)
a, b = np.asarray(operator((n,) * 4)(w64)), np.asarray(fn32(*in32))
print(f"   T w at the benchmark's w: {np.isfinite(b).sum()} of {b.size} finite, and float32 is "
      f"{np.max(np.abs(a - b) / np.abs(a)):.1e} from float64 (relative)")
