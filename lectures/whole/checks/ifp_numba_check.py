"""Why ifp_egm_jax's Numba solve is slower in float32: time one application of K_egm_nb on each dtype.

    python ifp_numba_check.py <ifp_egm_jax.md>

K_egm_nb is the lecture's, from lecture-jax at 4b939ec. A variant replaces np.interp with a hand-written
linear interpolation, to test whether Numba's np.interp is where float32 loses time.
"""
import re
import sys
from time import perf_counter

import numba
import numpy as np
import quantecon as qe

cells = re.findall(r"^```\{code-cell\}[^\n]*\n(.*?)^```", open(sys.argv[1]).read(), re.S | re.M)
src = next(c for c in cells if "def K_egm_nb" in c)
ns = {"numba": numba, "np": np}
exec(src, ns)
K = ns["K_egm_nb"]


@numba.njit
def interp(x, xp, fp):
    """Linear interpolation, clamped at the ends, as np.interp."""
    n = xp.shape[0]
    if x <= xp[0]:
        return fp[0]
    if x >= xp[n - 1]:
        return fp[n - 1]
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xp[mid] <= x:
            lo = mid
        else:
            hi = mid
    t = (x - xp[lo]) / (xp[hi] - xp[lo])
    return fp[lo] + t * (fp[hi] - fp[lo])

ns["interp"] = interp
exec(src.replace("np.interp(a, a_in[:, z], σ_in[:, z])", "interp(a, a_in[:, z], σ_in[:, z])"), ns)
K_hand = ns["K_egm_nb"]

mc = qe.tauchen(25, 0.99, 0.02)
constants, sizes = (0.99, 1.01, 1.5), (200, 25)
for dtype in (np.float64, np.float32):
    s_grid = np.linspace(0, 16, 200).astype(dtype)
    arrays = s_grid, np.exp(mc.state_values).astype(dtype), mc.P.astype(dtype)
    σ = np.reshape(np.repeat(s_grid, 25), (200, 25))
    a = σ.copy()
    for name, f in (("np.interp", K), ("hand interp", K_hand)):
        f(a, σ, constants, sizes, arrays)                     # compile
        t = perf_counter()
        for _ in range(20):
            f(a, σ, constants, sizes, arrays)
        print(f"  {np.dtype(dtype).name:8} {name:12} {(perf_counter() - t) / 20 * 1e3:6.2f} ms per application")
