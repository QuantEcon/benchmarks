"""
Does TF32's rounding of EGM's one matrix product explain P3's float32 answers on a TF32 card?

On NVIDIA cards from Ampere on, XLA does a float32 matrix product in TF32 by default: each operand
keeps 10 of its 23 fraction bits, and the products are summed in float32. This runs egm.py's NumPy
solve in float32 on the CPU, with the expectation's product sigma_next**(-gamma) @ P.T done on
operands rounded that way, and compares the answer with float64's, as summarise.py does: the largest
difference, relative to the float64 answer's largest value. Each case also prints the last few
iterations' changes, to show whether the iterate stops moving.

    python problems/checks/tf32_egm.py [sizes]        # 200,1000,5000,20000,50000 by default
"""
import platform
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import egm  # noqa: E402

SIZES = [int(n) for n in (sys.argv[1] if len(sys.argv) > 1 else "200,1000,5000,20000,50000").split(",")]


def tf32(x, mode):
    """x in float32, its fraction cut to TF32's 10 bits: to nearest (ties to even), or truncated."""
    b = np.ascontiguousarray(x, np.float32).view(np.uint32)
    if mode == "nearest":
        b = b + np.uint32(0x0FFF) + ((b >> 13) & np.uint32(1))
    return (b & np.uint32(0xFFFFE000)).view(np.float32)


def solve(m, dtype, rounding=None, which=None):
    """egm.py's NumPy solve, with the product's operands rounded to TF32 if asked."""
    t = np.dtype(dtype).type
    s, y, P = (m[k].astype(dtype) for k in ("s", "y", "P"))
    R, BETA, GAMMA, TOL = t(egm.R), t(egm.BETA), t(egm.GAMMA), t(egm.TOL)
    Pt = P.T.copy()
    if rounding and which in ("both", "P"):
        Pt = tf32(Pt, rounding)
    sigma = np.repeat(s[:, None], y.size, axis=1)
    a = sigma.copy()
    a_next = R * s[:, None] + y[None, :]
    sigma_next = np.empty_like(sigma)
    i, error, errors = 0, TOL + 1, []
    while error > TOL and i < egm.MAX_ITER:
        for k in range(y.size):
            sigma_next[:, k] = np.interp(a_next[:, k], a[:, k], sigma[:, k])
        lhs = sigma_next**(-GAMMA)
        if rounding and which in ("both", "sigma"):
            lhs = tf32(lhs, rounding)
        c = (BETA * R * (lhs @ Pt))**(t(-1) / GAMMA)
        c[0, :] = 0
        error = np.max(np.abs(sigma - c))
        errors.append(float(error))
        sigma, a, i = c, s[:, None] + c, i + 1
    return sigma, i, errors


CASES = [("float32", None, None),
         ("TF32 to nearest, both operands", "nearest", "both"),
         ("TF32 truncated, both operands", "truncate", "both"),
         ("TF32 to nearest, P only", "nearest", "P"),
         ("TF32 to nearest, sigma**(-gamma) only", "nearest", "sigma")]

if __name__ == "__main__":
    print(f"# {date.today()}: NumPy {np.__version__}, Python {platform.python_version()}, "
          f"{platform.platform()}")
    # Rounding P moves its rows' sums away from 1, by about as much as beta R is below 1
    P = egm.model("grid", 2, np.float64)["P"].astype(np.float32)
    for name, Q in (("float32", P), ("TF32", tf32(P, "nearest"))):
        gap = Q.astype(np.float64).sum(axis=1) - 1
        print(f"P's row sums less 1, in {name}: {gap.min():+.1e} to {gap.max():+.1e}, "
              f"mean {gap.mean():+.1e}; 1 - beta R is {1 - egm.BETA * egm.R:.1e}")
    for n in SIZES:
        m = egm.model("grid", n, np.float64)
        ref, ref_i, _ = solve(m, np.float64)
        print(f"== {n:,} points: float64 takes {ref_i:,} iterations")
        for name, rounding, which in CASES:
            sigma, i, errors = solve(m, np.float32, rounding, which)
            diff = np.max(np.abs(sigma.astype(np.float64) - ref)) / np.max(np.abs(ref))
            tail = ", ".join(f"{e:.1e}" for e in errors[-4:])
            print(f"  {name:38s} {i:>6,} iterations, relative difference {diff:.2g}; "
                  f"last changes {tail}", flush=True)
