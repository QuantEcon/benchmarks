"""
The wealth-consumption ratio of the Schorfheide-Song-Yaron long-run risk model,
from jstac/sdfs_via_autodiff (code/ssy at bfbdab2).

The ratio w solves w = T w, where T w = 1 + BETA * (H w^THETA)^(1/THETA) and H
is a positive linear operator on a grid over the four states (h_lambda, h_c,
h_z, z). The authors solve it by Newton-Kantorovich iteration, whose linear
solves need only Jacobian-vector products of T, which forward-mode autodiff
supplies. These are its two kernels: T itself, and the product.

H is summed as the authors wrote it: every factor is broadcast to the
(l, c, h, z, l', c', h', z') index space, multiplied, and summed over the
primed axes. XLA fuses that, so the N x N array, N = n^4, is never stored.

What changed from the original: Rouwenhorst's method is in NumPy, so that the
benchmark needs nothing beyond JAX; the line that turns on jax_enable_x64 is
gone, since the cell sets the precision; and the model's arrays are cast to the
cell's dtype, having been built in float64.
"""
import numpy as np
import jax
import jax.numpy as jnp

# Baseline values, Table VII of the paper, as in ssy_model.SSY
BETA, GAMMA, PSI = 0.999, 8.89, 1.97
RHO, RHO_Z, RHO_C, RHO_L = 0.987, 0.992, 0.991, 0.959
S_Z, S_C, S_L = np.sqrt(0.0039), np.sqrt(0.0096), 0.0004
MU_C = 0.0016
PHI_Z = 0.215 * 0.0035 * np.sqrt(1 - RHO**2)
PHI_C = 1.00 * 0.0035
THETA = (1 - GAMMA) / (1 - 1 / PSI)


def rouwenhorst(n, rho, sigma):
    """Values and transition matrix of an n-state approximation to an AR(1)."""
    p = (1 + rho) / 2
    P = np.array([[p, 1 - p], [1 - p, p]])
    for m in range(3, n + 1):
        Z = np.zeros((m, m))
        Z[:-1, :-1] += p * P
        Z[:-1, 1:] += (1 - p) * P
        Z[1:, :-1] += (1 - p) * P
        Z[1:, 1:] += p * P
        Z[1:-1] /= 2
        P = Z
    half_width = sigma / np.sqrt(1 - rho**2) * np.sqrt(n - 1)
    return np.linspace(-half_width, half_width, n), P


def discretize(shapes):
    """
    The model's arrays on a grid of shape (n_l, n_c, n_h, n_z), in float64:
    h_l, Q_l, Q_c, Q_h, z, Q_z, sigma_c. The volatility h_z sets z's own grid
    and transition matrix, so z[i, j] is the j-th z state when h_z is its i-th
    and Q_z[i, j, j'] the chance of going from j to j' there.
    """
    n_l, n_c, n_h, n_z = shapes
    h_l, Q_l = rouwenhorst(n_l, RHO_L, S_L)
    h_c, Q_c = rouwenhorst(n_c, RHO_C, S_C)
    h_z, Q_h = rouwenhorst(n_h, RHO_Z, S_Z)
    z, Q_z = map(np.array, zip(*(rouwenhorst(n_z, RHO, PHI_Z * np.exp(h)) for h in h_z)))
    return h_l, Q_l, Q_c, Q_h, z, Q_z, PHI_C * np.exp(h_c)


def _on(x, *axes):
    """x with its axes placed at `axes` of the 8-axis index space, and 1 elsewhere."""
    shape = [1] * 8
    for axis, size in zip(axes, x.shape):
        shape[axis] = size
    return x.reshape(shape)


def koopmans(w, arrays):
    """T w, for w of shape (n_l, n_c, n_h, n_z)."""
    h_l, Q_l, Q_c, Q_h, z, Q_z, sigma_c = arrays
    a1 = _on(jnp.exp(THETA * h_l), 4)
    a2 = _on(jnp.exp(0.5 * ((1 - GAMMA) * sigma_c) ** 2), 1)
    a3 = _on(jnp.exp((1 - GAMMA) * (MU_C + z)), 2, 3)
    H = (a1 * a2 * a3 * _on(Q_l, 0, 4) * _on(Q_c, 1, 5) * _on(Q_h, 2, 6) * _on(Q_z, 2, 3, 7))
    Hw = jnp.sum(_on(w, 4, 5, 6, 7) ** THETA * H, axis=(4, 5, 6, 7))
    return 1 + BETA * Hw ** (1 / THETA)


def _inputs(n, dtype):
    """
    A grid of n points on each axis, a w to apply T to and a direction v.

    w is near 1 rather than at the model's solution, which lies between 584 and
    924: raised to THETA = -16.02, 800 is 3e-47, below float32's smallest
    number. Built in float64 and cast, so both precisions see the same values.
    """
    shapes = (n,) * 4
    k = np.arange(n**4)
    w = (1 + 0.25 * np.sin(k)).reshape(shapes)
    v = np.cos(k).reshape(shapes)
    arrays = tuple(jnp.asarray(a, dtype) for a in discretize(shapes))
    return jnp.asarray(w, dtype), jnp.asarray(v, dtype), arrays


def ssy_operator(n, dtype, devices):
    """One application of T: N^2 products, fused and summed."""
    w, _, arrays = _inputs(n, dtype)
    return jax.jit(koopmans), (w, arrays)


def ssy_jvp(n, dtype, devices):
    """The Jacobian-vector product of T at w along v, by forward-mode autodiff."""
    w, v, arrays = _inputs(n, dtype)
    jvp = lambda w, v, arrays: jax.jvp(lambda x: koopmans(x, arrays), (w,), (v,))[1]
    return jax.jit(jvp), (w, v, arrays)
