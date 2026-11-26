"""
P9: the wealth-consumption ratio of a long-run risk model, by Newton's method
with autodiff and by successive approximation.

The Schorfheide–Song–Yaron model of jstac/sdfs_via_autodiff (code/ssy at
bfbdab2), with the paper's baseline parameters. The ratio w, on a grid of
n points on each of four states (h_λ, h_c, h_z, z), solves w = T w, where

    T w = 1 + β (H w^θ)^(1/θ)

and H is a positive linear operator. The authors solve it by
Newton–Kantorovich iteration: x ← x − J⁻¹ g(x) for g(x) = T x − x, each step's
linear solve by BiCGSTAB on products J v, which forward-mode autodiff supplies,
from w = 800 until no step is over 1e-7: 5 steps at n = 4. Successive
approximation, w ← T w from the same start and to the same tolerance, takes
9,636 applications of T there.

The authors write H with every factor broadcast to the
(l, c, h, z, l′, c′, h′, z′) index space and summed over the primed axes, which
is n⁸ products, and which NumPy would do on the N × N array, N = n⁴: 3.4 GB at
n = 12 and 28 GB at n = 20. But their factors are separable, so here H is summed
one axis at a time, h′ then z′ then c′ then l′, which is 4 n⁵ products and
gives the same answer: `checks/ssy_check.py` compares it with their code.

JAX takes `jax.jvp` of T, as the authors do. NumPy and Numba use the product
written by hand,

    J v = β/θ (H w^θ)^(1/θ − 1) H(θ w^(θ−1) v) − v,

and Numba's BiCGSTAB is written to SciPy's loop, as in P2. All three stop a
solve at a relative residual of 1e-5 or an absolute one of 1e-4, as the authors'
call does, and start it from zero.

float64 only: the solution lies near 800 (584 to 924 on the authors' grids),
and θ = −16.02, so w^θ is about 3e-47, below float32's smallest number, and T
returns inf at every state ([`checks/ssy.txt`](checks/ssy.txt) shows it).

Two sweeps, both on n points on each axis: `newton`, where a cell is the
Newton solve, and `successive`, where it is successive approximation, which
stops at smaller n. A cell records the iterations (Newton steps or
applications of T) and the largest |T w − w| of its answer as `residual`;
`verify` fails one that reached the cap or is above RESIDUAL.
"""
import numpy as np

from common import fixed_positions

# Baseline values, Table VII of the paper, as in ssy_model.SSY
BETA, GAMMA, PSI = 0.999, 8.89, 1.97
RHO, RHO_Z, RHO_C, RHO_L = 0.987, 0.992, 0.991, 0.959
S_Z, S_C, S_L = np.sqrt(0.0039), np.sqrt(0.0096), 0.0004
MU_C = 0.0016
PHI_Z = 0.215 * 0.0035 * np.sqrt(1 - RHO**2)
PHI_C = 1.00 * 0.0035
THETA = (1 - GAMMA) / (1 - 1 / PSI)

START = 800.0           # w's start, the authors'
TOL = 1e-7              # the largest step, the authors'
KRYLOV_RTOL = 1e-5      # BiCGSTAB's, JAX's default and the authors'
KRYLOV_ATOL = 1e-4      # and the authors' absolute one
NEWTON_ITER = 100       # the authors' cap on steps
SUCCESSIVE_ITER = 100_000
RESIDUAL = 1e-3         # the largest |T w − w| a cell's answer may have


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
    The model's arrays on a grid of shape (n_l, n_c, n_h, n_z), in float64, as
    the factors of T's operator: `a1` on l′, `a2` on c, `a3` on (h, z), and
    the transition matrices Q_l (l, l′), Q_c (c, c′), Q_h (h, h′) and Q_z
    (h, z, z′), where z's own grid and chain depend on the volatility h.
    Returns (a1 Q_l, a2, a3, Q_c, Q_h, Q_z): a1 is folded into Q_l.
    """
    n_l, n_c, n_h, n_z = shapes
    h_l, Q_l = rouwenhorst(n_l, RHO_L, S_L)
    h_c, Q_c = rouwenhorst(n_c, RHO_C, S_C)
    h_z, Q_h = rouwenhorst(n_h, RHO_Z, S_Z)
    z, Q_z = map(np.array, zip(*(rouwenhorst(n_z, RHO, PHI_Z * np.exp(h)) for h in h_z)))
    sigma_c = PHI_C * np.exp(h_c)
    a1 = np.exp(THETA * h_l)
    a2 = np.exp(0.5 * ((1 - GAMMA) * sigma_c) ** 2)
    a3 = np.exp((1 - GAMMA) * (MU_C + z))
    return Q_l * a1[None, :], a2[:, None, None], a3[None, None], Q_c, Q_h, Q_z


def model(sweep, size, dtype):
    if sweep not in ("newton", "successive"):
        raise ValueError(f"P9 has no sweep {sweep!r}")
    arrays = discretize((size,) * 4)
    return dict(arrays=tuple(a.astype(dtype) for a in arrays),
                w0=np.full((size,) * 4, START, dtype), sweep=sweep)


def verify(row):
    # Successive approximation takes thousands of applications, so a count that
    # equals either cap is a loop that ran out
    if row["iterations"] in (NEWTON_ITER, SUCCESSIVE_ITER):
        raise RuntimeError(f"no step under {TOL:g} after {row['iterations']} iterations")
    if not row["residual"] <= RESIDUAL:
        raise RuntimeError(f"the answer has |T w − w| of {row['residual']:.2g}, "
                           f"above {RESIDUAL:g}, after {row['iterations']} iterations")


def H(arrays, x):
    """The operator on x of shape (n_l, n_c, n_h, n_z), in NumPy, one axis at a time."""
    Ql, a2, a3, Qc, Qh, Qz = arrays
    n_l, n_c, n_h, n_z = x.shape
    x = np.moveaxis(np.tensordot(Qh, x, (1, 2)), 0, 2)          # Σ_h′ Q_h x
    x = x.transpose(2, 0, 1, 3).reshape(n_h, n_l * n_c, n_z)    # per h: Σ_z′ Q_z x
    x = (x @ Qz.transpose(0, 2, 1)).reshape(n_h, n_l, n_c, n_z).transpose(1, 2, 0, 3)
    x = np.moveaxis(np.tensordot(Qc, x, (1, 1)), 0, 1)          # Σ_c′ Q_c x
    x = np.tensordot(Ql, x, (1, 0))                             # Σ_l′ a1 Q_l x
    return a2 * a3 * x


def T(arrays, w):
    return 1 + BETA * H(arrays, w**THETA) ** (1 / THETA)


def numpy(m):
    import scipy.sparse.linalg as spla

    def solve(arrays, w0):
        if m["sweep"] == "successive":
            w, last = w0, np.inf
            for i in range(1, SUCCESSIVE_ITER + 1):
                w, last = T(arrays, w), w
                if np.max(np.abs(w - last)) <= TOL:
                    break
            return w, i, {"residual": np.max(np.abs(T(arrays, w) - w))}

        shape = w0.shape
        w = w0
        for i in range(1, NEWTON_ITER + 1):
            Hw = H(arrays, w**THETA)
            g = (1 + BETA * Hw ** (1 / THETA) - w).ravel()
            # J v - v, with J v = β/θ Hw^(1/θ-1) H(θ w^(θ-1) v)
            s, r = BETA / THETA * Hw ** (1 / THETA - 1), THETA * w ** (THETA - 1)
            op = spla.LinearOperator(
                (w.size,) * 2, dtype=w.dtype,
                matvec=lambda v: (s * H(arrays, r * v.reshape(shape))).ravel() - v)
            d, info = spla.bicgstab(op, g, rtol=KRYLOV_RTOL, atol=KRYLOV_ATOL)
            if info != 0:
                raise RuntimeError(f"BiCGSTAB stopped with info {info} in step {i}")
            w, last = w - d.reshape(shape), w
            if np.max(np.abs(w - last)) <= TOL:
                break
        return w, i, {"residual": np.max(np.abs(T(arrays, w) - w))}

    return solve, (m["arrays"], m["w0"])


def numba(m, parallel):
    import numba

    @numba.njit(parallel=parallel)
    def H(Ql, a2, a3, Qc, Qh, Qz, x, out, c, d):
        # The sums over h′, z′, c′ and l′ in turn, with the long axis inside.
        # c and d are scratch of x's shape. Matrix products through BLAS ran no
        # faster here, and contended with the prange threads
        n_l, n_c, n_h, n_z = x.shape
        for l in numba.prange(n_l):
            for ci in range(n_c):
                for h in range(n_h):
                    for z in range(n_z):
                        c[l, ci, h, z] = 0.0
                    for k in range(n_h):
                        q = Qh[h, k]
                        for z in range(n_z):
                            c[l, ci, h, z] += q * x[l, ci, k, z]
                    for z in range(n_z):
                        acc = 0.0
                        for k in range(n_z):
                            acc += Qz[h, z, k] * c[l, ci, h, k]
                        d[l, ci, h, z] = acc
        for l in numba.prange(n_l):
            for ci in range(n_c):
                for h in range(n_h):
                    for z in range(n_z):
                        c[l, ci, h, z] = 0.0
                for k in range(n_c):
                    q = Qc[ci, k]
                    for h in range(n_h):
                        for z in range(n_z):
                            c[l, ci, h, z] += q * d[l, k, h, z]
        for l in numba.prange(n_l):
            for ci in range(n_c):
                for h in range(n_h):
                    for z in range(n_z):
                        out[l, ci, h, z] = 0.0
            for k in range(n_l):
                q = Ql[l, k]
                for ci in range(n_c):
                    for h in range(n_h):
                        for z in range(n_z):
                            out[l, ci, h, z] += q * c[k, ci, h, z]
        for l in numba.prange(n_l):
            for ci in range(n_c):
                for h in range(n_h):
                    for z in range(n_z):
                        out[l, ci, h, z] *= a2[ci, 0, 0] * a3[0, 0, h, z]

    @numba.njit
    def dot(a, b):
        return np.sum(a * b)

    breakdown = np.finfo(np.float64).eps ** 2       # SciPy's, for ρ and ω

    @numba.njit
    def bicgstab(Ql, a2, a3, Qc, Qh, Qz, s, r, b, x, scratch):
        # SciPy's BiCGSTAB on v ↦ s H(r v) − v, from x = 0 and updating x in
        # place, to ‖residual‖ < max(rtol ‖b‖, atol), tested before each
        # iteration and at the half-step
        n = b.size
        shape = b.shape
        c, d, hv, e = scratch

        atol = max(KRYLOV_RTOL * np.sqrt(dot(b, b)), KRYLOV_ATOL)
        x[:] = 0.0
        res = b.copy()
        rhat = res.copy()
        p, v, t = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        rho_prev = alpha = omega = 1.0
        for it in range(10 * n):
            if np.sqrt(dot(res, res)) < atol:
                return 0
            rho = dot(rhat, res)
            if abs(rho) < breakdown:
                return -10
            if it == 0:
                p = res.copy()
            else:
                if abs(omega) < breakdown:
                    return -11
                p = res + (rho / rho_prev) * (alpha / omega) * (p - omega * v)
            e[:] = r * p
            H(Ql, a2, a3, Qc, Qh, Qz, e, v, c, d)
            v[:] = s * v - p
            rv = dot(rhat, v)
            if rv == 0.0:
                return -11
            alpha = rho / rv
            sv = res - alpha * v
            if np.sqrt(dot(sv, sv)) < atol:
                x += alpha * p
                return 0
            e[:] = r * sv
            H(Ql, a2, a3, Qc, Qh, Qz, e, t, c, d)
            t[:] = s * t - sv
            omega = dot(t, sv) / dot(t, t)
            x += alpha * p + omega * sv
            res = sv - omega * t
            rho_prev = rho
        return 10 * n

    @numba.njit
    def apply_T(Ql, a2, a3, Qc, Qh, Qz, w, out, scratch):
        c, d, hv, e = scratch
        e[:] = w**THETA
        H(Ql, a2, a3, Qc, Qh, Qz, e, hv, c, d)
        out[:] = 1 + BETA * hv ** (1 / THETA)

    @numba.njit
    def residual(Ql, a2, a3, Qc, Qh, Qz, w, scratch):
        out = np.empty_like(w)
        apply_T(Ql, a2, a3, Qc, Qh, Qz, w, out, scratch)
        return np.max(np.abs(out - w))

    @numba.njit
    def successive(Ql, a2, a3, Qc, Qh, Qz, w0, scratch):
        w, new = w0.copy(), np.empty_like(w0)
        for i in range(1, SUCCESSIVE_ITER + 1):
            apply_T(Ql, a2, a3, Qc, Qh, Qz, w, new, scratch)
            done = np.max(np.abs(new - w)) <= TOL
            w, new = new, w
            if done:
                break
        return w, i, residual(Ql, a2, a3, Qc, Qh, Qz, w, scratch), 0

    @numba.njit
    def newton(Ql, a2, a3, Qc, Qh, Qz, w0, scratch):
        w = w0.copy()
        c, d, hv, e = scratch
        g, s, r, step = np.empty_like(w0), np.empty_like(w0), np.empty_like(w0), np.empty_like(w0)
        for i in range(1, NEWTON_ITER + 1):
            e[:] = w**THETA
            H(Ql, a2, a3, Qc, Qh, Qz, e, hv, c, d)
            s[:] = BETA / THETA * hv ** (1 / THETA - 1)
            g[:] = 1 + BETA * hv ** (1 / THETA) - w
            r[:] = THETA * w ** (THETA - 1)
            status = bicgstab(Ql, a2, a3, Qc, Qh, Qz, s, r, g, step, scratch)
            if status != 0:
                return w, i, 0.0, status
            w -= step
            if np.max(np.abs(step)) <= TOL:
                break
        return w, i, residual(Ql, a2, a3, Qc, Qh, Qz, w, scratch), 0

    run = successive if m["sweep"] == "successive" else newton

    def solve(arrays, w0):
        scratch = tuple(np.empty_like(w0) for _ in range(4))
        w, i, res, status = run(*arrays, w0, scratch)
        if status != 0:
            raise RuntimeError(f"BiCGSTAB stopped with info {status} in step {i}")
        return w, i, {"residual": res}

    return solve, (m["arrays"], m["w0"])


def jax(m, devices):
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import bicgstab

    def H(arrays, x):
        Ql, a2, a3, Qc, Qh, Qz = arrays
        x = jnp.einsum("hk,lckz->lchz", Qh, x)
        x = jnp.einsum("hzk,lchk->lchz", Qz, x)
        x = jnp.einsum("ck,lkhz->lchz", Qc, x)
        x = jnp.einsum("lk,kchz->lchz", Ql, x)
        return a2 * a3 * x

    def T(arrays, w):
        return 1 + BETA * H(arrays, w**THETA) ** (1 / THETA)

    def solve(arrays, w0):
        if m["sweep"] == "successive":
            def cond(state):
                i, w, last = state
                return (jnp.max(jnp.abs(w - last)) > TOL) & (i < SUCCESSIVE_ITER)

            def body(state):
                i, w, _ = state
                return i + 1, T(arrays, w), w

            i, w, _ = jax.lax.while_loop(cond, body, (1, T(arrays, w0), w0))
            return w, i, {"residual": jnp.max(jnp.abs(T(arrays, w) - w))}

        g = lambda x: T(arrays, x) - x

        def cond(state):
            i, w, step = state
            return (jnp.max(jnp.abs(step)) > TOL) & (i < NEWTON_ITER)

        def body(state):
            i, w, _ = state
            # The step J⁻¹ g(w), from products J v by forward-mode autodiff
            step = bicgstab(lambda v: jax.jvp(g, (w,), (v,))[1], g(w),
                            tol=KRYLOV_RTOL, atol=KRYLOV_ATOL)[0]
            return i + 1, w - step, step

        i, w, _ = jax.lax.while_loop(cond, body, (0, w0, jnp.full_like(w0, jnp.inf)))
        return w, i, {"residual": jnp.max(jnp.abs(T(arrays, w) - w))}

    return jax.jit(solve), tuple(jax.device_put((m["arrays"], m["w0"]), devices[0]))


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
sample = fixed_positions
