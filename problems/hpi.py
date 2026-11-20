"""
P2: Howard policy iteration, with an iterative linear solve for each policy.

The optimal savings problem of lecture-jax's `opt_savings_2`, which is
`opt_savings_1`'s and P1's, with P1's parameters (#72). From the policy σ = 0,
each round finds the value of the current policy by solving

    (I - β P_σ) v = r_σ,    (P_σ v)(w, y) = Σ_{y'} v(σ(w, y), y') Q(y, y')

and takes the greedy policy against it, until the policy stops changing.
The value comes from BiCGSTAB, as in the lecture, on the operator
v ↦ v - β (v Qᵀ)[σ(w, y), y] without building the matrix, to a relative
residual of 1e-10. The lecture's 1e-5 lets each stack stop on a different
suboptimal policy. The first solve starts from zero and each later one from
the previous policy's value. The rewards are computed once, inside the timed
solve, and stored, as in P1.

After each round's solve, every stack recomputes its relative residual, and
returns the largest as `residual`. `verify` fails a cell whose residual is
above 1e-9 or which reached the lecture's cap of 250 rounds, since JAX's
BiCGSTAB reports no failure.

Two sweeps: `grid` is P1's wealth grid with 100 income states, so P1 and P2
compare directly; there the greedy step, n² y for n wealth points and y
income states, outgrows the solve, n y² an application. `income` holds the
lecture's 150 wealth points and grows the income states, so the solve grows
faster than the greedy step.
"""
import numpy as np

from common import fixed_positions, tauchen
from vfi import BETA, GAMMA, NU, RHO, R, W_MAX, W_MIN, Y_SIZE, _rewards

W_SIZE = 150            # the income sweep's wealth grid, the lecture's
TOL = 1e-10             # BiCGSTAB's relative residual
SOLVE_ITER = 1_000      # BiCGSTAB's iterations for one policy, at most
MAX_ITER = 250          # rounds, the lecture's cap
RESIDUAL = 1e-9         # the final relative residual a cell must reach


def model(sweep, size, dtype):
    n, ny = (size, Y_SIZE) if sweep == "grid" else (W_SIZE, size)
    x, Q = tauchen(ny, RHO, NU)
    return dict(w=np.linspace(W_MIN, W_MAX, n).astype(dtype),
                y=np.exp(x).astype(dtype), Q=Q.astype(dtype))


def verify(row):
    if row["iterations"] >= MAX_ITER:
        raise RuntimeError(f"the policy was still changing after {MAX_ITER} rounds")
    if not row["residual"] <= RESIDUAL:
        raise RuntimeError(f"a round's solve left a relative residual of "
                           f"{row['residual']:.2g}, above {RESIDUAL:g}")


def numpy(m):
    import scipy.sparse.linalg as spla

    def solve(w, y, Q):
        u = _rewards(np, w, y)
        n, ny = w.size, y.size
        ii, jj = np.arange(n)[:, None], np.arange(ny)[None, :]

        def L(sigma, x):
            x = x.reshape(n, ny)
            return (x - BETA * (x @ Q.T)[sigma, jj]).ravel()

        sigma = np.zeros((n, ny), np.int64)
        v = np.zeros(n * ny, w.dtype)
        i, changed, residual = 0, True, 0.0
        while changed and i < MAX_ITER:
            op = spla.LinearOperator((n * ny, n * ny), matvec=lambda x, s=sigma: L(s, x),
                                     dtype=w.dtype)
            r = u[ii, jj, sigma].ravel()
            v, info = spla.bicgstab(op, r, x0=v, rtol=TOL, atol=0, maxiter=SOLVE_ITER)
            if info != 0:
                raise RuntimeError(f"BiCGSTAB stopped with info {info} in round {i + 1}")
            residual = max(residual, np.linalg.norm(r - L(sigma, v)) / np.linalg.norm(r))
            new = np.argmax(u + BETA * (v.reshape(n, ny) @ Q.T).T[None, :, :], axis=2)
            changed = (new != sigma).any()
            sigma, i = new, i + 1
        return v.reshape(n, ny), i, {"residual": residual}

    return solve, (m["w"], m["y"], m["Q"])


def numba(m, parallel):
    import numba

    @numba.njit(parallel=parallel)
    def rewards(w, y):
        g = 1 - GAMMA
        u = np.empty((w.size, y.size, w.size), w.dtype)
        for i in numba.prange(w.size):
            for j in range(y.size):
                for ip in range(w.size):
                    c = R * w[i] + y[j] - w[ip]
                    u[i, j, ip] = c**g / g if c > 0 else -np.inf
        return u

    @numba.njit
    def apply(sigma, Qt, x, out):
        # out = x - β (x Qᵀ)[σ(w, y), y]: a matrix product, then a gather. The
        # gather stays serial in numba-parallel: it runs two or three times an
        # iteration, and starting a parallel region each time cost more than
        # the gather itself
        EV = x @ Qt
        for i in range(sigma.shape[0]):
            for j in range(sigma.shape[1]):
                out[i, j] = x[i, j] - BETA * EV[sigma[i, j], j]

    @numba.njit
    def dot(a, b):
        return np.sum(a * b)

    breakdown = np.finfo(np.float64).eps ** 2       # SciPy's, for ρ and ω

    @numba.njit
    def bicgstab(sigma, Qt, b, x):
        # SciPy's BiCGSTAB loop, updating x in place: the residual's 2-norm is
        # tested against TOL ‖b‖ before each iteration and at the half-step
        atol = TOL * np.sqrt(dot(b, b))
        t = np.empty_like(b)
        apply(sigma, Qt, x, t)
        r = b - t
        rhat = r.copy()
        p, v = np.zeros_like(b), np.zeros_like(b)
        rho_prev = alpha = omega = 1.0
        for it in range(SOLVE_ITER):
            if np.sqrt(dot(r, r)) < atol:
                return 0
            rho = dot(rhat, r)
            if abs(rho) < breakdown:
                return -10
            if it == 0:
                p = r.copy()
            else:
                if abs(omega) < breakdown:
                    return -11
                p = r + (rho / rho_prev) * (alpha / omega) * (p - omega * v)
            apply(sigma, Qt, p, v)
            rv = dot(rhat, v)
            if rv == 0.0:
                return -11
            alpha = rho / rv
            s = r - alpha * v
            if np.sqrt(dot(s, s)) < atol:
                x += alpha * p
                return 0
            apply(sigma, Qt, s, t)
            omega = dot(t, s) / dot(t, t)
            x += alpha * p + omega * s
            r = s - omega * t
            rho_prev = rho
        return SOLVE_ITER

    @numba.njit(parallel=parallel)
    def greedy(u, EV, sigma):
        # The first maximum, as argmax keeps it
        for i in numba.prange(u.shape[0]):
            for j in range(u.shape[1]):
                best, arg = -np.inf, 0
                for ip in range(u.shape[2]):
                    value = u[i, j, ip] + BETA * EV[ip, j]
                    if value > best:
                        best, arg = value, ip
                sigma[i, j] = arg

    @numba.njit
    def hpi(w, y, Q):
        u = rewards(w, y)
        Qt = np.ascontiguousarray(Q.T)
        n, ny = w.size, y.size
        sigma = np.zeros((n, ny), np.int64)
        new = np.empty_like(sigma)
        v = np.zeros((n, ny), w.dtype)
        r, t = np.empty_like(v), np.empty_like(v)
        i, changed, residual = 0, True, 0.0
        while changed and i < MAX_ITER:
            for a in range(n):
                for j in range(ny):
                    r[a, j] = u[a, j, sigma[a, j]]
            status = bicgstab(sigma, Qt, r, v)
            if status != 0:
                return v, i + 1, residual, status
            apply(sigma, Qt, v, t)
            residual = max(residual, np.sqrt(dot(r - t, r - t) / dot(r, r)))
            greedy(u, v @ Qt, new)
            changed = (new != sigma).any()
            sigma, new = new, sigma
            i += 1
        return v, i, residual, 0

    def solve(w, y, Q):
        v, i, residual, status = hpi(w, y, Q)
        if status != 0:
            # SciPy's codes: the iterations used, or -10 and -11 for a breakdown
            raise RuntimeError(f"BiCGSTAB stopped with info {status} in round {i}")
        return v, i, {"residual": residual}

    return solve, (m["w"], m["y"], m["Q"])


def jax(m, devices):
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import bicgstab

    def solve(w, y, Q):
        u = _rewards(jnp, w, y)
        n, ny = w.size, y.size
        ii, jj = jnp.arange(n)[:, None], jnp.arange(ny)[None, :]

        def L(sigma, x):
            return x - BETA * (x @ Q.T)[sigma, jj]

        def cond(state):
            i, sigma, v, changed, residual = state
            return changed & (i < MAX_ITER)

        def body(state):
            i, sigma, v, _, residual = state
            r = u[ii, jj, sigma]
            v, _ = bicgstab(lambda x: L(sigma, x), r, x0=v, tol=TOL, maxiter=SOLVE_ITER)
            residual = jnp.maximum(residual, jnp.linalg.norm(r - L(sigma, v)) / jnp.linalg.norm(r))
            new = jnp.argmax(u + BETA * (v @ Q.T).T[None, :, :], axis=2)
            return i + 1, new, v, jnp.any(new != sigma), residual

        init = (0, jnp.zeros((n, ny), int), jnp.zeros((n, ny), w.dtype), jnp.array(True),
                jnp.asarray(0.0, w.dtype))
        i, sigma, v, _, residual = jax.lax.while_loop(cond, body, init)
        return v, i, {"residual": residual}

    return jax.jit(solve), tuple(jax.device_put((m["w"], m["y"], m["Q"]), devices[0]))


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
sample = fixed_positions
