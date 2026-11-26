"""
P3: the endogenous grid method, for an income fluctuation problem.

The problem and parameters of lecture-jax's `ifp_egm_jax`: savings s on a
grid, income y on a Tauchen chain, CRRA utility, and Coleman-Reffett's
operator K applied by EGM until the consumption policy changes by less than
the lecture's tolerance. Each iteration interpolates the current policy at
next period's assets, R s + y', for every (s, y'):

    c(s, y) = (β R Σ_{y'} u'(σ(R s + y', y')) P(y, y'))^(-1/γ),   a = s + c

Next period's assets don't depend on today's income, so each stack
interpolates once per (s, y') and forms the expectation with one matrix
product. The lecture's Numba version interpolates again for every y, and its
JAX version loops in Python around a jitted K; here each stack runs the whole
solve in its own compiled or vectorised form.

The sweep is the savings grid; income keeps the lecture's 25 states.
"""
import numpy as np

from common import fixed_positions, tauchen

R, BETA, GAMMA = 1.01, 0.99, 1.5
S_MAX = 16.0
RHO, NU, Y_SIZE = 0.99, 0.02, 25
TOL, MAX_ITER = 1e-5, 100_000


def model(sweep, size, dtype):
    x, P = tauchen(Y_SIZE, RHO, NU)
    return dict(s=np.linspace(0, S_MAX, size).astype(dtype),
                y=np.exp(x).astype(dtype), P=P.astype(dtype))


def numpy(m):
    def solve(s, y, P):
        sigma = np.repeat(s[:, None], y.size, axis=1)      # consume everything
        a = sigma.copy()
        a_next = R * s[:, None] + y[None, :]                # [s, y']
        sigma_next = np.empty_like(sigma)
        i, error = 0, TOL + 1
        while error > TOL and i < MAX_ITER:
            for k in range(y.size):
                sigma_next[:, k] = np.interp(a_next[:, k], a[:, k], sigma[:, k])
            E = sigma_next**(-GAMMA) @ P.T                  # [s, y]
            c = (BETA * R * E)**(-1 / GAMMA)
            c[0, :] = 0
            error = np.max(np.abs(sigma - c))
            sigma, a, i = c, s[:, None] + c, i + 1
        return sigma, i

    return solve, (m["s"], m["y"], m["P"])


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["s"].dtype.type
    r, beta, gamma, inv = t(R), t(BETA), t(GAMMA), t(-1 / GAMMA)

    # One interpolation per next-period income state, each over the whole
    # sorted grid as NumPy's is, which is much faster than a search per
    # point; columns are made contiguous first
    @numba.njit(parallel=parallel)
    def interpolate(a_nextT, a, sigma, out):
        aT, sT = np.ascontiguousarray(a.T), np.ascontiguousarray(sigma.T)
        for k in numba.prange(a.shape[1]):
            out[:, k] = np.interp(a_nextT[k], aT[k], sT[k])

    @numba.njit
    def solve(s, y, P):
        n, ny = s.size, y.size
        sigma = np.empty((n, ny), s.dtype)
        a_nextT = np.empty((ny, n), s.dtype)                # [y', s]
        for i in range(n):
            for k in range(ny):
                sigma[i, k] = s[i]
                a_nextT[k, i] = r * s[i] + y[k]
        a = sigma.copy()
        Pt = np.ascontiguousarray(P.T)
        sigma_next = np.empty_like(sigma)
        i, error = 0, TOL + 1
        while error > TOL and i < MAX_ITER:
            interpolate(a_nextT, a, sigma, sigma_next)
            E = sigma_next**(-gamma) @ Pt
            c = (beta * r * E)**inv
            c[0, :] = 0
            error = np.max(np.abs(sigma - c))
            for j in range(n):
                for k in range(ny):
                    a[j, k] = s[j] + c[j, k]
            sigma = c
            i += 1
        return sigma, i

    return solve, (m["s"], m["y"], m["P"])


def jax(m, devices):
    import jax
    import jax.numpy as jnp

    interp = jax.vmap(jnp.interp, in_axes=(1, 1, 1), out_axes=1)

    def solve(s, y, P):
        sigma = jnp.repeat(s[:, None], y.size, axis=1)
        a_next = R * s[:, None] + y[None, :]

        def cond(state):
            i, a, sigma, error = state
            return (error > TOL) & (i < MAX_ITER)

        def body(state):
            i, a, sigma, _ = state
            E = interp(a_next, a, sigma)**(-GAMMA) @ P.T
            c = (BETA * R * E)**(-1 / GAMMA)
            c = c.at[0, :].set(0)
            return i + 1, s[:, None] + c, c, jnp.max(jnp.abs(sigma - c))

        init = (0, sigma, sigma, jnp.asarray(TOL + 1, s.dtype))
        i, _, sigma, _ = jax.lax.while_loop(cond, body, init)
        return sigma, i

    return jax.jit(solve), tuple(jax.device_put((m["s"], m["y"], m["P"]), devices[0]))


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
sample = fixed_positions
