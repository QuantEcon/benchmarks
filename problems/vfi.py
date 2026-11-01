"""
P1: value function iteration on a dense grid.

The optimal savings problem of lecture-jax's `opt_savings_1`, with its
parameters. Wealth w sits on a grid, income y follows a Tauchen chain, and next
period's wealth is chosen from the same grid:

    v(w, y) = max_{w'} u(R w + y - w') + β Σ_{y'} v(w', y') Q(y, y')

Each stack iterates from v = 0 until the largest change falls below the
lecture's tolerance. The rewards u(R w + y - w') don't change between
iterations, so each stack computes them once, inside the timed solve, and
forms the expectation with one matrix product per iteration. The lecture
recomputes both every time; storing them is what a careful author would do
in each of the three stacks, and keeps the stacks on one algorithm.

The sweep is the wealth grid; income keeps the lecture's 100 states.
"""
import numpy as np

from common import fixed_positions, tauchen

R, BETA, GAMMA = 1.01, 0.98, 2.0
W_MIN, W_MAX = 0.01, 5.0
RHO, NU, Y_SIZE = 0.9, 0.1, 100
TOL, MAX_ITER = 1e-5, 10_000


def model(sweep, size, dtype):
    x, Q = tauchen(Y_SIZE, RHO, NU)
    return dict(w=np.linspace(W_MIN, W_MAX, size).astype(dtype),
                y=np.exp(x).astype(dtype), Q=Q.astype(dtype))


def _rewards(xp, w, y):
    """u(R w + y - w') for every (w, y, w'), with infeasible choices at -inf."""
    c = R * w[:, None, None] + y[None, :, None] - w[None, None, :]
    safe = xp.where(c > 0, c, 1)
    return xp.where(c > 0, safe**(1 - GAMMA) / (1 - GAMMA), -xp.inf)


def numpy(m):
    def solve(w, y, Q):
        u = _rewards(np, w, y)
        v = np.zeros((w.size, y.size), w.dtype)
        i, error = 0, TOL + 1
        while error > TOL and i < MAX_ITER:
            EV = v @ Q.T                                   # EV[w', y]
            v_new = np.max(u + BETA * EV.T[None, :, :], axis=2)
            error = np.max(np.abs(v_new - v))
            v, i = v_new, i + 1
        return v, i

    return solve, (m["w"], m["y"], m["Q"])


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["w"].dtype.type
    r, beta, g = t(R), t(BETA), t(1 - GAMMA)

    @numba.njit(parallel=parallel)
    def rewards(w, y):
        u = np.empty((w.size, y.size, w.size), w.dtype)
        for i in numba.prange(w.size):
            for j in range(y.size):
                for ip in range(w.size):
                    c = r * w[i] + y[j] - w[ip]
                    u[i, j, ip] = c**g / g if c > 0 else -np.inf
        return u

    @numba.njit(parallel=parallel)
    def bellman(u, EV, v_new):
        for i in numba.prange(u.shape[0]):
            for j in range(u.shape[1]):
                best = -np.inf
                for ip in range(u.shape[2]):
                    best = max(best, u[i, j, ip] + beta * EV[ip, j])
                v_new[i, j] = best

    @numba.njit
    def solve(w, y, Q):
        u = rewards(w, y)
        Qt = np.ascontiguousarray(Q.T)
        v = np.zeros((w.size, y.size), w.dtype)
        v_new = np.empty_like(v)
        i, error = 0, TOL + 1
        while error > TOL and i < MAX_ITER:
            bellman(u, v @ Qt, v_new)
            error = np.max(np.abs(v_new - v))
            v, v_new = v_new, v
            i += 1
        return v, i

    return solve, (m["w"], m["y"], m["Q"])


def jax(m, devices):
    import jax
    import jax.numpy as jnp

    def solve(w, y, Q):
        u = _rewards(jnp, w, y)

        def cond(state):
            i, v, error = state
            return (error > TOL) & (i < MAX_ITER)

        def body(state):
            i, v, _ = state
            EV = v @ Q.T
            v_new = jnp.max(u + BETA * EV.T[None, :, :], axis=2)
            return i + 1, v_new, jnp.max(jnp.abs(v_new - v))

        init = (0, jnp.zeros((w.size, y.size), w.dtype), jnp.asarray(TOL + 1, w.dtype))
        i, v, _ = jax.lax.while_loop(cond, body, init)
        return v, i

    return jax.jit(solve), tuple(jax.device_put((m["w"], m["y"], m["Q"]), devices[0]))


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
sample = fixed_positions
