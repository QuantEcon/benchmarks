"""
P6: time iteration, with a root find at every grid point.

The optimal savings problem of lecture-python's `os_time_iter` (#78), with its
parameters: log utility, f(k) = k^α, multiplicative lognormal shocks drawn
once, and wealth x on a grid. The Coleman-Reffett operator K finds, at each
grid point, the consumption c in (1e-10, x - 1e-10) that solves the Euler
equation

    1/c = β mean_z[ f'(x - c) z / σ(f(x - c) z) ]

with σ the current policy, interpolated linearly as `np.interp` does. Time
iteration applies K from σ(x) = x until the policy changes by less than the
lecture's 1e-5. The exact policy is σ*(x) = (1 - αβ) x, so each cell records
its error against it, as `exact_error`.

Every stack finds the roots with Brent's method, as SciPy's `brentq`, which
the lecture uses, implements it: NumPy calls `brentq` in a Python loop over
the grid, as the lecture does; Numba and JAX use a port of `brentq`'s loop.
JAX applies it to every grid point at once, so each step evaluates the Euler
equation everywhere until the slowest point has converged. In float32, Brent's
tolerances are 1e-6 and 4 eps, since float64's 2e-12 is out of reach. A root
find that fails raises in NumPy and Numba; JAX can't raise, so it counts its
failures as `unconverged`, and `verify` fails a cell that has any.

Two sweeps: `grid` grows the grid with the lecture's 250 shocks, and
`shocks` grows the shocks with the lecture's 120 grid points.
"""
import numpy as np

from common import fixed_positions

ALPHA, BETA, MU, NU = 0.4, 0.96, 0.0, 0.1
GRID_MIN, GRID_MAX, GRID_SIZE, SHOCK_SIZE, SEED = 1e-4, 4.0, 120, 250, 1234
EDGE = 1e-10            # the bracket is (EDGE, x - EDGE), the lecture's
TOL = 1e-5              # time iteration's, on the largest change in the policy
MAX_ITER = 1_000        # time iteration's, the lecture's
BRENT_ITER = 100        # brentq's maxiter
# brentq's xtol and rtol in float64, its defaults; float32's
XTOL = {"float64": 2e-12, "float32": 1e-6}
RTOL = {"float64": 4 * np.finfo(np.float64).eps, "float32": 4 * np.finfo(np.float32).eps}


def model(sweep, size, dtype):
    if sweep not in ("grid", "shocks"):
        raise ValueError(f"P6 has no sweep {sweep!r}")
    n, ns = (size, SHOCK_SIZE) if sweep == "grid" else (GRID_SIZE, size)
    shocks = np.exp(MU + NU * np.random.default_rng(SEED).standard_normal(ns))
    return dict(grid=np.linspace(GRID_MIN, GRID_MAX, n).astype(dtype),
                shocks=shocks.astype(dtype))


def verify(row):
    if row["iterations"] >= MAX_ITER:
        raise RuntimeError(f"the policy was still changing after {MAX_ITER} iterations")
    if row.get("unconverged", 0) > 0:
        raise RuntimeError(f"{row['unconverged']:.0f} root finds didn't converge "
                           f"in {BRENT_ITER} iterations")


def exact_error(sigma, grid):
    """The largest error against σ*, relative to σ*'s largest value."""
    exact = (1 - ALPHA * BETA) * grid
    return abs(sigma - exact).max() / exact.max()


def numpy(m):
    from scipy.optimize import brentq

    xtol, rtol = XTOL[m["grid"].dtype.name], RTOL[m["grid"].dtype.name]

    def solve(grid, shocks):
        def euler_diff(c, sigma, x):
            k = x - c
            vals = 1 / np.interp(k**ALPHA * shocks, grid, sigma) * ALPHA * k**(ALPHA - 1) * shocks
            return 1 / c - BETA * np.mean(vals)

        sigma, new = grid.copy(), np.empty_like(grid)
        i, change = 0, TOL + 1
        # In float32, x - 1e-10 rounds to x, so at the bracket's upper end k is
        # 0 and f'(k) infinite, which still gives the Euler equation's sign
        with np.errstate(divide="ignore"):
            while change > TOL and i < MAX_ITER:
                for j, x in enumerate(grid):
                    new[j] = brentq(euler_diff, EDGE, x - EDGE, args=(sigma, x), xtol=xtol,
                                    rtol=rtol, maxiter=BRENT_ITER)
                change = np.max(np.abs(new - sigma))
                sigma, new = new, sigma
                i += 1
        return sigma, i, {"exact_error": exact_error(sigma, grid)}

    return solve, (m["grid"], m["shocks"])


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["grid"].dtype.type
    alpha, edge, zero, half, one, two, three = t(ALPHA), t(EDGE), t(0), t(0.5), t(1), t(2), t(3)
    xtol, rtol = t(XTOL[m["grid"].dtype.name]), t(RTOL[m["grid"].dtype.name])
    beta_mean = t(BETA / m["shocks"].size)      # β, and the mean's 1/n

    @numba.njit
    def interp(y, grid, sigma):
        # np.interp at one point: linear inside the grid, flat beyond it
        n = grid.size
        if y <= grid[0]:
            return sigma[0]
        if y >= grid[n - 1]:
            return sigma[n - 1]
        j = np.searchsorted(grid, y) - 1
        return sigma[j] + (sigma[j + 1] - sigma[j]) / (grid[j + 1] - grid[j]) * (y - grid[j])

    @numba.njit
    def euler_diff(c, sigma, x, grid, shocks):
        k = x - c
        if k == zero:
            # In float32, x - 1e-10 rounds to x. f'(0) is infinite, which
            # gives the sign, but would make the compensated sum NaN
            return -np.inf
        fk, dfk = k**alpha, alpha * k**(alpha - one)
        # Kahan's compensated sum: summed one by one in float32, the error grew
        # with the shocks, to 4e-4 against σ* at 100,000; NumPy's and JAX's
        # means sum pairwise
        total = comp = zero
        for z in shocks:
            term = dfk * z / interp(fk * z, grid, sigma) - comp
            new = total + term
            comp = (new - total) - term
            total = new
        return one / c - beta_mean * total

    @numba.njit
    def brent(sigma, x, grid, shocks):
        # SciPy's brentq loop: (root, 0), or (last iterate, 1) if it didn't
        # converge in BRENT_ITER iterations
        xpre, xcur = edge, x - edge
        fpre = euler_diff(xpre, sigma, x, grid, shocks)
        fcur = euler_diff(xcur, sigma, x, grid, shocks)
        if fpre == 0:
            return xpre, 0
        if fcur == 0:
            return xcur, 0
        xblk = fblk = spre = scur = zero
        for _ in range(BRENT_ITER):
            if fpre != 0 and fcur != 0 and (fpre < 0) != (fcur < 0):
                xblk, fblk = xpre, fpre
                spre = scur = xcur - xpre
            if abs(fblk) < abs(fcur):
                xpre, xcur, xblk = xcur, xblk, xcur
                fpre, fcur, fblk = fcur, fblk, fcur
            delta = (xtol + rtol * abs(xcur)) * half
            sbis = (xblk - xcur) * half
            if fcur == 0 or abs(sbis) < delta:
                return xcur, 0
            if abs(spre) > delta and abs(fcur) < abs(fpre):
                if xpre == xblk:
                    stry = -fcur * (xcur - xpre) / (fcur - fpre)            # interpolate
                else:
                    dpre = (fpre - fcur) / (xpre - xcur)                    # extrapolate
                    dblk = (fblk - fcur) / (xblk - xcur)
                    stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
                if two * abs(stry) < min(abs(spre), three * abs(sbis) - delta):
                    spre, scur = scur, stry                                 # a good short step
                else:
                    spre = scur = sbis                                      # bisect
            else:
                spre = scur = sbis
            xpre, fpre = xcur, fcur
            xcur += scur if abs(scur) > delta else (delta if sbis > 0 else -delta)
            fcur = euler_diff(xcur, sigma, x, grid, shocks)
        return xcur, 1

    @numba.njit(parallel=parallel)
    def K(sigma, grid, shocks, new, failed):
        for j in numba.prange(grid.size):
            root, status = brent(sigma, grid[j], grid, shocks)
            new[j], failed[j] = root, status

    @numba.njit
    def time_iter(grid, shocks):
        sigma, new = grid.copy(), np.empty_like(grid)
        failed = np.zeros(grid.size, np.int64)
        i, change = 0, t(TOL + 1)
        while change > TOL and i < MAX_ITER:
            K(sigma, grid, shocks, new, failed)
            if failed.any():
                return sigma, i + 1, 1
            change = np.max(np.abs(new - sigma))
            sigma, new = new, sigma
            i += 1
        return sigma, i, 0

    def solve(grid, shocks):
        sigma, i, failed = time_iter(grid, shocks)
        if failed:
            raise RuntimeError(f"a root find didn't converge in {BRENT_ITER} iterations, "
                               f"in iteration {i}")
        return sigma, i, {"exact_error": exact_error(sigma, grid)}

    return solve, (m["grid"], m["shocks"])


def jax(m, devices):
    import jax
    import jax.numpy as jnp

    xtol, rtol = XTOL[m["grid"].dtype.name], RTOL[m["grid"].dtype.name]

    def solve(grid, shocks):
        def euler_diff(c, sigma):
            # At every grid point at once: c and the result are [grid]
            k = grid - c
            y = (k**ALPHA)[:, None] * shocks[None, :]
            vals = (ALPHA * k**(ALPHA - 1))[:, None] * shocks[None, :] / jnp.interp(y, grid, sigma)
            return 1 / c - BETA * vals.mean(axis=1)

        def K(sigma):
            # SciPy's brentq loop at every grid point, each frozen once it
            # has converged, until all have or BRENT_ITER is reached
            xpre, xcur = jnp.full_like(grid, EDGE), grid - EDGE
            fpre, fcur = euler_diff(xpre, sigma), euler_diff(xcur, sigma)
            zero = jnp.zeros_like(grid)
            root = jnp.where(fpre == 0, xpre, xcur)
            done = (fpre == 0) | (fcur == 0)
            state = (0, xpre, xcur, zero, fpre, fcur, zero, zero, zero, root, done)

            def cond(state):
                it, *_, done = state
                return (it < BRENT_ITER) & ~done.all()

            def body(state):
                it, xpre, xcur, xblk, fpre, fcur, fblk, spre, scur, root, done = state
                new = (fpre != 0) & (fcur != 0) & ((fpre < 0) != (fcur < 0))
                xblk, fblk = jnp.where(new, xpre, xblk), jnp.where(new, fpre, fblk)
                spre = jnp.where(new, xcur - xpre, spre)
                scur = jnp.where(new, xcur - xpre, scur)
                swap = jnp.abs(fblk) < jnp.abs(fcur)
                xpre, xcur, xblk = (jnp.where(swap, xcur, xpre), jnp.where(swap, xblk, xcur),
                                    jnp.where(swap, xcur, xblk))
                fpre, fcur, fblk = (jnp.where(swap, fcur, fpre), jnp.where(swap, fblk, fcur),
                                    jnp.where(swap, fcur, fblk))
                delta = (xtol + rtol * jnp.abs(xcur)) / 2
                sbis = (xblk - xcur) / 2
                converged = ~done & ((fcur == 0) | (jnp.abs(sbis) < delta))
                root = jnp.where(converged, xcur, root)
                done = done | converged
                # Interpolate or extrapolate, or bisect
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = jnp.where(xpre == xblk,
                                 -fcur * (xcur - xpre) / (fcur - fpre),
                                 -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre)))
                good = ((jnp.abs(spre) > delta) & (jnp.abs(fcur) < jnp.abs(fpre))
                        & (2 * jnp.abs(stry) < jnp.minimum(jnp.abs(spre), 3 * jnp.abs(sbis) - delta)))
                spre, scur = jnp.where(good, scur, sbis), jnp.where(good, stry, sbis)
                xpre, fpre = xcur, fcur
                step = jnp.where(jnp.abs(scur) > delta, scur, jnp.where(sbis > 0, delta, -delta))
                xcur = jnp.where(done, xcur, xcur + step)
                fcur = euler_diff(xcur, sigma)
                return it + 1, xpre, xcur, xblk, fpre, fcur, fblk, spre, scur, root, done

            *_, xcur, _, _, _, _, _, root, done = jax.lax.while_loop(cond, body, state)
            return jnp.where(done, root, xcur), jnp.sum(~done)

        def cond(state):
            i, sigma, change, unconverged = state
            return (change > TOL) & (i < MAX_ITER)

        def body(state):
            i, sigma, _, unconverged = state
            new, failed = K(sigma)
            return i + 1, new, jnp.max(jnp.abs(new - sigma)), unconverged + failed

        init = (0, grid, jnp.asarray(TOL + 1, grid.dtype), jnp.zeros((), int))
        i, sigma, _, unconverged = jax.lax.while_loop(cond, body, init)
        return sigma, i, {"exact_error": exact_error(sigma, grid), "unconverged": unconverged}

    return jax.jit(solve), tuple(jax.device_put((m["grid"], m["shocks"]), devices[0]))


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
sample = fixed_positions
