"""
P8: maximum likelihood by Newton–Raphson, with the derivatives by hand or by
autodiff.

The Poisson regression of QuantEcon's maximum likelihood lectures,
lecture-jax's `mle` and lecture-python's `mle`: y_i ~ Poisson(μ_i), with
μ_i = exp(x_i'β), fitted from β = 0.1 in every coordinate by

    β ← β − H⁻¹ G,   G = X'(y − μ),   H = −X' diag(μ) X

until no coordinate changes by more than the lecture exercise's 1e-5, or the
lectures' 100 iterations. The step is a linear solve, where the lectures
invert H. NumPy and Numba use the hand-written G and H that lecture-python's
`mle` had before it moved to JAX. JAX takes `jax.grad` of the lectures'
log-likelihood and `jax.jacfwd` of that, as both lectures do now, inside
`lax.while_loop`. Its companion problem, `mle_hand`, runs this module's JAX
loop with the hand-written G and H, so JAX against JAX measures autodiff.
Two more, `mle_onepass` and `mle_blocked`, run Numba with μ, G and H in one
pass over the observations, added one observation at a time or by BLAS a
block at a time, where this problem's Numba forms G and H through BLAS over
all of X.

Two sweeps: `obs` grows the observations with the exercise's model,
X = [1, x, x²] and β = (−2.5, 0.25, 0.5); `params` grows the parameters at
100,000 observations, with X = [1, z_1, …, z_{k−1}] standard normal,
β_0 = −2 and slopes in a fixed random direction of length 0.75, so that log μ
has the `obs` model's mean and spread. The data are drawn once by NumPy.

A cell records the last step's largest change as `last_step`, and `verify`
fails one that reached 100 iterations or whose estimate isn't finite.
"""
import numpy as np

from common import fixed_positions

START = 0.1             # every coordinate of β's start, the lectures'
TOL = 1e-5              # the largest change in β, the lecture exercise's
MAX_ITER = 100          # the lectures'
OBS_BETA = (-2.5, 0.25, 0.5)
PARAMS_N, PARAMS_INTERCEPT, PARAMS_SPREAD = 100_000, -2.0, 0.75
SEED = 1234
BLOCK = 4096            # observations a block, in `mle_onepass`'s Numba pass
# and in `mle_blocked`'s, the fastest of 256, 512 and 1,024 on the M6
BLAS_BLOCK = {"float64": 256, "float32": 1024}


def model(sweep, size, dtype):
    rng = np.random.default_rng(SEED)
    if sweep == "obs":
        x = rng.standard_normal(size)
        X = np.column_stack([np.ones(size), x, x**2])
        beta = np.array(OBS_BETA)
    elif sweep == "params":
        X = np.column_stack([np.ones(PARAMS_N), rng.standard_normal((PARAMS_N, size - 1))])
        u = rng.standard_normal(size - 1)
        beta = np.concatenate([[PARAMS_INTERCEPT], PARAMS_SPREAD * u / np.linalg.norm(u)])
    else:
        raise ValueError(f"P8 has no sweep {sweep!r}")
    y = rng.poisson(np.exp(X @ beta)).astype(float)
    return dict(X=X.astype(dtype), y=y.astype(dtype))


def verify(row):
    if row["iterations"] >= MAX_ITER:
        raise RuntimeError(f"Newton–Raphson reached its cap of {MAX_ITER} iterations")
    # A NaN or infinite step stops the loop as if it had converged
    if not row["last_step"] <= TOL:
        raise RuntimeError(f"the estimate isn't finite: the last step changed it by {row['last_step']}")


def numpy(m):
    def solve(X, y):
        beta = np.full(X.shape[1], START, X.dtype)
        i, step = 0, np.inf
        while step > TOL and i < MAX_ITER:
            mu = np.exp(X @ beta)
            G = X.T @ (y - mu)
            H = -(X.T @ (mu[:, None] * X))
            new = beta - np.linalg.solve(H, G)
            step = np.max(np.abs(new - beta))
            beta, i = new, i + 1
        return beta, i, {"last_step": step}

    return solve, (m["X"], m["y"])


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["X"].dtype.type
    start, tol, inf = t(START), t(TOL), t(np.inf)

    # μ and y − μ in one pass over the observations
    @numba.njit(parallel=parallel)
    def mu_resid(X, y, beta, mu, r):
        n, k = X.shape
        for i in numba.prange(n):
            z = X[i, 0] * beta[0]
            for j in range(1, k):
                z += X[i, j] * beta[j]
            mu[i] = np.exp(z)
            r[i] = y[i] - mu[i]

    @numba.njit
    def newton(X, y):
        n, k = X.shape
        beta = np.full(k, start)
        mu, r = np.empty(n, X.dtype), np.empty(n, X.dtype)
        i, step = 0, inf
        while step > tol and i < MAX_ITER:
            mu_resid(X, y, beta, mu, r)
            G = X.T @ r
            H = -(X.T @ (mu.reshape((n, 1)) * X))
            new = beta - np.linalg.solve(H, G)
            step = np.max(np.abs(new - beta))
            beta = new
            i += 1
        return beta, i, step

    def solve(X, y):
        beta, i, step = newton(X, y)
        return beta, i, {"last_step": step}

    return solve, (m["X"], m["y"])


def numba_onepass(m, parallel):
    """
    Numba with μ, G and H in one pass over the observations, added one
    observation at a time (`mle_onepass`).
    """
    import numba

    @numba.njit(parallel=parallel)
    def derivatives(X, y, beta, Gs, Hs):
        n, k = X.shape
        for b in numba.prange(Gs.shape[0]):
            g = np.zeros(k, X.dtype)
            h = np.zeros((k, k), X.dtype)       # its lower triangle
            for i in range(b * BLOCK, min(n, (b + 1) * BLOCK)):
                z = X[i, 0] * beta[0]
                for j in range(1, k):
                    z += X[i, j] * beta[j]
                mu = np.exp(z)
                r = y[i] - mu
                for p in range(k):
                    g[p] += X[i, p] * r
                    w = mu * X[i, p]
                    for q in range(p + 1):
                        h[p, q] -= w * X[i, q]
            Gs[b] = g
            Hs[b] = h

    return numba_blocks(m, derivatives, BLOCK, lower=True)


def numba_blocked(m, parallel):
    """
    Numba with μ, G and H in one pass over the observations, each block of
    BLAS_BLOCK of them forming its own with BLAS while it is in cache
    (`mle_blocked`).
    """
    import numba

    block = BLAS_BLOCK[m["X"].dtype.name]

    @numba.njit(parallel=parallel)
    def derivatives(X, y, beta, Gs, Hs):
        n = X.shape[0]
        for b in numba.prange(Gs.shape[0]):
            lo, hi = b * block, min(n, (b + 1) * block)
            Xb = X[lo:hi]
            mu = np.exp(Xb @ beta)
            Gs[b] = Xb.T @ (y[lo:hi] - mu)
            Hs[b] = -(Xb.T @ (mu.reshape((hi - lo, 1)) * Xb))

    return numba_blocks(m, derivatives, block, lower=False)


def numba_blocks(m, derivatives, block, lower):
    """
    The Newton loop of `mle_onepass` and `mle_blocked`: each block of `block`
    observations forms its own G and H, its lower triangle if `lower`, and
    the blocks' are then added up, so threads don't share cache lines and
    float32's sums stay short.
    """
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["X"].dtype.type
    start, tol, inf = t(START), t(TOL), t(np.inf)

    @numba.njit
    def newton(X, y):
        n, k = X.shape
        blocks = (n + block - 1) // block
        Gs, Hs = np.empty((blocks, k), X.dtype), np.empty((blocks, k, k), X.dtype)
        beta = np.full(k, start)
        i, step = 0, inf
        while step > tol and i < MAX_ITER:
            derivatives(X, y, beta, Gs, Hs)
            G, H = np.zeros(k, X.dtype), np.zeros((k, k), X.dtype)
            for b in range(blocks):
                G += Gs[b]
                H += Hs[b]
            if lower:
                for p in range(k):
                    for q in range(p):
                        H[q, p] = H[p, q]
            new = beta - np.linalg.solve(H, G)
            step = np.max(np.abs(new - beta))
            beta = new
            i += 1
        return beta, i, step

    def solve(X, y):
        beta, i, step = newton(X, y)
        return beta, i, {"last_step": step}

    return solve, (m["X"], m["y"])


def jax_newton(m, devices, autodiff):
    """JAX's Newton loop, with G and H by autodiff (P8) or by hand (`mle_hand`)."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import gammaln

    if autodiff:
        # The lectures' log-likelihood. The last term doesn't depend on β, so
        # it drops out of G and H
        def logL(beta, X, y):
            mu = jnp.exp(X @ beta)
            return jnp.sum(y * jnp.log(mu) - mu - gammaln(y + 1))

        G, H = jax.grad(logL), jax.jacfwd(jax.grad(logL))

        def derivatives(beta, X, y):
            return G(beta, X, y), H(beta, X, y)
    else:
        # G = X'(y − μ), written as (y − μ) X: XLA's CPU backend copied X
        # transposed every iteration for X.T @ (y − μ), and reads X as it is
        # for this, as for jax.grad's G
        def derivatives(beta, X, y):
            mu = jnp.exp(X @ beta)
            return (y - mu) @ X, -(X.T @ (mu[:, None] * X))

    def solve(X, y):
        def cond(state):
            i, _, step = state
            return (step > TOL) & (i < MAX_ITER)

        def body(state):
            i, beta, _ = state
            G, H = derivatives(beta, X, y)
            new = beta - jnp.linalg.solve(H, G)
            return i + 1, new, jnp.max(jnp.abs(new - beta))

        init = (0, jnp.full(X.shape[1], START, X.dtype), jnp.asarray(jnp.inf, X.dtype))
        i, beta, step = jax.lax.while_loop(cond, body, init)
        return beta, i, {"last_step": step}

    return jax.jit(solve), tuple(jax.device_put((m["X"], m["y"]), devices[0]))


def jax(m, devices):
    return jax_newton(m, devices, autodiff=True)


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
sample = fixed_positions
