"""
P5: Markov chains, the stationary distribution of a household savings policy.

The household and prices of lecture-jax's `aiyagari_jax`: log utility, assets
a on an evenly spaced grid, income z on a two-state chain, and the budget
c + a' = (1 + r) a + w z. The lecture's equilibrium needs the stationary
distribution of (a, z) under the savings policy at every price it tries, and
aggregates households by it. There are two sweeps, one for each way of
computing it (#47):

  dense   the lecture's: Howard policy iteration chooses a' on the grid, so
          the policy is an index; the transition over every (a, z) is built
          as a dense matrix P, and (I - P' + 1 1') ψ = 1 is solved.
  sparse  Young's (2010) lottery, which also covers a policy off the grid,
          here EGM's: each state sends its mass to the two grid points either
          side of its savings, so P has at most four nonzeros a row, and the
          distribution is found by power iteration.

The policy is the input, computed by NumPy in float64 in `model`, outside the
timed cell. What is timed is building the transition from the policy, finding
the distribution, and summing out income. The answer is the distribution of
assets.

Power iteration stops when the L1 change falls below 1e-10 in float64 or 1e-6
in float32. float32's rounding leaves the change at about 1e-7, so below that
when or whether the loop stops depends on rounding (#47).

The sweep is the asset grid; income keeps the lecture's two states.
"""
import numpy as np

from common import fixed_positions

BETA, R, W = 0.96, 1.01, 1.0
Z = [0.1, 1.0]
PI = [[0.9, 0.1], [0.1, 0.9]]
A_MIN, A_MAX = 1e-10, 20.0
HPI_TOL, EGM_TOL = 1e-4, 1e-8
TOL = {"float64": 1e-10, "float32": 1e-6}
MAX_ITER = 10_000


def hpi_policy(a, z, Pi):
    """The lecture's Howard policy iteration, with log utility: an index policy."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n, nz = a.size, z.size
    c = W * z[None, :, None] + R * a[:, None, None] - a[None, None, :]
    u = np.where(c > 0, np.log(np.where(c > 0, c, 1)), -np.inf)      # u[i, j, ip]
    states = np.arange(n * nz)

    def value(sigma):
        # v = (I - β P_σ)^{-1} r_σ, with P_σ sparse: two nonzeros a row
        r = u[np.arange(n)[:, None], np.arange(nz)[None, :], sigma].ravel()
        cols = (sigma[:, :, None] * nz + np.arange(nz)).reshape(n * nz, nz)
        P = sp.csr_matrix((Pi[np.tile(np.arange(nz), n)].ravel(),
                           (np.repeat(states, nz), cols.ravel())), shape=(n * nz, n * nz))
        return spla.spsolve(sp.identity(n * nz, format="csr") - BETA * P, r).reshape(n, nz)

    sigma = np.zeros((n, nz), int)
    v = value(sigma)
    for _ in range(10_000):
        sigma = np.argmax(u + BETA * (v @ Pi.T).T[None, :, :], axis=2)
        v_new = value(sigma)
        error = np.max(np.abs(v_new - v))
        v = v_new
        if error < HPI_TOL:
            break
    return sigma


def egm_savings(a, z, Pi):
    """EGM with log utility on the grid itself: the savings policy a'(a, z)."""
    c = R * a[:, None] + W * z[None, :]                 # consume everything
    for _ in range(100_000):
        # Today's consumption and assets for each choice of a' on the grid
        c_endo = 1 / (BETA * R * ((1 / c) @ Pi.T))
        a_endo = (c_endo + a[:, None] - W * z[None, :]) / R
        c_new = np.empty_like(c)
        for j in range(z.size):
            c_new[:, j] = np.interp(a, a_endo[:, j], c_endo[:, j])
            binding = a < a_endo[0, j]
            c_new[binding, j] = R * a[binding] + W * z[j] - A_MIN
        error = np.max(np.abs(c_new - c))
        c = c_new
        if error < EGM_TOL:
            break
    return np.clip(R * a[:, None] + W * z[None, :] - c, A_MIN, A_MAX)


def model(sweep, size, dtype):
    a, z, Pi = np.linspace(A_MIN, A_MAX, size), np.array(Z), np.array(PI)
    m = dict(sweep=sweep, a=a.astype(dtype), Pi=Pi.astype(dtype))
    if sweep == "dense":
        m["sigma"] = hpi_policy(a, z, Pi).astype(np.int32)
    else:
        m["s"] = egm_savings(a, z, Pi).astype(dtype)
    return m


def inputs(m):
    return (m["a"], m["sigma"] if m["sweep"] == "dense" else m["s"], m["Pi"])


def numpy(m):
    import scipy.linalg
    import scipy.sparse as sp

    tol = TOL[m["a"].dtype.name]

    def dense(a, sigma, Pi):
        # M = I - P' + 1 1', as the lecture's compute_asset_stationary has it,
        # built in place: column (i, j) of P' has Π(j, j') at row (σ(i, j), j').
        # SciPy's solve, since NumPy's linalg computes in float64 even for
        # float32 arrays, and returns the answer rounded to float32
        n, nz = sigma.shape
        N = n * nz
        M = np.ones((N, N), a.dtype)
        M[np.arange(N), np.arange(N)] += 1
        rows = (sigma[:, :, None] * nz + np.arange(nz)).ravel()
        M[rows, np.repeat(np.arange(N), nz)] -= np.broadcast_to(Pi, (n, nz, nz)).ravel()
        psi = scipy.linalg.solve(M, np.ones(N, a.dtype), overwrite_a=True, check_finite=False)
        return psi.reshape(n, nz).sum(axis=1), 1

    def sparse(a, s, Pi):
        n, nz = s.shape
        k = np.clip(np.searchsorted(a, s, side="right") - 1, 0, n - 2)
        w = (s - a[k]) / (a[k + 1] - a[k])
        # P transposed, so each product pulls a state's mass from its sources
        jp = np.arange(nz)
        rows = np.stack([k[:, :, None] * nz + jp, (k[:, :, None] + 1) * nz + jp], axis=-1)
        vals = np.stack([(1 - w)[:, :, None] * Pi[None, :, :],
                         w[:, :, None] * Pi[None, :, :]], axis=-1)
        cols = np.broadcast_to(np.arange(n * nz).reshape(n, nz, 1, 1), rows.shape)
        PT = sp.csr_matrix((vals.ravel(), (rows.ravel(), cols.ravel())), shape=(n * nz, n * nz))
        psi = np.full(n * nz, 1 / (n * nz), a.dtype)
        i, error = 0, tol + 1
        while error > tol and i < MAX_ITER:
            new = PT @ psi
            error = np.abs(new - psi).sum()
            psi, i = new, i + 1
        return psi.reshape(n, nz).sum(axis=1), i

    return (dense if m["sweep"] == "dense" else sparse), inputs(m)


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["a"].dtype.type
    zero, one, tol = t(0), t(1), t(TOL[m["a"].dtype.name])

    @numba.njit(parallel=parallel)
    def fill(sigma, Pi, M):
        # M = I - P' + 1 1', written in place: column (i, j) of P' has Π(j, j')
        # at row (σ(i, j), j'), and no two (i, j) share a column
        n, nz = sigma.shape
        for r in numba.prange(n * nz):
            for c in range(n * nz):
                M[r, c] = one
            M[r, r] += one
        for i in numba.prange(n):
            for j in range(nz):
                for jp in range(nz):
                    M[sigma[i, j] * nz + jp, i * nz + j] -= Pi[j, jp]

    @numba.njit
    def dense(a, sigma, Pi):
        n, nz = sigma.shape
        M = np.empty((n * nz, n * nz), a.dtype)
        fill(sigma, Pi, M)
        psi = np.linalg.solve(M, np.ones(n * nz, a.dtype))
        return psi.reshape(n, nz).sum(axis=1), 1

    @numba.njit(parallel=parallel)
    def transposed(a, s, Pi):
        # P transposed in CSR: each state's four destinations and weights,
        # then a counting sort by destination. The indices are unsigned, so
        # Numba needn't check each one for a negative index from the end
        n, nz = s.shape
        N = n * nz
        dest = np.empty((N, 2 * nz), np.int64)
        weight = np.empty((N, 2 * nz), a.dtype)
        for i in numba.prange(n):
            for j in range(nz):
                k = min(max(np.searchsorted(a, s[i, j], side="right") - 1, 0), n - 2)
                w = (s[i, j] - a[k]) / (a[k + 1] - a[k])
                for jp in range(nz):
                    dest[i * nz + j, 2 * jp] = k * nz + jp
                    weight[i * nz + j, 2 * jp] = (one - w) * Pi[j, jp]
                    dest[i * nz + j, 2 * jp + 1] = (k + 1) * nz + jp
                    weight[i * nz + j, 2 * jp + 1] = w * Pi[j, jp]
        indptr = np.zeros(N + 1, np.int64)
        for d in dest.ravel():
            indptr[d + 1] += 1
        indptr = np.cumsum(indptr)
        indices = np.empty(dest.size, np.uint32)
        data = np.empty(dest.size, a.dtype)
        nxt = indptr[:-1].copy()
        for src in range(N):
            for e in range(2 * nz):
                p = nxt[dest[src, e]]
                indices[p], data[p] = src, weight[src, e]
                nxt[dest[src, e]] = p + 1
        return indptr.astype(np.uint64), indices, data

    @numba.njit(parallel=parallel)
    def step(indptr, indices, data, psi, new):
        # One product with P', and the L1 change. The change is summed in
        # float64: in float32, adding millions of tiny terms one at a time
        # stops changing the sum
        error = 0.0
        for r in numba.prange(psi.size):
            acc = zero
            for p in range(indptr[r], indptr[r + 1]):
                acc += data[p] * psi[indices[p]]
            new[r] = acc
            error += abs(acc - psi[r])
        return error

    @numba.njit
    def sparse(a, s, Pi):
        n, nz = s.shape
        indptr, indices, data = transposed(a, s, Pi)
        psi = np.empty(n * nz, a.dtype)
        psi[:] = 1 / (n * nz)
        new = np.empty_like(psi)
        i, error = 0, tol + 1
        while error > tol and i < MAX_ITER:
            error = step(indptr, indices, data, psi, new)
            psi, new = new, psi
            i += 1
        return psi.reshape(n, nz).sum(axis=1), i

    return (dense if m["sweep"] == "dense" else sparse), inputs(m)


def jax(m, devices):
    import jax
    import jax.numpy as jnp

    tol = TOL[m["a"].dtype.name]

    def dense(a, sigma, Pi):
        # The lecture's compute_asset_stationary and compute_stationary
        n, nz = sigma.shape
        ap_idx = jnp.reshape(jnp.arange(n), (1, 1, n, 1))
        A = jnp.where(jnp.reshape(sigma, (n, nz, 1, 1)) == ap_idx, 1, 0)
        P = jnp.reshape(A * jnp.reshape(Pi, (1, nz, 1, nz)), (n * nz, n * nz))
        M = jnp.identity(n * nz, a.dtype) - jnp.transpose(P) + jnp.ones((n * nz, n * nz), a.dtype)
        psi = jnp.linalg.solve(M, jnp.ones(n * nz, a.dtype))
        return jnp.reshape(psi, (n, nz)).sum(axis=1), 1

    def sparse(a, s, Pi):
        n, nz = s.shape
        k = jnp.clip(jnp.searchsorted(a, s, side="right") - 1, 0, n - 2)
        w = (s - a[k]) / (a[k + 1] - a[k])
        jp = jnp.arange(nz)
        dest = jnp.stack([k[:, :, None] * nz + jp, (k[:, :, None] + 1) * nz + jp],
                         axis=-1).reshape(n * nz, 2 * nz)
        weight = jnp.stack([(1 - w)[:, :, None] * Pi[None, :, :], w[:, :, None] * Pi[None, :, :]],
                           axis=-1).reshape(n * nz, 2 * nz)

        def cond(state):
            i, psi, error = state
            return (error > tol) & (i < MAX_ITER)

        def body(state):
            i, psi, _ = state
            # Each state's mass scattered to its four destinations
            new = jax.ops.segment_sum((psi[:, None] * weight).ravel(), dest.ravel(),
                                      num_segments=n * nz)
            return i + 1, new, jnp.abs(new - psi).sum()

        init = (0, jnp.full(n * nz, 1 / (n * nz), a.dtype), jnp.asarray(tol + 1, a.dtype))
        i, psi, _ = jax.lax.while_loop(cond, body, init)
        return jnp.reshape(psi, (n, nz)).sum(axis=1), i

    fn = dense if m["sweep"] == "dense" else sparse
    return jax.jit(fn), tuple(jax.device_put(inputs(m), devices[0]))


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}


def sample(psi):
    """
    The cumulative distribution of assets at fixed positions. The distribution
    itself is mostly zeros beside one atom at the borrowing constraint, so an
    error in its interior would barely register against the atom.
    """
    return fixed_positions(np.cumsum(np.asarray(psi, np.float64)))
