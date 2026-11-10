"""
P7: a sequential recursion, bound by the latency of each step.

The Kalman filter log-likelihood of precision/kernels.py's `kalman`: K states,
P_OBS observations, T_STEPS periods. Each step is a handful of small matrix
products and solves, and no step can start before the last ends, so the
series' length only scales every stack's time in proportion. What moves a
crossover is the work in each step, so there are two sweeps:

- `batch`: how many independent filters run together, at 10 states.
  Each has its own transition matrix A and all read the same data, as when
  a likelihood is evaluated across parameter values. JAX runs them with
  `vmap`, NumPy as one batched array, and Numba as a loop over filters
  (`prange` when parallel). `batch20`, `batch50` and `batch100` are the
  same at 20, 50 and 100 states, for benchmarks#58, and run only when
  named; spec.KALMAN_BATCH_STATES maps each sweep to its states.
- `state`: the number of states K, for one filter. Each step's products cost
  about K^3.

A's off-diagonal draws are scaled by 1/sqrt(K), so its spectral radius stays
near 0.93 at every K; at K = 10 this is precision's matrix.
"""
import numpy as np

from spec import KALMAN_BATCH_STATES

T_STEPS, P_OBS = 1_000, 3


def model(sweep, size, dtype):
    n, k = (size, KALMAN_BATCH_STATES[sweep]) if sweep in KALMAN_BATCH_STATES else (1, size)
    rng = np.random.default_rng(0)
    draw = lambda *shape: rng.standard_normal(shape, dtype=np.float32)
    A = 0.9 * np.eye(k, dtype=np.float32) + 0.01 * np.sqrt(10 / k) * draw(n, k, k)
    return dict(A=A.astype(dtype), H=draw(P_OBS, k).astype(dtype),
                Y=draw(T_STEPS, P_OBS).astype(dtype),
                Q=(0.1 * np.eye(k)).astype(dtype), R=(0.5 * np.eye(P_OBS)).astype(dtype))


def _inputs(m):
    return m["A"], m["H"], m["Y"], m["Q"], m["R"]


def numpy(m):
    def solve(A, H, Y, Q, R):
        n, k = A.shape[:2]
        x = np.zeros((n, k), A.dtype)
        P = np.repeat(np.eye(k, dtype=A.dtype)[None], n, axis=0)
        ll = np.zeros(n, A.dtype)
        At = A.transpose(0, 2, 1)
        for y in Y:
            x, P = (A @ x[..., None])[..., 0], A @ P @ At + Q       # predict
            HP = H @ P
            S = HP @ H.T + R
            e = y - x @ H.T
            K = np.linalg.solve(S, HP).transpose(0, 2, 1)           # P H' S^-1
            x, P = x + (K @ e[..., None])[..., 0], P - K @ HP        # update
            Se = np.linalg.solve(S, e[..., None])[..., 0]
            ll = ll - 0.5 * (np.sum(e * Se, axis=1) + np.linalg.slogdet(S)[1])
        return np.concatenate([x, ll[:, None]], axis=1), Y.shape[0]

    return solve, _inputs(m)


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    half = m["A"].dtype.type(0.5)

    @numba.njit
    def filter_one(A, H, Y, Q, R, out):
        k = A.shape[0]
        x = np.zeros(k, A.dtype)
        P = np.eye(k, dtype=A.dtype)
        At, Ht = np.ascontiguousarray(A.T), np.ascontiguousarray(H.T)
        ll = out[k]                                                 # zero, in the cell's precision
        for t in range(Y.shape[0]):
            x, P = A @ x, A @ P @ At + Q                            # predict
            HP = H @ P
            S = HP @ Ht + R
            e = Y[t] - H @ x
            K = np.ascontiguousarray(np.linalg.solve(S, HP).T)      # P H' S^-1
            x, P = x + K @ e, P - K @ HP                            # update
            ll -= half * (e @ np.linalg.solve(S, e) + np.linalg.slogdet(S)[1])
        out[:k] = x
        out[k] = ll

    @numba.njit(parallel=parallel)
    def solve(A, H, Y, Q, R):
        n, k = A.shape[:2]
        out = np.zeros((n, k + 1), A.dtype)
        for b in numba.prange(n):
            filter_one(A[b], H, Y, Q, R, out[b])
        return out, Y.shape[0]

    return solve, _inputs(m)


def jax(m, devices):
    import jax
    import jax.numpy as jnp

    # precision/kernels.py's kalman, with its noise matrices passed in
    def loglik(A, H, Y, Q, R):
        def step(carry, y):
            x, P, ll = carry
            x, P = A @ x, A @ P @ A.T + Q                          # predict
            S = H @ P @ H.T + R
            e = y - H @ x
            K = jnp.linalg.solve(S, H @ P).T                       # P H' S^-1
            x, P = x + K @ e, P - K @ H @ P                        # update
            ll = ll - 0.5 * (e @ jnp.linalg.solve(S, e) + jnp.linalg.slogdet(S)[1])
            return (x, P, ll), None

        k = A.shape[0]
        init = (jnp.zeros(k, Y.dtype), jnp.eye(k, dtype=Y.dtype), jnp.zeros((), Y.dtype))
        (x, _, ll), _ = jax.lax.scan(step, init, Y)
        return jnp.append(x, ll)

    def solve(A, H, Y, Q, R):
        out = jax.vmap(loglik, in_axes=(0, None, None, None, None))(A, H, Y, Q, R)
        return out, Y.shape[0]

    return jax.jit(solve), tuple(jax.device_put(_inputs(m), devices[0]))


def sample(answer):
    """Up to 32 filters' log-likelihoods, then the first filter's final state."""
    out = np.asarray(answer, dtype=float)
    ll = out[:, -1]
    idx = np.linspace(0, ll.size - 1, min(ll.size, 32)).round().astype(int)
    return ll[idx].tolist() + out[0, :-1][:32].tolist()


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
