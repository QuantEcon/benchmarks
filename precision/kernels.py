"""
The JAX kernels the precision benchmark times.

Each builder takes a size, a dtype and the devices to run on, and returns a
jitted function with its inputs already placed. Random inputs are drawn in
float32 and cast, and grids are built in the cell's own precision, so a
float32 cell and a float64 cell see the same values up to rounding and their
results can be compared.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    from jax import shard_map
except ImportError:  # JAX releases before shard_map left experimental
    from jax.experimental.shard_map import shard_map

from spec import BLOCK

BETA = 0.96


def _row_sharding(devices):
    """Shardings that split the leading axis across devices, and replicate."""
    if len(devices) == 1:
        return None, None
    mesh = Mesh(np.array(devices), ("rows",))
    return NamedSharding(mesh, P("rows")), NamedSharding(mesh, P())


def matmul(n, dtype, devices):
    """Dense matrix multiply: bound by arithmetic."""
    ka, kb = jax.random.split(jax.random.key(0))
    draw = jax.jit(lambda k: jax.random.uniform(k, (n, n), jnp.float32).astype(dtype))
    return jax.jit(jnp.matmul), (draw(ka), draw(kb))


def _grid(m, dtype):
    """Wealth today and tomorrow, on one grid."""
    return jnp.linspace(0.1, 10.0, m, dtype=dtype)


def _consumption(w, w_next, r=0.03, y=1.0):
    """What is consumed when wealth w is carried to w_next."""
    return (1 + r) * w + y - w_next


def _utility(c):
    """Log utility, with non-positive consumption ruled out."""
    return jnp.where(c > 0, jnp.log(jnp.where(c > 0, c, 1.0)), -jnp.inf)


def bellman_stored(m, dtype, devices):
    """One Bellman step over a stored m x m reward array: bound by memory."""
    rows, replicated = _row_sharding(devices)
    w = _grid(m, dtype)
    v = jnp.log(1.0 + w)
    rewards = lambda w: _utility(_consumption(w[:, None], w[None, :]))
    if rows is None:
        R = jax.jit(rewards)(w)
    else:
        w, v = jax.device_put(w, replicated), jax.device_put(v, replicated)
        R = jax.jit(rewards, out_shardings=rows)(w)
    step = jax.jit(lambda R, v: jnp.max(R + BETA * v[None, :], axis=1))
    return step, (R, v)


def bellman_recompute(m, dtype, devices):
    """
    The same Bellman step, recomputing rewards from the grid each time.

    This is how the stored step is made to fit a small card: memory traded
    for arithmetic, much of it transcendental. It works in blocks of rows,
    because written as one broadcast XLA's CPU backend materialises the full
    m x m array anyway, which saves nothing.
    """
    w = _grid(m, dtype)
    v = jnp.log(1.0 + w)

    def local(w_rows, w, v):
        row = lambda wi: jnp.max(_utility(_consumption(wi, w)) + BETA * v)
        return jax.lax.map(row, w_rows, batch_size=min(BLOCK, w_rows.shape[0]))

    if len(devices) == 1:
        return jax.jit(local), (w, w, v)
    mesh = Mesh(np.array(devices), ("rows",))
    step = shard_map(local, mesh=mesh, in_specs=(P("rows"), P(), P()), out_specs=P("rows"))
    rows, replicated = NamedSharding(mesh, P("rows")), NamedSharding(mesh, P())
    inputs = (jax.device_put(w, rows), jax.device_put(w, replicated),
              jax.device_put(v, replicated))
    return jax.jit(step), inputs


def kalman(T, dtype, devices):
    """A Kalman filter log-likelihood over T steps: many small sequential ops."""
    k, p = 10, 3
    k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
    f32 = jnp.float32
    A = (0.9 * jnp.eye(k, dtype=f32)
         + 0.01 * jax.random.normal(k1, (k, k), f32)).astype(dtype)
    H = jax.random.normal(k2, (p, k), f32).astype(dtype)
    Y = jax.random.normal(k3, (T, p), f32).astype(dtype)

    def loglik(A, H, Y):
        Q = 0.1 * jnp.eye(k, dtype=Y.dtype)
        R = 0.5 * jnp.eye(p, dtype=Y.dtype)

        def step(carry, y):
            x, P, ll = carry
            x, P = A @ x, A @ P @ A.T + Q                  # predict
            S = H @ P @ H.T + R
            e = y - H @ x
            K = jnp.linalg.solve(S, H @ P).T               # P H' S^-1; P, S symmetric
            x, P = x + K @ e, P - K @ H @ P                # update
            ll = ll - 0.5 * (e @ jnp.linalg.solve(S, e) + jnp.linalg.slogdet(S)[1])
            return (x, P, ll), None

        init = (jnp.zeros(k, Y.dtype), jnp.eye(k, dtype=Y.dtype), jnp.zeros((), Y.dtype))
        (x, _, ll), _ = jax.lax.scan(step, init, Y)
        return jnp.append(x, ll)

    return jax.jit(loglik), (A, H, Y)


BUILDERS = {
    "matmul": matmul,
    "bellman_stored": bellman_stored,
    "bellman_recompute": bellman_recompute,
    "kalman": kalman,
}
