"""
P4: Monte Carlo simulation of many paths, parallel across paths and
sequential in time.

The cross-section of household wealth in lecture-jax's `wealth_dynamics_jax`,
with its parameters: every household starts at mean income and is shifted
forward SIM_LENGTH periods by a Kesten process,

    w' = y + R s_0 w  if w >= ŵ,   w' = y  otherwise,

with labour income y and returns R lognormal around a common aggregate state
z. The aggregate path is drawn once, by NumPy in `model`, and shared by every
stack, so given z the households are independent and the stacks' answers can
be compared by binomial standard errors (`sample` and summarise.py). Each
stack draws the households' shocks with its own generator, inside the timed
solve, as the lecture does.

The sweep is the number of households.
"""
import numpy as np

W_HAT, S_0 = 1.0, 0.75
C_Y, MU_Y, SIGMA_Y = 1.0, 1.0, 0.2
C_R, MU_R, SIGMA_R = 0.05, 0.1, 0.5
A, B, SIGMA_Z = 0.5, 0.0, 0.1
SIM_LENGTH = 200
SEED = 1234

# Wealth levels whose exceedance shares are compared between cells: about
# the 10th, 25th, 50th, 75th, 90th, 95th and 99th percentiles after SIM_LENGTH
# periods, from NumPy in float64 with a million households
THRESHOLDS = [15.0, 20.0, 40.0, 80.0, 200.0, 350.0, 1350.0]


def model(sweep, size, dtype):
    z_mean = B / (1 - A)
    z_var = SIGMA_Z**2 / (1 - A**2)
    exp_z_mean = np.exp(z_mean + z_var / 2)
    y_mean = C_Y * exp_z_mean + np.exp(MU_Y + SIGMA_Y**2 / 2)
    rng = np.random.default_rng(0)
    z = np.empty(SIM_LENGTH + 1)
    z[0] = z_mean
    for t in range(SIM_LENGTH):
        z[t + 1] = A * z[t] + B + SIGMA_Z * rng.standard_normal()
    return dict(w0=np.full(size, y_mean, dtype), z=z.astype(dtype))


def numpy(m):
    def solve(w0, z):
        rng = np.random.default_rng(SEED)
        w = w0
        for t in range(1, z.size):
            U = rng.standard_normal((2, w.size), dtype=w.dtype)
            y = C_Y * np.exp(z[t]) + np.exp(MU_Y + SIGMA_Y * U[0])
            R = C_R * np.exp(z[t]) + np.exp(MU_R + SIGMA_R * U[1])
            w = y + np.where(w < W_HAT, 0, R * S_0 * w)
        return w, z.size - 1

    return solve, (m["w0"], m["z"])


def numba(m, parallel):
    import numba

    # Numba promotes a float32 array to float64 when it meets a Python float,
    # so the constants take the cell's precision
    t = m["w0"].dtype.type
    w_hat, s_0, c_y, mu_y, sigma_y, c_r, mu_r, sigma_r = (
        t(W_HAT), t(S_0), t(C_Y), t(MU_Y), t(SIGMA_Y), t(C_R), t(MU_R), t(SIGMA_R))

    # The lecture's update_cross_section: each household's path in turn, with
    # a return drawn only for households above the savings threshold
    @numba.njit(parallel=parallel)
    def solve(w0, z):
        np.random.seed(SEED)
        w_out = np.empty_like(w0)
        for i in numba.prange(w0.size):
            w = w0[i]
            for s in range(1, z.size):
                wp = c_y * np.exp(z[s]) + np.exp(mu_y + sigma_y * t(np.random.randn()))
                if w >= w_hat:
                    R = c_r * np.exp(z[s]) + np.exp(mu_r + sigma_r * t(np.random.randn()))
                    wp += R * s_0 * w
                w = wp
            w_out[i] = w
        return w_out, z.size - 1

    return solve, (m["w0"], m["z"])


def jax(m, devices):
    import jax
    import jax.numpy as jnp

    # The lecture's update_cross_section_jax_compiled
    def solve(w0, z, key):
        def body(s, state):
            key, w = state
            key, subkey = jax.random.split(key)
            U = jax.random.normal(subkey, (2, w.size), w.dtype)
            y = C_Y * jnp.exp(z[s]) + jnp.exp(MU_Y + SIGMA_Y * U[0])
            R = C_R * jnp.exp(z[s]) + jnp.exp(MU_R + SIGMA_R * U[1])
            return key, y + jnp.where(w < W_HAT, 0, R * S_0 * w)

        _, w = jax.lax.fori_loop(1, z.size, body, (key, w0))
        return w, z.size - 1

    inputs = jax.device_put((m["w0"], m["z"]), devices[0]) + (jax.random.key(SEED),)
    return jax.jit(solve), inputs


def sample(answer):
    """The share of households above each threshold. The tail is too heavy
    for the mean to compare well."""
    w = np.asarray(answer, dtype=float)
    return [float(np.mean(w > x)) for x in THRESHOLDS]


BUILDERS = {"numpy": numpy, "numba": numba, "jax": jax}
