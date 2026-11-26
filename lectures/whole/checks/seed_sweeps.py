"""float64 sweeps over seeds, for the lecture-jax issues: hopenhayn's p*, mle_jax's Newton-Raphson, markov_asset_jax's check.

    python seed_sweeps.py <lecture-jax lectures/ at 4b939ec> [hopenhayn|mle|markov|factorial]...

Executes the lectures' own cells unchanged; only the seed, the draw count or the iteration count varies.
"""
import contextlib
import io
import re
import sys

import numpy as np

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

L = sys.argv[1]
which = sys.argv[2:] or ["hopenhayn", "mle", "markov", "factorial"]


def cells(name, keep):
    src = re.findall(r"^```\{code-cell\}[^\n]*\n(.*?)^```", open(f"{L}/{name}.md").read(), re.S | re.M)
    ns = {}
    for i in keep:
        exec("\n".join(l for l in src[i - 1].splitlines()
                       if not l.startswith((":", "!")) and "matplotlib" not in l and "statsmodels" not in l
                       and "jax_enable_x64" not in l), ns)
    return ns


if "hopenhayn" in which:
    ns = cells("hopenhayn", [2] + list(range(4, 21)))
    for size in (200, 2000):
        ps = []
        for seed in range(1234, 1244):
            parameters, grids = ns["create_model"](seed=seed, E_draw_size=size, A_draw_size=size)
            ps.append(float(ns["compute_p_star"](parameters, grids)[0]))
        ps = np.array(ps)
        print(f"hopenhayn, {size} draws, seeds 1234-1243: p* {np.round(ps, 4).tolist()}; "
              f"mean {ps.mean():.4f}, sd {ps.std(ddof=1):.4f} ({ps.std(ddof=1) / ps.mean():.1%}), range {ps.min():.4f} to {ps.max():.4f}")

if "mle" in which or "factorial" in which:
    ns = cells("mle_jax", [1] + list(range(7, 12)))     # imports, the model, the likelihood and newton_raphson

if "mle" in which:
    out = []
    for seed in range(20):
        key = jax.random.key(seed)
        x = jax.random.normal(key, (500_000, 1))
        y = jax.random.poisson(key, jnp.exp(-2.5 + 0.25 * x + 0.5 * x**2), (500_000, 1))
        X = jnp.hstack((jnp.ones((500_000, 1)), x, x**2))
        with contextlib.redirect_stdout(io.StringIO()) as log:
            b = ns["newton_raphson"](ns["create_regression_model"](X, y), jnp.array([0.1, 0.1, 0.1]).reshape(3, 1), tol=1e-5, display=False)
        n = int(re.search(r"Number of iterations: (\d+)", log.getvalue()).group(1))
        b = np.asarray(b).ravel()
        ok = n < 100 and np.all(np.isfinite(b)) and np.allclose(b, [-2.5, 0.25, 0.5], atol=0.05)
        out.append((seed, n, ok, int(y.max())))
    print("mle_jax, seeds 0-19, float64, from [0.1, 0.1, 0.1]: "
          f"{sum(o[2] for o in out)} of 20 converge near the true β")
    for seed, n, ok, ymax in out:
        if not ok:
            print(f"  seed {seed}: {n} iterations, largest y {ymax}")

if "factorial" in which:
    y = jnp.array([20, 21, 25, 100, 9779])
    got = jnp.log(ns["jax_factorial"](y))
    true = jax.scipy.special.gammaln(y + 1.0)
    print(f"mle_jax's log(jax_factorial(y)) for y = {y.tolist()}: {np.round(np.asarray(got), 2).tolist()}; "
          f"log(y!) is {np.round(np.asarray(true), 2).tolist()} (backend {jax.default_backend()}, {jax.devices()[0].device_kind})")

if "markov" in which:
    ns = cells("markov_asset_jax", (3, 5, 7, 13, 14, 16))
    A = jnp.asarray(ns["compute_A"](ns["sv_model"]))
    exact = float(np.max(np.abs(np.linalg.eigvals(np.asarray(A)))))

    def estimate(b, k):
        sr = 0
        for _ in range(k):
            b1 = A @ b
            sr = jnp.sum(b1 * b) / jnp.sum(b * b)
            b = b1 / jnp.linalg.norm(b1)
        return float(sr)

    print(f"markov_asset_jax: exact spectral radius {exact:.4f}")
    for k in (15, 100, 1000):
        ests = np.array([estimate(jax.random.normal(jax.random.key(s), (A.shape[1],)), k) for s in range(1234, 1254)])
        print(f"  {k:>4} steps, seeds 1234-1253: estimates {ests.min():.4f} to {ests.max():.4f}; "
              f"{int((ests >= 1).sum())} of 20 at or above 1, where the lecture's assertion fails")
