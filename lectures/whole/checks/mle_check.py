"""mle_jax's Newton-Raphson, cell #19, with the data drawn in one precision and solved in another.

    python mle_check.py gen <64|32> <out.npz>          draw x and y as the lecture does
    python mle_check.py solve <64|32> <data.npz>       solve as the lecture does

The lecture's code, from lecture-jax at 4b939ec, is unchanged apart from loading the data.
"""
import sys
from typing import NamedTuple

import numpy as np

mode, bits, path = sys.argv[1], sys.argv[2], sys.argv[3]
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", bits == "64")

if mode == "gen":
    β_0, β_1, β_2 = -2.5, 0.25, 0.5
    seed = 32
    shape = (500_000, 1)
    key = jax.random.key(seed)
    x = jax.random.normal(key, shape)
    λ = jnp.exp(β_0 + β_1 * x + β_2 * x**2)
    y = jax.random.poisson(key, λ, shape)
    np.savez(path, x=np.asarray(x), y=np.asarray(y))
    print(f"x {x.dtype}: max |x| {float(jnp.max(jnp.abs(x))):.3f}; y {y.dtype}: max {int(y.max())}")
    sys.exit()

d = np.load(path)
x, y = jnp.asarray(d["x"].astype(jnp.zeros(0).dtype)), jnp.asarray(d["y"].astype(jnp.zeros(0, int).dtype))
shape = x.shape


class RegressionModel(NamedTuple):
    X: jnp.ndarray
    y: jnp.ndarray


def create_regression_model(X, y):
    n, k = X.shape
    y = y.reshape(n, 1)
    X = X.reshape(n, k)
    return RegressionModel(X=X, y=y)


@jax.jit
def _factorial(n):
    return jax.lax.exp(jax.lax.lgamma(n + 1.0)).astype(int)

jax_factorial = jax.vmap(_factorial)


@jax.jit
def poisson_logL(β, model):
    y = model.y
    μ = jnp.exp(model.X @ β)
    return jnp.sum(model.y * jnp.log(μ) - μ - jnp.log(jax_factorial(y)))

G_poisson_logL = jax.grad(poisson_logL)
H_poisson_logL = jax.jacfwd(G_poisson_logL)


def newton_raphson(model, β, tol=1e-3, max_iter=100):
    i, error = 0, 100
    while jnp.any(error > tol) and i < max_iter:
        H, G = jnp.squeeze(H_poisson_logL(β, model)), G_poisson_logL(β, model)
        β_new = β - (jnp.dot(jnp.linalg.inv(H), G))
        error = jnp.abs(β_new - β)
        β = β_new
        if i < 2:
            print(f"  iteration {i}: β {np.round(np.asarray(β).ravel(), 3)}, logL {float(poisson_logL(β, model)):.6g}")
        i += 1
    print(f"  iterations {i}, β_hat {np.asarray(β).ravel()}")
    return β


X = jnp.hstack((jnp.ones(shape), x, x**2))
init_β = jnp.array([0.1, 0.1, 0.1]).reshape(X.shape[1], 1)
poi = create_regression_model(X, y)
H0 = jnp.squeeze(H_poisson_logL(init_β, poi))
print(f"  X {X.dtype}; cond(H) at the start {float(jnp.linalg.cond(H0)):.3g}")
newton_raphson(poi, init_β, tol=1e-5)
