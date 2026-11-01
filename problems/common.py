"""
What the problem modules share: the income chain, and the answer's sample.

NumPy only, so a NumPy cell imports neither JAX nor Numba.
"""
from math import erf, sqrt

import numpy as np

SAMPLES = 64


def tauchen(n, rho, sigma, n_std=3):
    """
    Tauchen's discretisation of y' = rho y + sigma e, as quantecon's `tauchen`
    computes it with its defaults: n states spanning n_std unconditional
    standard deviations either side of zero, and the transition matrix.
    """
    std = sigma / sqrt(1 - rho**2)
    x = np.linspace(-n_std * std, n_std * std, n)
    half = (x[1] - x[0]) / 2
    cdf = np.vectorize(lambda z: 0.5 * (1 + erf(z / sqrt(2))))
    upper = cdf((x[None, :] - rho * x[:, None] + half) / sigma)
    lower = cdf((x[None, :] - rho * x[:, None] - half) / sigma)
    P = upper - lower
    P[:, 0] = upper[:, 0]
    P[:, -1] = 1 - lower[:, -1]
    return x, P


def fixed_positions(answer):
    """Values at fixed positions in the answer, for comparing cells."""
    flat = np.ravel(answer)
    idx = np.linspace(0, flat.size - 1, min(flat.size, SAMPLES)).round().astype(int)
    return flat[idx].astype(float).tolist()
