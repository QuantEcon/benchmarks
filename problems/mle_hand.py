"""
P8's companion: the same Newton–Raphson as `mle`, in JAX only, with the
hand-written G and H that NumPy and Numba use there, so that `mle` against
`mle_hand` measures what autodiff costs JAX.

The model, the loop, the sweeps and the checks are `mle`'s, and the answers
are checked against `mle`'s NumPy rows.
"""
from mle import jax_newton, model, sample, verify


def jax(m, devices):
    return jax_newton(m, devices, autodiff=False)


BUILDERS = {"jax": jax}
