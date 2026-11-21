"""
P8's second companion: the same Newton–Raphson as `mle`, in Numba only, with
μ, G and H formed in one `prange` pass over the observations, added one
observation at a time, where `mle`'s Numba forms G and H as matrix products
through BLAS. It runs on `obs`, and on `params` to 300 parameters, to show
where adding one observation at a time stops paying as the parameters grow.

The model, the loop and the checks are `mle`'s, and the answers are checked
against `mle`'s NumPy rows.
"""
from mle import model, numba_onepass, sample, verify


def numba(m, parallel):
    return numba_onepass(m, parallel)


BUILDERS = {"numba": numba}
