"""
P8's third companion: the same Newton–Raphson as `mle`, in Numba only, with
one `prange` pass over the observations in which each block of them forms its
own μ, G and H with BLAS while it is in cache. `mle`'s Numba forms G and H
through BLAS over all of X, and `mle_onepass` adds one observation at a time,
so the three show how to structure the pass as the parameters grow.

The model, the loop and the checks are `mle`'s, and the answers are checked
against `mle`'s NumPy rows.
"""
from mle import model, numba_blocked, sample, verify


def numba(m, parallel):
    return numba_blocked(m, parallel)


BUILDERS = {"numba": numba}
