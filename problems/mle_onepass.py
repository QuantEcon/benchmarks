"""
P8's second companion: the same Newton–Raphson as `mle`, in Numba only, with
μ, G and H formed in one `prange` pass over the observations, where `mle`'s
Numba forms G and H as matrix products through BLAS. It runs on the `obs`
sweep, where each iteration is a few passes over the data, to find the size
at which reading X once an iteration overtakes BLAS.

The model, the loop and the checks are `mle`'s, and the answers are checked
against `mle`'s NumPy rows.
"""
from mle import model, numba_onepass, sample, verify


def numba(m, parallel):
    return numba_onepass(m, parallel)


BUILDERS = {"numba": numba}
