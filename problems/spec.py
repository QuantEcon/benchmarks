"""
The problem-type suite: the problems, the sizes each one sweeps, the
backends, and roughly how much memory a cell needs.

Kept free of JAX and Numba so the driver can plan cells, and skip ones that
cannot fit, without starting either.
"""
GIB = 2**30
ITEMSIZE = {"float32": 4, "float64": 8}

# P7's batch sweeps and the states each filter has. `batch` is Phase 1's; the
# others are benchmarks#58's, to find where the crossover lies as both grow.
KALMAN_BATCH_STATES = {"batch": 10, "batch20": 20, "batch50": 50, "batch100": 100}

# A backend is a stack on a device. NumPy and Numba run on the CPU only.
# `numba` is the Numba source compiled with parallel=False, so its prange
# loops run in order; `numba-parallel` is the same source with parallel=True.
BACKENDS = ["numpy", "numba", "numba-parallel", "jax-cpu", "jax-gpu"]

# Each problem gives its label in benchmarks#5, the lecture its model comes
# from, the precisions it runs in, its sweeps (an `optional` one runs only
# when --sweeps names it), roughly how many bytes a cell
# needs on one device (`n` is the size and `s` the item size), and how its
# answer is checked against NumPy's in float64 at the same size:
#   close   the largest difference in the sampled answer, over the largest
#           value, is within rtol for the cell's precision;
#   shares  each share of households above a threshold is within z binomial
#           standard errors of NumPy's.
PROBLEMS = {
    "vfi": dict(
        label="P1",
        lecture="opt_savings_1",
        # In float32, a tolerance of 1e-5 on values near -50 is within a few
        # units of roundoff, so the loop need not stop
        dtypes=["float64"],
        sweeps={"grid": dict(sizes=[100, 200, 400, 700, 1000, 1500],
                             quick=[100])},
        # The stored rewards, w x 100 x w, and NumPy's temporaries of that shape
        memory=lambda sweep, n, s: 3 * n * 100 * n * s,
        check=dict(kind="close", rtol={"float64": 1e-8}),
    ),
    "egm": dict(
        label="P3",
        lecture="ifp_egm_jax",
        dtypes=["float64", "float32"],
        sweeps={"grid": dict(sizes=[200, 1000, 5000, 20000, 50000],
                             quick=[200])},
        memory=lambda sweep, n, s: 20 * n * 25 * s,
        check=dict(kind="close", rtol={"float64": 1e-8, "float32": 1e-3}),
    ),
    "wealth": dict(
        label="P4",
        lecture="wealth_dynamics_jax",
        dtypes=["float64", "float32"],
        sweeps={"households": dict(sizes=[1_000, 10_000, 100_000, 1_000_000, 10_000_000],
                                   quick=[10_000])},
        memory=lambda sweep, n, s: 12 * n * s,
        check=dict(kind="shares", z=5),
    ),
    "kalman": dict(
        label="P7",
        lecture=None,  # precision/kernels.py's kalman
        dtypes=["float64", "float32"],
        sweeps={"batch": dict(sizes=[1, 10, 100, 1_000, 10_000], quick=[10]),
                "state": dict(sizes=[2, 5, 10, 20, 50, 100, 200], quick=[10]),
                **{name: dict(sizes=[1, 10, 100, 300, 1_000, 3_000, 10_000], quick=[10],
                              optional=True)
                   for name in KALMAN_BATCH_STATES if name != "batch"}},
        memory=lambda sweep, n, s: (20 * n * KALMAN_BATCH_STATES[sweep] ** 2
                                    if sweep in KALMAN_BATCH_STATES else 20 * n * n) * s,
        check=dict(kind="close", rtol={"float64": 1e-8, "float32": 1e-3}),
    ),
}
