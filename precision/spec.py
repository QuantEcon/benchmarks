"""
The precision benchmark matrix: the workloads, their sizes, what each one
measures, and how much memory a cell needs.

Kept free of JAX so the driver can plan cells, and skip ones that cannot
fit, without initialising a backend.
"""

GIB = 2**30

# Rows of rewards the recomputing Bellman step holds at once
BLOCK = 512

# Each workload says what bounds it, the sizes to sweep, whether it can be
# split across devices, how to turn a time into a rate, and roughly how many
# bytes each device needs. `size` is n for matmul, the grid size m for the
# Bellman steps and the number of time steps T for the Kalman filter;
# `s` is the item size in bytes and `d` the number of devices.
WORKLOADS = {
    "matmul": dict(
        bound="compute",
        sizes=[256, 512, 1024, 2048, 4096, 8192],
        quick=[256, 1024],
        shardable=False,
        unit="GFLOPS",
        work=lambda n, s: 2 * n**3 / 1e9,
        memory=lambda n, s, d: 3 * n * n * s + 4 * n * n,
    ),
    # 60k x 60k is 14.4 GB in float32 and 28.8 GB in float64, so on a 24 GB
    # card the capacity line falls between the two precisions.
    "bellman_stored": dict(
        bound="memory",
        sizes=[1000, 2000, 5000, 10000, 20000, 40000, 60000],
        quick=[1000, 5000],
        shardable=True,
        unit="GB/s",
        work=lambda m, s: m * m * s / 1e9,
        memory=lambda m, s, d: m * m * s / d + 16 * m * s,
    ),
    "bellman_recompute": dict(
        bound="compute (transcendental)",
        sizes=[1000, 2000, 5000, 10000, 20000, 40000, 60000],
        quick=[1000, 5000],
        shardable=True,
        unit="Gcells/s",
        work=lambda m, s: m * m / 1e9,
        memory=lambda m, s, d: (2 * BLOCK + 16) * m * s,  # a block, plus a remainder
    ),
    "kalman": dict(
        bound="launch latency",
        sizes=[1000, 10000, 100000],
        quick=[1000],
        shardable=False,
        unit="ksteps/s",
        work=lambda t, s: t / 1e3,
        memory=lambda t, s, d: 64 * t * s,
    ),
}

ITEMSIZE = {"float32": 4, "float64": 8}
