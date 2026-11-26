"""
A STREAM-style triad, a = b + s c, in float64 with Numba, at 1, 2, 4, 8 and 16 threads, up to as
many as Numba may use, and at that many: on a 12-core Mac, 1, 2, 4, 8 and 12. The arrays are far
larger than any cache: 2^28 elements, 2 GiB each, by default.

    NUMBA_NUM_THREADS=16 python stream_triad.py <out.jsonl> [log2 of elements] [reps]

The arrays are first touched in parallel, by the threads that use them. Each thread count runs
once to warm up, then `reps` times (default 10, STREAM's own). Each output line is one thread
count: the best and median time, and GB/s at the best time, counted as STREAM counts, 24 bytes per
element: two reads and a write. On x86 a store without a non-temporal hint also reads the line
first, so the memory system moves 4/3 of that.

Written for project-benchmarks-jax#47, for project-compute: whether a workload's GB/s comes close
to what the part's memory channels deliver, at one thread and at all of them.
"""
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import numba
import numpy as np

out = sys.argv[1]
n = 2 ** (int(sys.argv[2]) if len(sys.argv) > 2 else 28)
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
S = 3.0


@numba.njit(parallel=True)
def init(a, b, c):
    for i in numba.prange(a.size):
        a[i] = 0.0
        b[i] = 1.0
        c[i] = 2.0


@numba.njit(parallel=True)
def triad(a, b, c, s):
    for i in numba.prange(a.size):
        a[i] = b[i] + s * c[i]


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


a, b, c = np.empty(n), np.empty(n), np.empty(n)
init(a, b, c)
most = numba.config.NUMBA_NUM_THREADS
counts = sorted({k for k in (1, 2, 4, 8, 16) if k <= most} | {most})
try:
    usable = len(os.sched_getaffinity(0))
except (AttributeError, OSError):
    usable = None
with open(out, "a") as f:
    for k in counts:
        numba.set_num_threads(k)
        triad(a, b, c, S)  # warm-up, and the compile on the first count
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            triad(a, b, c, S)
            times.append(time.perf_counter() - t0)
        assert a[:: n // 64].tolist() == [1.0 + S * 2.0] * 64
        nbytes = 24 * n
        row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "probe": "triad",
               "threads": k, "elements": n, "array_bytes": a.nbytes, "bytes": nbytes,
               "best_s": round(min(times), 4), "median_s": round(statistics.median(times), 4),
               "gb_per_s": round(nbytes / min(times) / 1e9, 2),
               "times_s": [round(x, 4) for x in times],
               "env": {"host": platform.node(), "machine": platform.machine(),
                       "python": platform.python_version(), "numpy": np.__version__,
                       "numba": numba.__version__, "threading_layer": numba.threading_layer(),
                       "numba_num_threads": most, "cpus_usable": usable,
                       "thp": read("/sys/kernel/mm/transparent_hugepage/enabled")}}
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"triad {k:2} threads  best {row['best_s']:7.4f} s  median {row['median_s']:7.4f} s"
              f"  {row['gb_per_s']:7.2f} GB/s", flush=True)
