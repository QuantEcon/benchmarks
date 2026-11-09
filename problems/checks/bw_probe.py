"""
One core's memory bandwidth, and P1's two serial Bellman steps in miniature, on P1's own stored
rewards at a wealth grid of 1,500 (1,500 x 100 x 1,500 float64, 1.8 GB). Run it from a
QuantEcon/benchmarks clone's root, on one CPU, with every thread pool at one:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        taskset -c 0 python bw_probe.py <out.jsonl> [reps] [size]

(macOS has no taskset; run it without.) Each probe runs once to warm up, then `reps` times
(default 3). Each output line is one probe: the best and median warm time, the bytes it must move
at least, and the GB/s that implies at the best time.

    read        u.sum(): one vectorised pass over u, the most one core reads
    copy        np.copyto(t, u), into a buffer made once: read u, write t
    numpy       np.max(u + b, axis=2), as P1's numpy stack: a fresh 1.8 GB temporary each time
    numpy-out   the same into a temporary made once, so without the allocation
    numba       P1's serial numba Bellman step, as in problems/vfi.py: a running max over u,
                with EV read across its rows
    numba-evT   the same with EV transposed, so EV is read along a row

Written for project-benchmarks-jax#47, to test whether P1's serial cells are limited by one core's
bandwidth: compare numba with read, and numpy with numpy-out.
"""
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, "problems")
import vfi  # noqa: E402

out = sys.argv[1]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
size = int(sys.argv[3]) if len(sys.argv) > 3 else 1500


def read(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def cpu0_mhz():
    for line in open("/proc/cpuinfo") if os.path.exists("/proc/cpuinfo") else []:
        if line.startswith("cpu MHz"):
            return round(float(line.split(":")[1]))
    return None


m = vfi.model(None, size, np.float64)
u = vfi._rewards(np, m["w"], m["y"])
EV = np.random.default_rng(0).standard_normal((size, vfi.Y_SIZE))
b = vfi.BETA * EV.T[None, :, :]
t = np.empty_like(u)
v_new = np.empty((size, vfi.Y_SIZE))

import numba  # noqa: E402

beta = vfi.BETA


@numba.njit
def bellman(u, EV, v_new):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            best = -np.inf
            for ip in range(u.shape[2]):
                best = max(best, u[i, j, ip] + beta * EV[ip, j])
            v_new[i, j] = best


@numba.njit
def bellman_evT(u, EVT, v_new):
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            best = -np.inf
            for ip in range(u.shape[2]):
                best = max(best, u[i, j, ip] + beta * EVT[j, ip])
            v_new[i, j] = best


EVT = np.ascontiguousarray(EV.T)
n = u.nbytes
PROBES = [
    ("read", lambda: u.sum(), n),
    ("copy", lambda: np.copyto(t, u), 2 * n),
    ("numpy", lambda: np.max(u + b, axis=2), 3 * n),
    ("numpy-out", lambda: np.add(u, b, out=t).max(axis=2), 3 * n),
    ("numba", lambda: bellman(u, EV, v_new), n),
    ("numba-evT", lambda: bellman_evT(u, EVT, v_new), n),
]

try:
    affinity = sorted(os.sched_getaffinity(0))
except (AttributeError, OSError):
    affinity = None
env = {"host": platform.node(), "machine": platform.machine(), "python": platform.python_version(),
       "numpy": np.__version__, "numba": numba.__version__, "cpus_usable": affinity,
       "thp": read("/sys/kernel/mm/transparent_hugepage/enabled"),
       "threads": {k: os.environ.get(k) for k in
                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS")}}
with open(out, "a") as f:
    for name, fn, nbytes in PROBES:
        fn()  # warm-up, and numba's compile
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        row = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "probe": name,
               "size": size, "bytes": nbytes, "best_s": round(min(times), 4),
               "median_s": round(statistics.median(times), 4),
               "gb_per_s": round(nbytes / min(times) / 1e9, 2), "cpu0_mhz": cpu0_mhz(),
               "times_s": [round(x, 4) for x in times], "env": env}
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(f"{name:10} best {row['best_s']:8.4f} s  median {row['median_s']:8.4f} s  "
              f"{row['gb_per_s']:7.2f} GB/s  cpu0 {row['cpu0_mhz']} MHz", flush=True)
