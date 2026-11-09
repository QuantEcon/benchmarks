# The M6 triad's thread check (2026-11-09): CPU seconds per wall second at 1 and 12 threads, run as
#   NUMBA_NUM_THREADS=12 .venv/bin/python check_cpu_wall.py
# under /usr/bin/time -l from ~/bench/benchmarks, after stream_triad.py with NUMBA_NUM_THREADS=1.
import numba, numpy as np, time
@numba.njit(parallel=True)
def triad(a,b,c,s):
    for i in numba.prange(a.size): a[i]=b[i]+s*c[i]
n=2**28; a=np.zeros(n); b=np.ones(n); c=np.full(n,2.0)
triad(a,b,c,3.0)
for k in (1,12):
    numba.set_num_threads(k); t=time.process_time(); w=time.perf_counter()
    for _ in range(10): triad(a,b,c,3.0)
    print(k,'threads: cpu/wall', round((time.process_time()-t)/(time.perf_counter()-w),2))
