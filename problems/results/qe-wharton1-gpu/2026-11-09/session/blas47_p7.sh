#!/bin/bash
# QuantEcon/project-benchmarks-jax#47, the second half of the OpenBLAS check (blas47.sh did P1):
# P7's batch of 10,000 independent Kalman filters in float64, whose Numba filter calls @,
# np.linalg.solve and slogdet inside the prange loop, so each of Numba's threads calls into
# OpenBLAS. On #47's AWS instances parallel Numba there was 0.8-1.2x serial, and on the
# Threadripper 0.33x, against 6.2x and 9.4x on the Macs, whose NumPy and SciPy use Accelerate.
# numba-parallel and numba, each with OpenBLAS as the suite left it and with
# OPENBLAS_NUM_THREADS=1, at the default placement. Otherwise as blas47.sh: from the run checkout
# as it stands, into this run's own folder, in tmux under econ-ai's machine lock, which the
# caller takes and this releases on exit, with clock_sampler.py every 2 s.
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
B=~/bench/benchmarks
R=$HERE/p7-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $R
log() { echo "== $* $(date -u +%T)" | tee -a $R/run.log; }

state() {
  echo "== $1: $(date "+%F %T %Z") ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  ps -eo pid,user,stat,pcpu,comm --sort=-pcpu | head -8
  numactl --hardware | grep free
}
state before > $R/state.txt
python3 $HERE/clock_sampler.py $R/clock.jsonl $R/run.log 2 < /dev/null > /dev/null 2>&1 &
SAMPLER=$!
trap 'state after >> $R/state.txt; echo "EXIT ${STATUS:-1}" >> $R/run.log; sleep 3; kill $SAMPLER 2>/dev/null; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> $R/state.txt' EXIT
cd $B

{ git log -1 --format='benchmarks at %h: %s'; git status --porcelain
  git diff --stat HEAD origin/main -- problems/vfi.py problems/run_suite.py problems/run_cell.py problems/spec.py problems/constraints.txt
  echo "(no lines above: the harness matches main)"
  sha256sum $HERE/clock_sampler.py $0
  .venv/bin/python -c "import numba, numpy, scipy; print('numba', numba.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__); print('threading layer default:', numba.config.THREADING_LAYER)"
  numactl --hardware; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|NUMA'
  echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled); numa_balancing: $(cat /proc/sys/kernel/numa_balancing)"
  echo "cpufreq: governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null), boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
} > $R/setup.txt 2>&1

cell() {  # cell <label> <backends> <OPENBLAS_NUM_THREADS or empty>
  local label=$1 be=$2 nt=$3
  log "$label $be start"
  env ${nt:+OPENBLAS_NUM_THREADS=$nt} .venv/bin/python problems/run_suite.py \
    --problems kalman --sweeps batch --sizes 10000 --dtypes float64 --backends $be \
    --label $label --host qe-wharton1-gpu --out $R/rows --timeout 1800 >> $R/run.log 2>&1
  log "$label $be done, exit $?"
}

cell blas1-numa-default numba-parallel 1
cell blas-default-numa-default numba-parallel ""
cell blas1-numa-default numba 1
cell blas-default-numa-default numba ""
STATUS=0
log ALLDONE
