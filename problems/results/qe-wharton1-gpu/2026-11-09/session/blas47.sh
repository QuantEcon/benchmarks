#!/bin/bash
# QuantEcon/project-benchmarks-jax#47: does OpenBLAS's thread pool hold back parallel Numba on x86?
# The suite caps no thread pool, so P1's per-iteration v @ Q.T runs on OpenBLAS's threads, which
# spin after each call while Numba's own threads run the Bellman step. On #47's AWS instances
# parallel Numba was 4.1-4.6x serial on 16 cores; on the Macs, whose NumPy uses Accelerate, 5.3x
# and 8.8x. Here, on qe-wharton1-gpu's Threadripper 2990WX (Matt, 2026-11-09: "Yes, let's run that
# test"): P1 (vfi) at 1,500 in float64, numba-parallel and numba, with OpenBLAS as the suite left
# it and with OPENBLAS_NUM_THREADS=1, at the default placement and on node 0 (16 CPUs with its own
# memory, as a 16-CPU slice is). numba-parallel runs twice in each setting, alternating, so drift
# falls on both alike. From the run checkout as it stands (its problems/ harness is main's), into
# this run's own folder. Run in tmux under econ-ai's machine lock, which the caller takes and this
# releases on exit. clock_sampler.py (main at f427574) samples each CPU every 2 s.
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
B=~/bench/benchmarks
R=$HERE/run-$(date -u +%Y%m%dT%H%M%SZ)
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
  echo "node 0: $(numactl --cpunodebind=0 --membind=0 numactl --show | tr '\n' ' ')"
} > $R/setup.txt 2>&1

cell() {  # cell <label> <numactl arguments> <backends> <OPENBLAS_NUM_THREADS or empty>
  local label=$1 args=$2 be=$3 nt=$4
  log "$label $be start"
  env ${nt:+OPENBLAS_NUM_THREADS=$nt} ${args:+numactl $args} .venv/bin/python problems/run_suite.py \
    --problems vfi --sweeps grid --sizes 1500 --dtypes float64 --backends $be \
    --label $label --host qe-wharton1-gpu --out $R/rows --timeout 1800 >> $R/run.log 2>&1
  log "$label $be done, exit $?"
}

N0="--cpunodebind=0 --membind=0"
for rep in 1 2; do
  cell blas-default-numa-default "" numba-parallel ""
  cell blas1-numa-default "" numba-parallel 1
  cell blas-default-numa-one-node "$N0" numba-parallel ""
  cell blas1-numa-one-node "$N0" numba-parallel 1
done
cell blas-default-numa-default "" numba ""
cell blas1-numa-default "" numba 1
STATUS=0
log ALLDONE
