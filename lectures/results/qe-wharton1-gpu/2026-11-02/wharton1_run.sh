#!/bin/bash
# On qe-wharton1-gpu, after project-econ-on-ai-hardware's S(K) set, under its machine lock, which
# the caller has taken and this releases on exit:
#  1. QuantEcon/benchmarks#5's problem-type suite at 5fbf033, every size, on the CPU backends and on
#     the TITAN RTX at 0b:00.0;
#  2. QuantEcon/benchmarks#4: lecture-python.myst's 33 JAX lectures at 087518ec0, as written, three
#     times, on the card and then the Threadripper, lecture by lecture, two hours' cap each, as on
#     the T4 runner.
# state.txt samples the machine every ten minutes. Set up by wharton1_setup.sh.
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
D=$(date -u +%F)
R=~/bench/run-$D
mkdir -p $R
state() {
  echo "== $1: $(date "+%F %T %Z") ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,stat,pcpu,comm --sort=-pcpu | head -8
}
sample() { while sleep 600; do state sample >> $R/state.txt; done; }
state before > $R/state.txt
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> $R/state.txt; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> $R/state.txt' EXIT
echo "== start $(date -u +%FT%TZ) at $(git -C ~/bench/benchmarks log -1 --format=%h)" >> $R/state.txt

cd ~/bench/benchmarks
echo "== problems start $(date -u +%T)" >> $R/state.txt
.venv/bin/python problems/run_suite.py --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu \
  --gpus 0 --host qe-wharton1-gpu > $R/problems.log 2>&1
s=$?; echo "== problems done $(date -u +%T), exit $s" >> $R/state.txt

L=~/bench/lectures-python
O=$L/results/qe-wharton1-gpu/$D/lecture-python
mkdir -p $O
cp $L/environment.txt $O/environment.txt
echo "== lectures start $(date -u +%T); lecture-python.myst at 087518ec0" >> $R/state.txt
cd $L && ROOT=$L CAP=7200 ./lectures_set.sh $O 3 cuda cpu > $O/lectures-set.log 2>&1
s=$?; echo "== lectures done $(date -u +%T), exit $s" >> $R/state.txt
find $O -name '*.err' -empty -delete
echo ALLDONE >> $R/state.txt
