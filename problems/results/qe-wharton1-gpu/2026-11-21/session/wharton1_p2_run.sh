#!/bin/bash
# On qe-wharton1-gpu, after project-econ-on-ai-hardware's Threadripper run, under its machine lock,
# which the caller has taken and this releases on exit (Q25, 2026-11-21):
#  1. QuantEcon/benchmarks#5's Phase 2 problem types, every size, on the four CPU backends (the
#     2990WX) and on the TITAN RTX at 0b:00.0, at ~/bench/benchmarks's checkout;
#  2. the two checks problems/README.md asks of P8's first Linux and GPU run: `mle_blocked`'s
#     `params` sweep again with OpenBLAS held to one thread, labelled, which shows whether its BLAS
#     calls inside `prange` oversubscribe the cores; and hoist_check.py on the TITAN, which shows
#     whether XLA computes autodiff's X·I product once, outside the Newton loop, as on the CPU.
# state.txt samples the machine every ten minutes.
# Launch: ssh qe-wharton1-gpu 'mkdir econ-ai/.lock-machine && nohup setsid bash ~/bench/wharton1_p2_run.sh \
#   > ~/bench/wharton1_p2_run.out 2>&1 < /dev/null &'
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
PROBLEMS=stationary,hpi,time_iter,mle,mle_hand,mle_onepass,mle_blocked
D=$(date -u +%F)
R=~/bench/run-$D-p2
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

cd ~/bench/benchmarks
echo "== start $(date -u +%FT%TZ) at $(git log -1 --format=%h)" >> $R/state.txt
echo "== problems start $(date -u +%T): $PROBLEMS" >> $R/state.txt
.venv/bin/python problems/run_suite.py --problems $PROBLEMS \
  --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu --gpus 0 --host qe-wharton1-gpu \
  --out $R/results > $R/problems.log 2>&1
s=$?; echo "== problems done $(date -u +%T), exit $s" >> $R/state.txt

echo "== OpenBLAS check start $(date -u +%T)" >> $R/state.txt
OPENBLAS_NUM_THREADS=1 .venv/bin/python problems/run_suite.py --problems mle_blocked --sweeps params \
  --backends numba,numba-parallel --host qe-wharton1-gpu --label openblas-1-thread \
  --out $R/results > $R/openblas.log 2>&1
s=$?; echo "== OpenBLAS check done $(date -u +%T), exit $s" >> $R/state.txt

echo "== hoisting check start $(date -u +%T)" >> $R/state.txt
for k in 10 300; do
  JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false .venv/bin/python ~/bench/hoist_check.py problems $k >> $R/hoist.txt 2>&1
done
echo "== hoisting check done $(date -u +%T)" >> $R/state.txt
echo ALLDONE >> $R/state.txt
