#!/bin/bash
# On qe-wharton1-gpu, under project-econ-on-ai-hardware's machine lock, which the caller has taken and
# this releases on exit (Q27 and Q28, Matt, 2026-11-21): the one cell of the 2026-11-21 Phase 2 run that passed
# the 900 s limit, P6's `numpy` at a grid of 50,000 in float32, again with a limit of 1,800 s, labelled
# timeout-1800, into the same run folder's results.
# Launch: ssh qe-wharton1-gpu 'mkdir econ-ai/.lock-machine && nohup setsid bash ~/bench/wharton1_p2_rerun.sh \
#   > ~/bench/wharton1_p2_rerun.out 2>&1 < /dev/null &'
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
R=~/bench/run-2026-11-21-p2
state() {
  echo "== $1: $(date "+%F %T %Z") ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,stat,pcpu,comm --sort=-pcpu | head -8
}
sample() { while sleep 300; do state sample >> $R/state-rerun.txt; done; }
state before > $R/state-rerun.txt
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> $R/state-rerun.txt; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> $R/state-rerun.txt' EXIT

cd ~/bench/benchmarks
echo "== start $(date -u +%FT%TZ) at $(git log -1 --format=%h)" >> $R/state-rerun.txt
.venv/bin/python problems/run_suite.py --problems time_iter --sweeps grid --sizes 50000 --backends numpy \
  --dtypes float32 --timeout 1800 --host qe-wharton1-gpu --label timeout-1800 \
  --out $R/results > $R/rerun.log 2>&1
s=$?; echo "== rerun done $(date -u +%T), exit $s" >> $R/state-rerun.txt

# Q28 (Matt, 2026-11-21): is the card's float64 compile, which grows with the observations to 22 s at
# 10,000,000, XLA's autotuner? The same two cells with it switched off, labelled autotune0.
echo "== autotune0 start $(date -u +%T)" >> $R/state-rerun.txt
XLA_FLAGS=--xla_gpu_autotune_level=0 .venv/bin/python problems/run_suite.py --problems mle,mle_hand \
  --sweeps obs --sizes 10000000 --backends jax-gpu --dtypes float64 --gpus 0 --host qe-wharton1-gpu \
  --label autotune0 --out $R/results > $R/autotune0.log 2>&1
s=$?; echo "== autotune0 done $(date -u +%T), exit $s" >> $R/state-rerun.txt
echo ALLDONE >> $R/state-rerun.txt
