#!/bin/bash
# project-benchmarks-jax#47's one-core probe (bw_probe.py, from the laptop session, 4,615 bytes,
# sha256 3ab1785b2331ed8c…), on qe-mini1's M6 for comparison with the three AWS instances
# (Matt approved, 2026-11-09). Run from the runs clone (~/bench/benchmarks at 5fbf033, whose
# problems/vfi.py is the same file as at 1874f1c) with its venv, which matches problems/constraints.txt
# at 1874f1c (numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0). Every thread pool at one; macOS has no
# taskset. The long-running Claude sessions stay running, in a quiet window agreed with them, logged
# in at the screen. state.txt samples the machine every 5 s.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=~/bench/bw47-$(date -u +%F)-qe-mini1
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
}
sample() { while sleep 5; do state sample >> "$OUT/state.txt"; done; }
{ sw_vers; echo "console: $(stat -f %Su /dev/console)"; shasum -a 256 "$HERE/bw_probe.py"; state before; } > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
cd ~/bench/benchmarks
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python "$HERE/bw_probe.py" "$OUT/bandwidth.jsonl" 2>&1 | tee "$OUT/run.log"
echo "== exit ${PIPESTATUS[0]} $(date -u +%T)" | tee -a "$OUT/run.log"
