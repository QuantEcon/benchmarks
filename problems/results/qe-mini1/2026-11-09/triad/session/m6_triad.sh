#!/bin/bash
# project-benchmarks-jax#47's STREAM-style triad (stream_triad.py, from the #47 session), on
# qe-mini1's M6 at 1 thread up to all 12 cores, for project-compute (Matt, 2026-11-09, by way of the
# #47 session: "Please advise organiser they can proceed"). Run from the runs clone (~/bench/benchmarks)
# with its venv (numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, the problems/constraints.txt pins at
# 1874f1c). The long-running Claude sessions stay running, in a quiet window agreed with them, logged
# in at the screen. state.txt samples the machine every 5 s.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=~/bench/triad47-$(date -u +%F)-qe-mini1
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
}
sample() { while sleep 5; do state sample >> "$OUT/state.txt"; done; }
{ sw_vers; echo "console: $(stat -f %Su /dev/console)"
  sysctl hw.ncpu hw.perflevel0.name hw.perflevel0.physicalcpu hw.perflevel1.name hw.perflevel1.physicalcpu \
    hw.perflevel2.name hw.perflevel2.physicalcpu hw.memsize 2>/dev/null
  shasum -a 256 "$HERE/stream_triad.py"; state before; } > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
cd ~/bench/benchmarks
NUMBA_NUM_THREADS=$(sysctl -n hw.ncpu) .venv/bin/python "$HERE/stream_triad.py" "$OUT/triad.jsonl" 2>&1 | tee "$OUT/run.log"
echo "== exit ${PIPESTATUS[0]} $(date -u +%T)" | tee -a "$OUT/run.log"
