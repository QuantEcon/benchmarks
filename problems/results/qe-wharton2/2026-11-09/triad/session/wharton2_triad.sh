#!/bin/bash
# project-benchmarks-jax#47's STREAM-style triad (stream_triad.py, from the #47 session), on
# qe-wharton2's M5 Pro at 1 thread up to all 18 cores, for project-compute (Matt, 2026-11-09, by way
# of the #47 session; econ-on-ai-hardware's OK). Run from ~/bench/benchmarks with its venv (numpy
# 2.5.4, numba 0.68.1, llvmlite 0.50.0, the 1874f1c pins). Launched in screen under caffeinate -i -s,
# as the bandwidth probe was. Tools come from ~/bench/tools/env/bin first, so the Xcode git stub never
# runs. state.txt samples the machine every 5 s, with no user column and no console user's name.
set -u
export PATH=$HOME/bench/tools/env/bin:$PATH
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HERE/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
}
sample() { while sleep 5; do state sample >> "$OUT/state.txt"; done; }
{ sw_vers; pmset -g batt | head -2; pmset -g | grep -i powermode
  if [ "$(stat -f %Su /dev/console)" = "$(id -un)" ]; then echo "console: this account"; else echo "console: another account"; fi
  sysctl hw.ncpu hw.perflevel0.name hw.perflevel0.physicalcpu hw.perflevel1.name hw.perflevel1.physicalcpu \
    hw.perflevel2.name hw.perflevel2.physicalcpu hw.memsize 2>/dev/null
  shasum -a 256 "$HERE/stream_triad.py"; state before; } > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"; touch "$HERE/done"' EXIT
cd ~/bench/benchmarks
NUMBA_NUM_THREADS=$(sysctl -n hw.ncpu) .venv/bin/python "$HERE/stream_triad.py" "$OUT/triad.jsonl" 2>&1 | tee "$OUT/run.log"
echo "== exit ${PIPESTATUS[0]} $(date -u +%T)" | tee -a "$OUT/run.log"
# The same check as on the M6: the one-thread point with Numba's pool itself at one thread.
NUMBA_NUM_THREADS=1 .venv/bin/python "$HERE/stream_triad.py" "$OUT/check-pool1.jsonl" 2>&1 | tee "$OUT/check-pool1.log"
echo "== exit ${PIPESTATUS[0]} $(date -u +%T)" | tee -a "$OUT/check-pool1.log"
