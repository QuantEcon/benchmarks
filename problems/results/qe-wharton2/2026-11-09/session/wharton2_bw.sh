#!/bin/bash
# project-benchmarks-jax#47's one-core probe (bw_probe.py, 4,615 bytes, sha256 3ab1785b2331ed8c…) on
# qe-wharton2's M5 Pro, for comparison with the three AWS instances and the M6 (Matt, 2026-11-09:
# "Please proceed"). Run from ~/bench/benchmarks (4ed7c58, whose problems/vfi.py is the same file as
# at 1874f1c) with its venv (numpy 2.5.4, numba 0.68.1, llvmlite 0.50.0, the 1874f1c pins), every
# thread pool at one; macOS has no taskset. Launched in screen under caffeinate -i -s, as the M5 Pro
# run of 2026-11-06 was. Tools come from ~/bench/tools/env/bin first, so the Xcode git stub never
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
  shasum -a 256 "$HERE/bw_probe.py"; state before; } > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"; touch "$HERE/done"' EXIT
cd ~/bench/benchmarks
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  .venv/bin/python "$HERE/bw_probe.py" "$OUT/bandwidth.jsonl" 2>&1 | tee "$OUT/run.log"
echo "== exit ${PIPESTATUS[0]} $(date -u +%T)" | tee -a "$OUT/run.log"
