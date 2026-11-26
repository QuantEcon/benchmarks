#!/bin/bash
# QuantEcon/benchmarks#58 on qe-wharton2's M5 Pro (Matt, 2026-11-11: "Both precisions"): the same
# four parts as the M6's run of 2026-11-10 (m6_kalman58.sh), on the four CPU stacks in both precisions.
# From the runs checkout at 62df721 and its venv, as the M5 Pro's suite ran on 2026-11-06. A guest
# machine, so nothing is paused; launched in screen under caffeinate -ims. Everything it writes goes in
# a .noindex folder. state.txt has the machine every 30 s, with no user column and no console user's
# name; samples.txt the media and Spotlight processes every 2 s.
#   quick  the three new sweeps at 10 filters, as a check, kept apart in quick/
#   c      `batch` and `state` at their usual sizes, all four stacks, 900 s a cell
#   a      the new sweeps at 1 to 300 filters, all four stacks, 1,800 s a cell
#   b      the new sweeps at 1,000 to 10,000 filters, without serial Numba, 1,800 s a cell
set -u
export PATH=$HOME/bench/tools/env/bin:$PATH
R=~/bench/benchmarks
PY=$R/.venv/bin/python
OUT=~/bench/kalman58-$(date -u +%F)-qe-wharton2.noindex
mkdir -p "$OUT/session"
cp "$0" "$OUT/session/"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
  pmset -g therm | grep -v '^Note'
  pmset -g batt | head -2 | tail -1
}
D='mediaanalysisd|photoanalysisd|ANECompilerService|mds_stores|mdworker|/mds$|mdbulkimport|corespotlightd'
sample() {
  local k=0
  while sleep 2; do
    ps -Ao pcpu=,comm= | grep -E "$D" | awk -v t="$(date -u +%T)" '$1 >= 1 {sub(".*/", "", $2); printf "%s %s %s\n", t, $1, $2}' >> "$OUT/samples.txt"
    k=$((k + 1)); [ $((k % 15)) -eq 0 ] && state sample >> "$OUT/state.txt"
  done
}
log() { echo "== $* $(date -u +%T)" | tee -a "$OUT/run.log" >> "$OUT/samples.txt"; }
{ sw_vers
  if [ "$(stat -f %Su /dev/console)" = "$(id -un)" ]; then echo "console: this account"; else echo "console: another account"; fi
  state before; } > "$OUT/state.txt"
{ cd "$R" && git log -1 --format='benchmarks at %h: %s' && git status --porcelain -- problems ':!problems/results'
  "$PY" -c "import numpy, numba, jax; print('numpy', numpy.__version__, '| numba', numba.__version__, '| jax', jax.__version__)"
  sysctl hw.ncpu hw.perflevel0.name hw.perflevel0.physicalcpu hw.perflevel1.name hw.perflevel1.physicalcpu hw.memsize 2>/dev/null
  shasum -a 256 "$0"; } > "$OUT/setup.txt" 2>&1
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
cd "$R"
NEW=batch20,batch50,batch100
ALL=numpy,numba,numba-parallel,jax-cpu
suite() {  # suite <part> <arguments...>
  local part=$1; shift
  log "$part start"
  "$PY" problems/run_suite.py --problems kalman --host qe-wharton2 "$@" >> "$OUT/run.log" 2>&1
  log "$part done, exit $?"
}
suite quick --quick --sweeps $NEW --backends $ALL --label kalman-grid-quick --out "$OUT/quick"
suite c --sweeps batch,state --backends $ALL --label kalman-grid --out "$OUT/results"
suite a --sweeps $NEW --sizes 1,10,100,300 --backends $ALL --timeout 1800 --label kalman-grid --out "$OUT/results"
suite b --sweeps $NEW --sizes 1000,3000,10000 --backends numpy,numba-parallel,jax-cpu --timeout 1800 --label kalman-grid --out "$OUT/results"
log done
