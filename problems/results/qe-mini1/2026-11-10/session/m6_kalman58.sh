#!/bin/bash
# QuantEcon/benchmarks#58's M6 half on qe-mini1, macOS 27.2.1 (26B131): P7's batch sweeps at 20, 50
# and 100 states, and `batch` and `state` again, on the four CPU stacks in both precisions, in a quiet
# window agreed with the sessions here, which stay running (Matt, 2026-11-10: "Agree", "Yes please").
# From the runs checkout at #60's merge commit and its venv. Everything this writes goes in a
# .noindex folder, so Spotlight and media analysis leave it alone (precision/runs/2026-11-10/qe-mini1/
# noindex/). In order, as #58 gives it:
#   quick  the three new sweeps at 10 filters, as a check, kept apart in quick/
#   c      `batch` and `state` at their usual sizes, all four stacks, 900 s a cell
#   a      the new sweeps at 1 to 300 filters, all four stacks, 1,800 s a cell
#   b      the new sweeps at 1,000 to 10,000 filters, without serial Numba, 1,800 s a cell
# state.txt has the machine every 30 s; samples.txt the media and Spotlight processes every 2 s.
set -u
R=~/bench/benchmarks
PY=$R/.venv/bin/python
OUT=~/bench/kalman58-$(date -u +%F)-qe-mini1.noindex
mkdir -p "$OUT/session"
cp "$0" "$OUT/session/"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC), console $(stat -f %Su /dev/console)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
  pmset -g therm | grep -v '^Note'
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
{ sw_vers; state before; } > "$OUT/state.txt"
{ cd "$R" && git log -1 --format='benchmarks at %h: %s' && git status --porcelain -- problems ':!problems/results'
  "$PY" -c "import numpy, numba, jax; print('numpy', numpy.__version__, '| numba', numba.__version__, '| jax', jax.__version__)"
  shasum -a 256 "$0"; } > "$OUT/setup.txt" 2>&1
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
cd "$R"
NEW=batch20,batch50,batch100
ALL=numpy,numba,numba-parallel,jax-cpu
suite() {  # suite <part> <arguments...>
  local part=$1; shift
  log "$part start"
  "$PY" problems/run_suite.py --problems kalman --host qe-mini1 "$@" >> "$OUT/run.log" 2>&1
  log "$part done, exit $?"
}
suite quick --quick --sweeps $NEW --backends $ALL --label kalman-grid-quick --out "$OUT/quick"
suite c --sweeps batch,state --backends $ALL --label kalman-grid --out "$OUT/results"
suite a --sweeps $NEW --sizes 1,10,100,300 --backends $ALL --timeout 1800 --label kalman-grid --out "$OUT/results"
suite b --sweeps $NEW --sizes 1000,3000,10000 --backends numpy,numba-parallel,jax-cpu --timeout 1800 --label kalman-grid --out "$OUT/results"
log done
