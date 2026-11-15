#!/bin/bash
# m5_ynn4.sh: QuantEcon/project-benchmarks-jax#4 on qe-wharton2's M5 Pro (Matt, 2026-11-15: "Yes please").
# Part 1 of the M6's precision/runs/2026-11-09/qe-mini1/mac-float64/session/m6_window.sh,
# unchanged but for the host: the 2026-10-25 ynn_dot_check.py and #4's reproducer, each in float32, float64 and
# float32 with YNNPACK's fusion cleared; then run_matrix.py's kalman and matmul at 1,000 and 10,000, with and
# without it, from a worktree at 5fbf033, the M6's runs commit. #48's probe already ran here on 2026-11-06.
# About 2 minutes. In screen `ynn4` under caffeinate; state.txt samples the machine every 5 s, and its ps lines
# are redacted with scratch/redact_ps.py before anything is published.
set -u
export PATH="$HOME/bench/tools/env/bin:$HOME/.local/bin:$PATH"   # the real git first: /usr/bin/git is the Xcode stub
D=~/bench/ynn4                      # this script, repro4.py and ynn_dot_check.py, copied from main
R=~/bench/benchmarks                # the runs clone and its venv
PY=$R/.venv/bin/python
WT=~/bench/ynn4/src-5fbf033         # run_matrix.py as the M6 ran it
OUT=$D/out-$(date -u +%F)
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
}
{ sw_vers; sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize; pmset -g batt | head -2
  pmset -g | grep -i -E 'lowpowermode|powermode'; state before; } > "$OUT/state.txt"
log() { echo "== $* $(date -u +%T)" | tee -a "$OUT/run.log"; }
stop() { log "stopped: $*"; echo stopped > "$D/ynn4.done"; exit 1; }

# Stop before anything runs if the Mac is on battery or in Low Power Mode, or if econ-ai's lock is held.
pmset -g batt | grep -q "AC Power" || stop "not on AC power"
pmset -g | grep -i lowpowermode | grep -q ' 1' && stop "Low Power Mode is on"
[ -d ~/econ-ai/.lock-machine ] && stop "econ-ai's lock is held"

log "setup: worktree at 5fbf033"
if [ ! -d "$WT" ]; then
  git -C "$R" cat-file -e 5fbf033^{commit} 2>/dev/null || git -C "$R" fetch -q origin
  git -C "$R" worktree add -q --detach "$WT" 5fbf033 >> "$OUT/run.log" 2>&1 || stop "worktree failed"
fi
{ git -C "$WT" log -1 --format='run_matrix at %h: %s'; shasum -a 256 "$D"/*.py "$D"/m5_ynn4.sh
  "$PY" -c 'import sys, jax, numpy, scipy; print("python", sys.version.split()[0], "jax", jax.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__)'
} >> "$OUT/run.log" 2>&1

sample() { while sleep 5; do state sample >> "$OUT/state.txt"; done; }
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
NOYNN=--xla_cpu_experimental_ynn_fusion_type=

C=$D/ynn_dot_check.py
log "ynn_dot_check, default"
for d in float32 float64; do "$PY" "$C" $d >> "$OUT/ynn_dot_check.txt" 2>> "$OUT/run.log"; done
log "ynn_dot_check, fusion cleared, float32 three times"
for i in 1 2 3; do XLA_FLAGS=$NOYNN "$PY" "$C" float32 >> "$OUT/ynn_dot_check.txt" 2>> "$OUT/run.log"; done

log "#4's reproducer"
for d in float32 float64; do JAX_PLATFORMS=cpu "$PY" "$D/repro4.py" $d >> "$OUT/repro4.txt" 2>> "$OUT/run.log"; done
echo "with XLA_FLAGS=$NOYNN:" >> "$OUT/repro4.txt"
JAX_PLATFORMS=cpu XLA_FLAGS=$NOYNN "$PY" "$D/repro4.py" float32 >> "$OUT/repro4.txt" 2>> "$OUT/run.log"

for label in default ynn-cleared; do
  log "run_matrix, kalman and matmul, $label"
  if [ $label = default ]; then F=""; else F=$NOYNN; fi
  (cd "$WT" && XLA_FLAGS=$F "$PY" precision/run_matrix.py --workloads kalman,matmul --sizes 1000,10000 \
    --dtypes float32,float64 --devices cpu --host qe-wharton2 --label "ynn4-$label" --out "$OUT/matrix-$label") \
    >> "$OUT/matrix.txt" 2>> "$OUT/run.log"
  echo "== exit $? ($label)" >> "$OUT/matrix.txt"
done
log "done"
echo done > "$D/ynn4.done"
