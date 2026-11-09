#!/bin/bash
# qe-mini1's M6 on macOS 27.2 (26B123), 2026-11-09, in a quiet window agreed with the sessions here,
# which stay running (Matt, 2026-11-09: "Agree, please run"). Two parts, as on macOS 27.0.1 before:
#  1. QuantEcon/project-benchmarks-jax#4: QuantEcon/benchmarks' precision/runs/2026-10-25/checks/
#     ynn_dot_check.py and #4's own reproducer, each in float32, float64 and float32 with YNNPACK's
#     fusion cleared; then the harness's kalman and matmul cells at 1,000 and 10,000, with and without it.
#  2. #48's probe, unchanged on main (ba58764) since 3806db6, three full runs, as on 2026-11-06.
# state.txt samples the machine every 5 s. An optional label keeps a rerun's output apart: the first
# run, at 01:26 UTC, met audiomxd and configd at about a core between them (no console user after the
# upgrade's reboot), so it was run again as `m6_window.sh rerun` once Matt had logged in.
set -u
L=${1:+-$1}
B=~/work/quantecon/benchmarks      # main, for the 2026-10-25 checks and the probe
R=~/bench/benchmarks               # the runs clone and its venv, for run_matrix.py
PY=$R/.venv/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=~/bench/ynn4-$(date -u +%F)-qe-mini1$L
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
}
sample() { while sleep 5; do state sample >> "$OUT/state.txt"; done; }
{ sw_vers; state before; } > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
log() { echo "== $* $(date -u +%T)" | tee -a "$OUT/run.log"; }
NOYNN=--xla_cpu_experimental_ynn_fusion_type=

C=$B/precision/runs/2026-10-25/checks/ynn_dot_check.py
log "ynn_dot_check, default"
for d in float32 float64; do "$PY" "$C" $d >> "$OUT/ynn_dot_check.txt" 2>> "$OUT/run.log"; done
log "ynn_dot_check, fusion cleared, float32 three times"
for i in 1 2 3; do XLA_FLAGS=$NOYNN "$PY" "$C" float32 >> "$OUT/ynn_dot_check.txt" 2>> "$OUT/run.log"; done

log "#4's reproducer"
for d in float32 float64; do JAX_PLATFORMS=cpu "$PY" "$HERE/repro4.py" $d >> "$OUT/repro4.txt" 2>> "$OUT/run.log"; done
echo "with XLA_FLAGS=$NOYNN:" >> "$OUT/repro4.txt"
JAX_PLATFORMS=cpu XLA_FLAGS=$NOYNN "$PY" "$HERE/repro4.py" float32 >> "$OUT/repro4.txt" 2>> "$OUT/run.log"

for label in default ynn-cleared; do
  log "run_matrix, kalman and matmul, $label"
  if [ $label = default ]; then F=""; else F=$NOYNN; fi
  (cd "$R" && XLA_FLAGS=$F "$PY" precision/run_matrix.py --workloads kalman,matmul --sizes 1000,10000 \
    --dtypes float32,float64 --devices cpu --host qe-mini1 --label "ynn4-$label" --out "$OUT/matrix-$label") \
    >> "$OUT/matrix.txt" 2>> "$OUT/run.log"
  echo "== exit $? ($label)" >> "$OUT/matrix.txt"
done

for run in run1 run2 run3; do
  log "probe48 $run"
  bash "$HERE/m6_probe.sh" ${1:+$1-}$run
done
log "done"
