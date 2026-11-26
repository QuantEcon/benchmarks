#!/bin/bash
# QuantEcon/project-benchmarks-jax#48's probe, measured on qe-mini1's M6 (Matt, 2026-11-06: "You may use
# the M6 now"), in a quiet window agreed with the sessions here, which stay running. state.txt samples
# the machine every 30 s. The probe runs from the benchmarks clone at its commit, with the benchmarks
# venv and the MLX venv in ~/bench/mlx.
set -u
B=~/work/quantecon/benchmarks
# The first run met mediaanalysisd at 114% of a core mid-run; repeats take a label, e.g. run2
OUT=~/bench/probe48/$(date -u +%F)-qe-mini1${1:+-$1}
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -10
  memory_pressure | tail -1
}
sample() { while sleep 5; do state sample >> "$OUT/state.txt"; done; }
state before > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
cd "$B"
echo "== probe at $(git log -1 --format='%h %s')" >> "$OUT/state.txt"
PROBE_HOST=qe-mini1 PY=$HOME/bench/benchmarks/.venv/bin/python MLXPY=$HOME/bench/mlx/.venv/bin/python \
  bash precision/probes/mac_float64/run.sh "$OUT"
echo "== run.sh exit $?" >> "$OUT/state.txt"
