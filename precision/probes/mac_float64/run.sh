#!/bin/bash
# QuantEcon/project-benchmarks-jax#48's probe on one Mac, each variant in a fresh process. From the
# root of a QuantEcon/benchmarks checkout:
#   PROBE_HOST=<host> MLXPY=<an MLX venv's python> bash precision/probes/mac_float64/run.sh <out> [--quick]
# PY is the benchmarks venv's python (.venv/bin/python by default). MLXPY comes from a venv built
# from requirements-mlx.txt; without it the MLX half of step 4 is left out. --quick runs small sizes,
# as a check that everything runs, not a measurement. Written for macOS's bash 3.2.
set -u
OUT=${1:?the output folder}
QUICK=${2:-}
P=precision/probes/mac_float64/probe.py
PY=${PY:-.venv/bin/python}
MLXPY=${MLXPY:-}
mkdir -p "$OUT"
if [ "$QUICK" = --quick ]; then
  MM=8,64,512; CL=2000,4000; BE=1000,2000; PV=256,512
else
  MM=8,64,512,1024,4096; CL=16000,18000; BE=2000,8000,14000; PV=1000,4000
fi
log() { echo "== $* $(date -u +%T)" >> "$OUT/run.log"; }
# A step that fails is logged and counted, and the run goes on: on a borrowed machine, the other
# steps' results are worth keeping. The run ends with a nonzero exit if any step failed
NFAIL=0
FAILED=""
failed() { NFAIL=$((NFAIL + 1)); FAILED="$FAILED, $1"; log "failed: $1"; }
{ git log -1 --format='benchmarks at %h: %s' 2>&1 || echo "not a git checkout"
  git status --porcelain 2>&1; } > "$OUT/commit.txt"

log features
"$PY" "$P" features > "$OUT/features.txt" 2>&1 || failed features
if [ -n "$MLXPY" ]; then
  "$MLXPY" -c "import mlx.core as mx, numpy; print('mlx', mx.__version__, '| numpy', numpy.__version__)" \
    >> "$OUT/features.txt" 2>&1 || failed "features, mlx"
fi

for v in numpy jax-ynn jax-xla-dot; do
  log matmul $v
  "$PY" "$P" matmul $v $MM >> "$OUT/matmul.jsonl" 2>> "$OUT/run.log" || failed "matmul $v"
done

# The dump goes in a .noindex folder: in an ordinary one, its few hundred files woke macOS's
# mediaanalysisd for about half a minute of a core (precision/runs/2026-11-10/qe-mini1/noindex/)
for v in default no-reduce-fusion add rowmax; do
  log cliff $v
  PROBE_DUMP="$OUT/xla-dump.noindex/$v" "$PY" "$P" cliff $v $CL >> "$OUT/cliff.jsonl" 2>> "$OUT/run.log" \
    || failed "cliff $v"
done
# Of the dump, only the optimised modules say which ops YNNPACK took
if [ -d "$OUT/xla-dump.noindex" ]; then
  find "$OUT/xla-dump.noindex" -type f ! -name '*after_optimizations.txt' -delete
fi

for k in bellman pv; do
  if [ $k = bellman ]; then S=$BE; else S=$PV; fi
  log step jax $k
  "$PY" "$P" step jax $k $S >> "$OUT/step.jsonl" 2>> "$OUT/run.log" || failed "step jax $k"
  if [ -n "$MLXPY" ]; then
    log step mlx $k
    "$MLXPY" "$P" step mlx $k $S >> "$OUT/step.jsonl" 2>> "$OUT/run.log" || failed "step mlx $k"
  fi
done

"$PY" "$P" summarise "$OUT" > "$OUT/summary.md" || failed summarise
if [ $NFAIL -gt 0 ]; then
  log "done, $NFAIL failed: ${FAILED#, }"
  exit 1
fi
log "done, every step ok"
