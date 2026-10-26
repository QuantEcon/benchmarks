#!/bin/bash
# QuantEcon/benchmarks#4: every lecture in lectures.txt as used (as written, JAX's defaults), timed per cell.
#   ./lectures_set.sh <out dir> <repetitions> <platform>...      platforms: cuda (card 0b:00.0) or cpu
# Each lecture has an hour; perl's alarm stands in for timeout, which macOS lacks.
# DTYPES runs each lecture at each precision in turn, on each platform: `as-written` by default,
# or for example "float64 float32" (run_lecture.py's --dtype). LIST names the lectures.
# ROOT holds lectures.txt and lectures/<lecture>.md; PY has the lectures' imports; ENV_INFO is
# benchmarks' precision/. The defaults are the machines' layout; the workflow sets its own.
cd "${ROOT:-$HOME/bench/lectures-all}"
PY=${PY:-.venv/bin/python}; RUNNER=${RUNNER:-run_lecture.py}; ENV_INFO=${ENV_INFO:-$HOME/bench/benchmarks/precision}
DTYPES=${DTYPES:-as-written}; LIST=${LIST:-lectures.txt}
OUT=$1; REPS=$2; shift 2
mkdir -p "$OUT"
cap() { perl -e 'alarm shift; exec @ARGV' 3600 "$@"; }
for rep in $(seq 1 "$REPS"); do
  for lec in $(cat "$LIST"); do
    for plat in "$@"; do
      mkdir -p "$OUT/$lec"
      if [ "$plat" = cuda ]; then dev="JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"; else dev="JAX_PLATFORMS=cpu"; fi
      for dt in $DTYPES; do
        if [ "$dt" = as-written ]; then err="$plat-$rep.err"; else err="$plat-$dt-$rep.err"; fi
        cap env $dev MPLBACKEND=Agg "$PY" "$RUNNER" "lectures/$lec.md" --dtype "$dt" \
          --out "$OUT/$lec/$plat-$dt-$rep.json" --env-info "$ENV_INFO" \
          > /dev/null 2> "$OUT/$lec/$err"
        status=$?
        summary=$("$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"{r['lecture_s']:.1f} s, cells ok {r['all_ok']}, {r['default_float']}\")" "$OUT/$lec/$plat-$dt-$rep.json" 2>/dev/null || echo "no result")
        echo "$(date +%T) $lec $plat $dt rep $rep: exit $status, $summary"
      done
    done
  done
done
echo ALLDONE
