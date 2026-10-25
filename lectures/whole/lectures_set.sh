#!/bin/bash
# QuantEcon/benchmarks#4: every lecture in lectures.txt as used (as written, JAX's defaults), timed per cell.
#   ./lectures_set.sh <out dir> <repetitions> <platform>...      platforms: cuda (card 0b:00.0) or cpu
# Each lecture has an hour; perl's alarm stands in for timeout, which macOS lacks.
cd ~/bench/lectures-all
OUT=$1; REPS=$2; shift 2
mkdir -p "$OUT"
cap() { perl -e 'alarm shift; exec @ARGV' 3600 "$@"; }
for rep in $(seq 1 "$REPS"); do
  for lec in $(cat lectures.txt); do
    for plat in "$@"; do
      mkdir -p "$OUT/$lec"
      if [ "$plat" = cuda ]; then dev="JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"; else dev="JAX_PLATFORMS=cpu"; fi
      cap env $dev MPLBACKEND=Agg .venv/bin/python run_lecture.py "lectures/$lec.md" --dtype as-written \
        --out "$OUT/$lec/$plat-as-written-$rep.json" --env-info ~/bench/benchmarks/precision \
        > /dev/null 2> "$OUT/$lec/$plat-$rep.err"
      status=$?
      summary=$(.venv/bin/python -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"{r['lecture_s']:.1f} s, cells ok {r['all_ok']}, {r['default_float']}\")" "$OUT/$lec/$plat-as-written-$rep.json" 2>/dev/null || echo "no result")
      echo "$(date +%T) $lec $plat rep $rep: exit $status, $summary"
    done
  done
done
echo ALLDONE
