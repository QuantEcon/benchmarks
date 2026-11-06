#!/bin/bash
# tpu_lectures.sh, on the TPU VM, after tpu_cells.sh: run_by_hand.sh's lecture section at 5cd9d3d, with JAX_PLATFORMS=tpu.
# The five lectures as written, at lecture-jax 4b939ec, through lectures/run_lecture.py unchanged: one repetition
# first (the cold run a student pays), then the aiyagari and ifp scale scripts, then repetitions 2 and 3 if time allows.
# Packages: the TITAN RTX run's list without jax, jaxlib and the CUDA wheels, under precision/constraints.txt.
until [ -f ~/session/cells.done ]; do sleep 10; done
source ~/venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
cd ~/benchmarks
OUT=~/out/lectures; mkdir -p $OUT/whole $OUT/aiyagari_jax/scale $OUT/ifp_egm_jax/scale
LECTURE_JAX=https://raw.githubusercontent.com/QuantEcon/lecture-jax/4b939ec49a7f1901245f68749c1b1e18e3454d34/lectures
LECTURES="overborrowing wealth_dynamics_jax schelling_shocks opt_invest hopenhayn"
AIYAGARI=200,1000,2000,5000,10000; IFP=200,1000,5000,20000,50000
STOP=$(date -u -d '2026-11-06 04:35' +%s)   # leave 10 min before the watchdog's 04:45:20 UTC
echo "== packages: start $(date -u '+%T UTC')"
grep -vE '^(jax|jaxlib|jax-cuda[0-9]+-(plugin|pjrt)|nvidia-[a-z0-9-]+)==' \
  lectures/results/qe-wharton1-gpu/2026-10-25/whole/environment.txt > $OUT/requirements.txt
uv pip install -q -r $OUT/requirements.txt -c ~/constraints.txt
uv pip freeze > $OUT/environment.txt
python -c 'import jax; print(f"JAX {jax.__version__}: {jax.devices()}")'
LDIR=$(mktemp -d)
for lecture in $LECTURES aiyagari_jax ifp_egm_jax; do curl -fsSL -o "$LDIR/$lecture.md" "$LECTURE_JAX/$lecture.md"; done
export MPLBACKEND=Agg
run_rep() {
  rep=$1
  for lecture in $LECTURES; do
    [ $(date +%s) -lt $STOP ] || { echo "$lecture rep $rep: skipped, out of time"; continue; }
    mkdir -p "$OUT/whole/$lecture"
    JAX_PLATFORMS=tpu timeout 900 python lectures/run_lecture.py "$LDIR/$lecture.md" --dtype as-written \
      --out "$OUT/whole/$lecture/tpu-as-written-$rep.json" --env-info precision \
      > /dev/null 2> "$OUT/whole/$lecture/tpu-$rep.err" && rm -f "$OUT/whole/$lecture/tpu-$rep.err" \
      || echo "$lecture rep $rep: exit $?"
    python -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"$lecture rep $rep: {r['lecture_s']:.1f} s, cells ok {r['all_ok']}, {r['default_float']}\")" \
      "$OUT/whole/$lecture/tpu-as-written-$rep.json" 2>/dev/null || echo "$lecture rep $rep: no result"
  done
}
echo "== lectures rep 1: start $(date -u '+%T UTC')"
run_rep 1 | tee -a $OUT/whole/lectures-set.log
echo "== scale scripts: start $(date -u '+%T UTC')"
for dtype in float64 float32; do
  [ $(date +%s) -lt $STOP ] || { echo "scale $dtype: skipped, out of time"; continue; }
  JAX_PLATFORMS=tpu timeout 600 python lectures/aiyagari_jax/aiyagari_scale.py "$LDIR/aiyagari_jax.md" --dtype $dtype \
    --sizes $AIYAGARI --out "$OUT/aiyagari_jax/scale/tpu-$dtype-1.json" || echo "aiyagari_jax $dtype: exit $?"
  JAX_PLATFORMS=tpu timeout 600 python lectures/ifp_egm_jax/ifp_scale.py "$LDIR/ifp_egm_jax.md" --dtype $dtype \
    --sizes $IFP --out "$OUT/ifp_egm_jax/scale/tpu-$dtype-1.json" || echo "ifp_egm_jax $dtype: exit $?"
done
python lectures/aiyagari_jax/summarise_scale.py $OUT/aiyagari_jax/scale/*.json > $OUT/aiyagari-summary.md 2>&1 || true
for rep in 2 3; do echo "== lectures rep $rep: start $(date -u '+%T UTC')"; run_rep $rep | tee -a $OUT/whole/lectures-set.log; done
rm -rf "$LDIR"
echo "$(date -u '+%F %T UTC')" > ~/session/lectures.done
