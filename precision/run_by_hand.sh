#!/bin/bash
# Phase 2 of #1 outside RunsOn: what the workflow's precision job does, on a rented GPU machine.
#
#   git clone -b phase2-cloud-cards https://github.com/QuantEcon/benchmarks.git && cd benchmarks
#   bash precision/run_by_hand.sh <host> [devices] [--quick]
#
# <host> names the machine in the results as <provider>-<instance>, e.g. lambda-gpu_1x_a100.
# [devices] is cpu,gpu by default; add gpu:2 on a two-card machine. --quick runs the small sizes,
# one lecture once and the scale scripts on small grids, to check the machine.
#
# It records the machine (nvidia-smi, its topology, the driver, the CPU), installs a Python 3.12
# environment with uv in .venv-phase2, and stops if JAX sees no GPU. It picks jax[cuda13] on a
# driver of 580 or later, and otherwise jax[cuda12], labelled cuda12. Then it runs:
#   1. the precision matrix;
#   2. matmul again with JAX_DEFAULT_MATMUL_PRECISION=highest, labelled highest, since TF32 is the default;
#   3. five lectures as written, three times each on the card;
#   4. the aiyagari_jax and ifp_egm_jax scale scripts on the card.
# Results go to results/<host>/ and lectures/results/<host>/<date>/, packed into <host>-<date>.tgz.
set -euo pipefail
HOST=$1; DEVICES=${2:-cpu,gpu}; QUICK=${3:-}
DATE=$(date -u +%F)
OUT="lectures/results/$HOST/$DATE"
LECTURE_JAX=https://raw.githubusercontent.com/QuantEcon/lecture-jax/4b939ec49a7f1901245f68749c1b1e18e3454d34/lectures
LECTURES="overborrowing wealth_dynamics_jax schelling_shocks opt_invest hopenhayn"
REPS=3; AIYAGARI=200,1000,2000,5000,10000; IFP=200,1000,5000,20000,50000
[ "$QUICK" = --quick ] && { LECTURES=schelling_shocks; REPS=1; AIYAGARI=200,1000; IFP=200,1000; }
mkdir -p "results/$HOST" "$OUT"

echo "== The machine"
{
  nvidia-smi
  nvidia-smi topo -m
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id,power.limit --format=csv
  lscpu | head -25
  free -g
} > "$OUT/machine.txt" 2>&1
driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
if [ "${driver%%.*}" -ge 580 ]; then CUDA=cuda13; LABEL=""; else CUDA=cuda12; LABEL=cuda12; fi
echo "driver $driver: jax[$CUDA]"

echo "== Python 3.12 and JAX"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -q -p 3.12 .venv-phase2
source .venv-phase2/bin/activate
uv pip install -q "jax[$CUDA]" -c precision/constraints.txt
python -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}'); assert jax.devices()[0].platform == 'gpu', 'JAX sees no GPU'"
limit='import jax; d = jax.devices()[0]; print(d.device_kind, "allocator limit, GiB:", round(d.memory_stats()["bytes_limit"] / 2**30, 2))'
XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "$limit" | tee -a "$OUT/machine.txt"
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python -c "$limit" | tee -a "$OUT/machine.txt"

echo "== The precision matrix"
gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
python precision/run_matrix.py --host "$HOST" --devices "$DEVICES" --gpus "$gpus" ${QUICK:+--quick} ${LABEL:+--label "$LABEL"}
JAX_DEFAULT_MATMUL_PRECISION=highest python precision/run_matrix.py --host "$HOST" --devices gpu --gpus "$gpus" \
  --workloads matmul ${QUICK:+--quick} --label "${LABEL:+$LABEL-}highest"

echo "== The lectures' packages, as the TITAN RTX run had them"
grep -vE '^(jax|jaxlib|jax-cuda[0-9]+-(plugin|pjrt)|nvidia-[a-z0-9-]+)==' \
  lectures/results/qe-wharton1-gpu/2026-10-25/whole/environment.txt > "$OUT/requirements.txt"
uv pip install -q -r "$OUT/requirements.txt" -c precision/constraints.txt
uv pip freeze > "$OUT/environment.txt"

echo "== Five lectures as written, on the card"
LDIR=$(mktemp -d)
for lecture in $LECTURES aiyagari_jax ifp_egm_jax; do
  curl -fsSL -o "$LDIR/$lecture.md" "$LECTURE_JAX/$lecture.md"
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
for rep in $(seq 1 $REPS); do
  for lecture in $LECTURES; do
    mkdir -p "$OUT/whole/$lecture"
    JAX_PLATFORMS=cuda timeout 3600 python lectures/run_lecture.py "$LDIR/$lecture.md" --dtype as-written \
      --out "$OUT/whole/$lecture/cuda-as-written-$rep.json" --env-info precision \
      > /dev/null 2> "$OUT/whole/$lecture/cuda-$rep.err" && rm -f "$OUT/whole/$lecture/cuda-$rep.err" \
      || echo "$lecture rep $rep: exit $?"
    python -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"$lecture rep $rep: {r['lecture_s']:.1f} s, cells ok {r['all_ok']}, {r['default_float']}\")" \
      "$OUT/whole/$lecture/cuda-as-written-$rep.json" 2>/dev/null || echo "$lecture rep $rep: no result"
  done
done | tee "$OUT/whole/lectures-set.log"

echo "== The scale scripts, on the card"
mkdir -p "$OUT/aiyagari_jax/scale" "$OUT/ifp_egm_jax/scale"
for dtype in float64 float32; do
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 900 python lectures/aiyagari_jax/aiyagari_scale.py \
    "$LDIR/aiyagari_jax.md" --dtype $dtype --sizes $AIYAGARI --out "$OUT/aiyagari_jax/scale/cuda-$dtype-1.json" || echo "aiyagari_jax $dtype: exit $?"
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 900 python lectures/ifp_egm_jax/ifp_scale.py \
    "$LDIR/ifp_egm_jax.md" --dtype $dtype --sizes $IFP --out "$OUT/ifp_egm_jax/scale/cuda-$dtype-1.json" || echo "ifp_egm_jax $dtype: exit $?"
done
rm -rf "$LDIR"

python precision/summarise.py "results/$HOST" > "$OUT/precision-summary.md" || true
python lectures/aiyagari_jax/summarise_scale.py "$OUT"/aiyagari_jax/scale/*.json > "$OUT/aiyagari-summary.md" || true
tar czf "$HOST-$DATE.tgz" "results/$HOST" "lectures/results/$HOST"
echo "== Done: $HOST-$DATE.tgz. Copy it off the machine, then terminate the machine."
