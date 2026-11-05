#!/bin/bash
# Phase 2 of #1 outside RunsOn: what the workflow's precision job does, on a rented GPU machine.
#
#   git clone https://github.com/QuantEcon/benchmarks.git && cd benchmarks
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
#
# On an AMD machine, one without nvidia-smi, it records the machine with amd-smi instead, and
# installs jax 0.12.0 (constraints-rocm.txt) with the plugin for the ROCm in /opt/rocm:
# jax-rocm7-* for ROCm 7 and jax-rocm10-* for ROCm 10. For a machine whose ROCm is neither,
# ROCM_INDEX names an index of ROCm's own Python packages to install first, and ROCM_MAJOR their
# major version; ROCm/TheRock's RELEASES.md lists the indexes. The lectures' and the scale
# scripts' results are then named rocm where NVIDIA's are named cuda.
set -euo pipefail
HOST=$1; DEVICES=${2:-cpu,gpu}; QUICK=${3:-}
DATE=$(date -u +%F)
OUT="lectures/results/$HOST/$DATE"
LECTURE_JAX=https://raw.githubusercontent.com/QuantEcon/lecture-jax/4b939ec49a7f1901245f68749c1b1e18e3454d34/lectures
LECTURES="overborrowing wealth_dynamics_jax schelling_shocks opt_invest hopenhayn"
REPS=3; AIYAGARI=200,1000,2000,5000,10000; IFP=200,1000,5000,20000,50000
[ "$QUICK" = --quick ] && { LECTURES=schelling_shocks; REPS=1; AIYAGARI=200,1000; IFP=200,1000; }
mkdir -p "results/$HOST" "$OUT"
if [ -n "$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null)" ]; then PLATFORM=cuda; else PLATFORM=rocm; fi

echo "== The machine"
if [ $PLATFORM = cuda ]; then
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
else
  rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo none)
  ROCM_MAJOR=${ROCM_MAJOR:-${rocm%%.*}}; LABEL=""
  {
    head -4 /etc/os-release
    echo "kernel $(uname -r), amdgpu module $(cat /sys/module/amdgpu/version 2>/dev/null || echo 'built in'), ROCm in /opt/rocm: $rocm"
    amd-smi version
    amd-smi static
    amd-smi topology
    lscpu | head -25
    free -g
  } > "$OUT/machine.txt" 2>&1 || true
  case $ROCM_MAJOR in
    7|10) echo "ROCm $rocm: jax-rocm$ROCM_MAJOR-plugin" ;;
    *) echo "ROCm $rocm has no jax 0.12.0 plugin: set ROCM_MAJOR to 7 or 10, and ROCM_INDEX for ROCm's own packages"
       exit 1 ;;
  esac
fi

echo "== Python 3.12 and JAX"
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# The full run after --quick keeps the check's environment: uv won't replace one unasked
[ -d .venv-phase2 ] || uv venv -q -p 3.12 .venv-phase2
source .venv-phase2/bin/activate
if [ $PLATFORM = cuda ]; then
  CONSTRAINTS=precision/constraints.txt
  uv pip install -q "jax[$CUDA]" -c $CONSTRAINTS
else
  CONSTRAINTS=precision/constraints-rocm.txt
  if [ -n "${ROCM_INDEX:-}" ]; then
    # ROCm's packages carry device code for the cards named, by gfx target; GFX overrides amd-smi's
    GFX=${GFX:-$(python -c 'import sys; sys.path.insert(0, "precision"); import env_info; print(next((c["gfx_target"] for c in env_info.amd_gpus() if c["gfx_target"]), ""))')}
    [ -n "$GFX" ] || { echo "amd-smi gave no gfx target: set GFX, e.g. GFX=gfx942"; exit 1; }
    uv pip install -q --index-url "$ROCM_INDEX" "rocm[libraries,device-$GFX]"
  fi
  uv pip install -q jax "jax-rocm$ROCM_MAJOR-plugin" "jax-rocm$ROCM_MAJOR-pjrt" -c $CONSTRAINTS
fi
JAX_PLATFORMS=$PLATFORM python -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}'); assert jax.devices()[0].platform == 'gpu', 'JAX sees no GPU'"
limit='import jax; d = jax.devices()[0]; print(d.device_kind, "allocator limit, GiB:", round(d.memory_stats()["bytes_limit"] / 2**30, 2))'
XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "$limit" | tee -a "$OUT/machine.txt" || true
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 python -c "$limit" | tee -a "$OUT/machine.txt" || true

echo "== The precision matrix"
gpus=$(python -c 'import sys; sys.path.insert(0, "precision"); import env_info; print(",".join(str(c["index"]) for c in env_info.nvidia_gpus() or env_info.amd_gpus()))')
python precision/run_matrix.py --host "$HOST" --devices "$DEVICES" --gpus "$gpus" ${QUICK:+--quick} ${LABEL:+--label "$LABEL"}
JAX_DEFAULT_MATMUL_PRECISION=highest python precision/run_matrix.py --host "$HOST" --devices gpu --gpus "$gpus" \
  --workloads matmul ${QUICK:+--quick} --label "${LABEL:+$LABEL-}highest"

echo "== The lectures' packages, as the TITAN RTX run had them"
grep -vE '^(jax|jaxlib|jax-cuda[0-9]+-(plugin|pjrt)|nvidia-[a-z0-9-]+)==' \
  lectures/results/qe-wharton1-gpu/2026-10-25/whole/environment.txt > "$OUT/requirements.txt"
uv pip install -q -r "$OUT/requirements.txt" -c $CONSTRAINTS
uv pip freeze > "$OUT/environment.txt"

echo "== Five lectures as written, on the card"
LDIR=$(mktemp -d)
for lecture in $LECTURES aiyagari_jax ifp_egm_jax; do
  curl -fsSL -o "$LDIR/$lecture.md" "$LECTURE_JAX/$lecture.md"
done
# HIP also reads CUDA_VISIBLE_DEVICES, so each platform's lectures see only its own selection
if [ $PLATFORM = cuda ]; then
  unset HIP_VISIBLE_DEVICES; export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
else
  unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER; export HIP_VISIBLE_DEVICES=0
fi
export MPLBACKEND=Agg
mkdir -p "$OUT/whole"
for rep in $(seq 1 $REPS); do
  for lecture in $LECTURES; do
    mkdir -p "$OUT/whole/$lecture"
    JAX_PLATFORMS=$PLATFORM timeout 3600 python lectures/run_lecture.py "$LDIR/$lecture.md" --dtype as-written \
      --out "$OUT/whole/$lecture/$PLATFORM-as-written-$rep.json" --env-info precision \
      > /dev/null 2> "$OUT/whole/$lecture/$PLATFORM-$rep.err" && rm -f "$OUT/whole/$lecture/$PLATFORM-$rep.err" \
      || echo "$lecture rep $rep: exit $?"
    python -c "import json,sys; r=json.load(open(sys.argv[1])); print(f\"$lecture rep $rep: {r['lecture_s']:.1f} s, cells ok {r['all_ok']}, {r['default_float']}\")" \
      "$OUT/whole/$lecture/$PLATFORM-as-written-$rep.json" 2>/dev/null || echo "$lecture rep $rep: no result"
  done
done | tee "$OUT/whole/lectures-set.log"

echo "== The scale scripts, on the card"
mkdir -p "$OUT/aiyagari_jax/scale" "$OUT/ifp_egm_jax/scale"
for dtype in float64 float32; do
  JAX_PLATFORMS=$PLATFORM XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 900 python lectures/aiyagari_jax/aiyagari_scale.py \
    "$LDIR/aiyagari_jax.md" --dtype $dtype --sizes $AIYAGARI --out "$OUT/aiyagari_jax/scale/$PLATFORM-$dtype-1.json" || echo "aiyagari_jax $dtype: exit $?"
  JAX_PLATFORMS=$PLATFORM XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 900 python lectures/ifp_egm_jax/ifp_scale.py \
    "$LDIR/ifp_egm_jax.md" --dtype $dtype --sizes $IFP --out "$OUT/ifp_egm_jax/scale/$PLATFORM-$dtype-1.json" || echo "ifp_egm_jax $dtype: exit $?"
done
rm -rf "$LDIR"

python precision/summarise.py "results/$HOST" > "$OUT/precision-summary.md" || true
python lectures/aiyagari_jax/summarise_scale.py "$OUT"/aiyagari_jax/scale/*.json > "$OUT/aiyagari-summary.md" || true
tar czf "$HOST-$DATE.tgz" "results/$HOST" "lectures/results/$HOST"
echo "== Done: $HOST-$DATE.tgz. Copy it off the machine, then terminate the machine."
