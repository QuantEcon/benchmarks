#!/bin/bash
# Does XLA's GPU autotuner make compiling the stored Bellman step need twice its array? (QuantEcon/benchmarks#1, Phase 2)
#
#   bash a10_autotune.sh        on a Lambda gpu_1x_a10, as ubuntu
#
# The stored step at 20k, 40k and 60k in both precisions, each cell in a fresh process, first with
# XLA's defaults and then with --xla_gpu_autotune_level=0. On 2026-10-26 the A10 ran out of memory
# compiling 40k in float64 and 60k in float32; if the autotuner is the cause, both compile with it off.
set -euo pipefail
HOST=lambda-gpu_1x_a10
COMMIT=40f8062
git clone -q https://github.com/QuantEcon/benchmarks.git && cd benchmarks && git checkout -q $COMMIT
DATE=$(date -u +%F)
OUT=lectures/results/$HOST/$DATE
mkdir -p "$OUT"
{
  nvidia-smi
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id,power.limit --format=csv
  lscpu | head -25
  free -g
} > "$OUT/machine.txt" 2>&1

export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -q -p 3.12 .venv
source .venv/bin/activate
uv pip install -q "jax[cuda12]" -c precision/constraints.txt
python -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}'); assert jax.devices()[0].platform == 'gpu'"
uv pip freeze > "$OUT/environment.txt"

for autotune in default 0; do
  if [ $autotune = 0 ]; then export XLA_FLAGS=--xla_gpu_autotune_level=0; label=cuda12-autotune0; else unset XLA_FLAGS; label=cuda12; fi
  echo "== autotuner: $autotune"
  python precision/run_matrix.py --host $HOST --devices gpu --gpus 0 --workloads bellman_stored \
    --sizes 20000,40000,60000 --dtypes float64,float32 --label $label
done 2>&1 | tee "$OUT/autotune.log"
tar czf "$HOME/$HOST-$DATE.tgz" "results/$HOST" "lectures/results/$HOST"
echo "== Done: ~/$HOST-$DATE.tgz"
