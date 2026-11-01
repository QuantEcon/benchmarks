#!/bin/bash
# After blackwell3.sh: is XLA's autotuner what makes float64 compile slowly on the newer cards?
# Phase 2 timed the recomputing step's compile at 19 s (20k, 40k) and 92 s (60k) on this card, and
# the float64 dot's first call here takes 68 s at n = 8192. Both again with the autotuner off.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/out
while pgrep -f "bash blackwell3.sh" > /dev/null; do sleep 15; done
source "$HOME/venv-0.12.0.1-cuda13/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$HOME/benchmarks"
echo "== 7. the recomputing step's compile, autotuner on and off"
python precision/run_matrix.py --host aws-g7e.2xlarge --devices gpu --gpus 0 --workloads bellman_recompute \
  --sizes 20000,40000 --dtypes float64,float32
XLA_FLAGS=--xla_gpu_autotune_level=0 python precision/run_matrix.py --host aws-g7e.2xlarge --devices gpu --gpus 0 \
  --workloads bellman_recompute --sizes 20000,40000 --dtypes float64,float32 --label autotune0
cp results/aws-g7e.2xlarge/*.jsonl "$OUT/"
echo "== 8. the float64 dot with the autotuner off"
XLA_FLAGS=--xla_gpu_autotune_level=0 python "$HERE/dot64.py" autotune0 --sizes 2048,8192,16384 --kinds normal \
  >> "$OUT/dot64.jsonl" 2> "$OUT/dot64-autotune0.err" || echo "autotune0: exit $?"
tail -1 "$OUT/dot64.jsonl"
tar czf "$HOME/blackwell-$(date -u +%F).tgz" -C "$HOME" out
echo "== Done (4): ~/blackwell-$(date -u +%F).tgz"
