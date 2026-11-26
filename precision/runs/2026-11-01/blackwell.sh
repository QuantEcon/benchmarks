#!/bin/bash
# One session on an RTX PRO 6000 Blackwell (EC2 g7e.2xlarge) for two questions:
#   1. QuantEcon/project-benchmarks-jax#20: does jax-ml/jax#41888's float64 scatter bug reach sm_120?
#   2. QuantEcon/project-econ-on-ai-hardware#4, question 1: does JAX's float64 matmul reach cuBLAS's emulation?
#
#   bash blackwell.sh           with scatter_41888.py and dot64.py beside it, as ubuntu
#
# Everything goes to ~/out, packed into ~/blackwell-<date>.tgz.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/out; mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH" XLA_PYTHON_CLIENT_PREALLOCATE=false
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
git clone -q https://github.com/QuantEcon/benchmarks.git "$HOME/benchmarks" && git -C "$HOME/benchmarks" checkout -q 40f8062
{
  nvidia-smi
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id,power.limit,compute_cap --format=csv
  lscpu | head -25
  free -g
  ls -d /usr/local/cuda* 2>/dev/null
} > "$OUT/machine.txt" 2>&1

venv() {  # <dir> <requirement>...
  [ -d "$1" ] || { uv venv -q -p 3.12 "$1" && uv pip install -q --python "$1/bin/python" "${@:2}"; }
}

echo "== 1. jax#41888's reproducer"
for v in 0.12.0.1 0.11.2; do
  for cuda in cuda13 cuda12; do
    dir=$HOME/venv-$v-$cuda
    venv "$dir" "jax[$cuda]==$v" "jaxlib==$v" || { echo "jax $v $cuda: install failed"; continue; }
    uv pip freeze --python "$dir/bin/python" > "$OUT/environment-$v-$cuda.txt"
    echo "-- jax $v, jax[$cuda]"
    timeout 900 "$dir/bin/python" "$HERE/scatter_41888.py" 2>&1 | tail -8
  done
done | tee "$OUT/scatter-41888.txt"

echo "== 2. float64 matmul, with and without cuBLAS's emulation"
PY=$HOME/venv-0.12.0.1-cuda13/bin/python
"$PY" "$HOME/benchmarks/precision/env_info.py" > "$OUT/env_info.json"
run() {  # <label> [VAR=value]...
  echo "-- $1"
  env "${@:2}" "$PY" "$HERE/dot64.py" "$1" >> "$OUT/dot64.jsonl" 2> "$OUT/dot64-$1.err" || echo "$1: exit $?"
}
run native
run emulate CUBLAS_EMULATE_DOUBLE_PRECISION=1
run emulate-eager CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager
run lt XLA_FLAGS=--xla_gpu_enable_cublaslt=true
run lt-emulate XLA_FLAGS=--xla_gpu_enable_cublaslt=true CUBLAS_EMULATE_DOUBLE_PRECISION=1
run lt-emulate-eager XLA_FLAGS=--xla_gpu_enable_cublaslt=true CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager

echo "== 3. which cuBLAS calls XLA makes, logged at one small size"
for s in native emulate; do
  extra=(); [ $s = emulate ] && extra=(CUBLAS_EMULATE_DOUBLE_PRECISION=1)
  env "${extra[@]}" CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas-$s.log" \
    CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE="$OUT/cublaslt-$s.log" \
    "$PY" "$HERE/dot64.py" "logged-$s" --sizes 1024 --kinds normal --reps 1 > /dev/null 2>&1 || echo "logged $s: exit $?"
done

echo "== 4. CuPy, as a control: does the emulation work on this card when cuBLAS is called directly?"
CUPY=$HOME/venv-cupy
if venv "$CUPY" "cupy-cuda13x[ctk]" numpy; then
  uv pip freeze --python "$CUPY/bin/python" > "$OUT/environment-cupy.txt"
  for s in native emulate emulate-eager; do
    extra=()
    [ $s != native ] && extra+=(CUBLAS_EMULATE_DOUBLE_PRECISION=1)
    [ $s = emulate-eager ] && extra+=(CUBLAS_EMULATION_STRATEGY=eager)
    echo "-- cupy $s"
    env "${extra[@]}" "$CUPY/bin/python" "$HERE/dot64.py" "cupy-$s" --lib cupy >> "$OUT/dot64.jsonl" 2> "$OUT/dot64-cupy-$s.err" || echo "cupy $s: exit $?"
  done
else
  echo "CuPy: install failed"
fi

python3 - "$OUT/dot64.jsonl" <<'EOF' | tee "$OUT/dot64-summary.txt"
import json, sys
print(f"{'label':<18} {'kind':<10} {'n':>6} {'TFLOPS':>7} {'error/u':>8}  targets")
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['label']:<18} {r['kind']:<10} {r['n']:>6} {r['tflops']:7.2f} {r['error_u']:8.2f}  {','.join(r['targets'])}")
EOF
tar czf "$HOME/blackwell-$(date -u +%F).tgz" -C "$HOME" out
echo "== Done: ~/blackwell-$(date -u +%F).tgz"
