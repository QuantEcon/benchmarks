#!/bin/bash
# Replaces the rest of blackwell.sh and blackwell2.sh once the native run ends. XLA sends the float64
# dot to its own Triton kernels, not cuBLAS, and autotunes them at float64 speed (68 s at n = 8192), so
# the emulation variables are tried once on that path, and otherwise with Triton's GEMMs off, which
# sends dots to cuBLAS.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/out
while pgrep -f "dot64.py native" > /dev/null; do sleep 15; done
PY=$HOME/venv-0.12.0.1-cuda13/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
NOTRITON=--xla_gpu_enable_triton_gemm=false
run() {  # <python> <label> <dot64 args> [VAR=value]...
  echo "-- $2"
  env "${@:4}" "$1" "$HERE/dot64.py" "$2" $3 >> "$OUT/dot64.jsonl" 2> "$OUT/dot64-$2.err" || echo "$2: exit $?"
}
echo "== 2b. the emulation variable on XLA's default path"
run "$PY" emulate "--sizes 2048,8192" CUBLAS_EMULATE_DOUBLE_PRECISION=1
echo "== 5. float64 matmul through cuBLAS"
run "$PY" cublas "" XLA_FLAGS=$NOTRITON
run "$PY" cublas-emulate "" XLA_FLAGS=$NOTRITON CUBLAS_EMULATE_DOUBLE_PRECISION=1
run "$PY" cublas-emulate-eager "" XLA_FLAGS=$NOTRITON CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager
run "$PY" cublaslt-emulate "--sizes 2048,8192" "XLA_FLAGS=$NOTRITON --xla_gpu_enable_cublaslt=true" CUBLAS_EMULATE_DOUBLE_PRECISION=1

echo "== 6. the compiled dot, and cuBLAS's log, with Triton on and off"
for s in default notriton notriton-emulate; do
  flags=(); [ $s != default ] && flags+=(XLA_FLAGS=$NOTRITON)
  [ $s = notriton-emulate ] && flags+=(CUBLAS_EMULATE_DOUBLE_PRECISION=1)
  env "${flags[@]}" CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas-$s.log" \
    CUBLASLT_LOG_LEVEL=5 CUBLASLT_LOG_FILE="$OUT/cublaslt-$s.log" \
    "$PY" - > "$OUT/hlo-$s.txt" 2>&1 <<'PY'
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
a = jnp.ones((2048, 2048))
c = jax.jit(jnp.matmul).lower(a, a).compile()
print(c.as_text())
c(a, a).block_until_ready()
PY
done

echo "== 4. CuPy, as a control: does the emulation work on this card when cuBLAS is called directly?"
CUPY=$HOME/venv-cupy
if [ -d "$CUPY" ] || { uv venv -q -p 3.12 "$CUPY" && uv pip install -q --python "$CUPY/bin/python" "cupy-cuda13x[ctk]" numpy; }; then
  uv pip freeze --python "$CUPY/bin/python" > "$OUT/environment-cupy.txt"
  run "$CUPY/bin/python" cupy-native "--lib cupy"
  run "$CUPY/bin/python" cupy-emulate "--lib cupy" CUBLAS_EMULATE_DOUBLE_PRECISION=1
  run "$CUPY/bin/python" cupy-emulate-eager "--lib cupy" CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager
else
  echo "CuPy: install failed"
fi

python3 - "$OUT/dot64.jsonl" <<'PY' | tee "$OUT/dot64-summary.txt"
import json, sys
print(f"{'label':<22} {'kind':<10} {'n':>6} {'TFLOPS':>7} {'error/u':>8}  first_s  targets")
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['label']:<22} {r['kind']:<10} {r['n']:>6} {r['tflops']:7.2f} {r['error_u']:8.2f} {r['first_s']:8.1f}  {','.join(r['targets'])}")
PY
tar czf "$HOME/blackwell-$(date -u +%F).tgz" -C "$HOME" out
echo "== Done (3): ~/blackwell-$(date -u +%F).tgz"
