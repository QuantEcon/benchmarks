#!/bin/bash
# After blackwell5.sh: CuPy reaches cuBLAS's emulation through the legacy API, and XLA calls cuBLASLt.
#   9. Does JAX reach it with cuBLASLt off as well as Triton's GEMMs?
#  10. Over what range of input exponents does the emulation pay? (sigma of a lognormal input)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/out
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=$HOME/venv-0.12.0.1-cuda13/bin/python
CUPY=$HOME/venv-cupy/bin/python
LEGACY="--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=false"
run() {  # <python> <label> <dot64 args> [VAR=value]...
  echo "-- $2"
  env "${@:4}" "$1" "$HERE/dot64.py" "$2" $3 >> "$OUT/dot64.jsonl" 2> "$OUT/dot64-$2.err" || echo "$2: exit $?"
}
echo "== 9. JAX through the legacy cuBLAS API"
run "$PY" legacy "--kinds uniform,normal" "XLA_FLAGS=$LEGACY"
run "$PY" legacy-emulate "--kinds uniform,normal" "XLA_FLAGS=$LEGACY" CUBLAS_EMULATE_DOUBLE_PRECISION=1
run "$PY" legacy-emulate-autotune0 "--kinds uniform,normal" "XLA_FLAGS=$LEGACY --xla_gpu_autotune_level=0" CUBLAS_EMULATE_DOUBLE_PRECISION=1
XLA_FLAGS="$LEGACY" CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas-legacy-emulate.log" \
  "$PY" - > "$OUT/hlo-legacy-emulate.txt" 2>&1 <<'PY'
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
a = jnp.ones((2048, 2048))
c = jax.jit(jnp.matmul).lower(a, a).compile()
print(c.as_text())
c(a, a).block_until_ready()
PY
echo "== 10. the emulation against the inputs' range, n = 8192"
K=lognormal0.5,lognormal1,lognormal2,lognormal3,lognormal4
run "$CUPY" cupy-native-range "--lib cupy --sizes 8192 --kinds $K"
run "$CUPY" cupy-emulate-range "--lib cupy --sizes 8192 --kinds $K" CUBLAS_EMULATE_DOUBLE_PRECISION=1
python3 - "$OUT/dot64.jsonl" <<'PY' | tee "$OUT/dot64-summary.txt"
import json, sys
print(f"{'label':<26} {'kind':<13} {'n':>6} {'TFLOPS':>7} {'error/u':>8}  first_s  targets")
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['label']:<26} {r['kind']:<13} {r['n']:>6} {r['tflops']:7.2f} {r['error_u']:8.2f} {r['first_s']:8.1f}  {','.join(r['targets'])}")
PY
tar czf "$HOME/blackwell-$(date -u +%F).tgz" -C "$HOME" out
echo "== Done (6): ~/blackwell-$(date -u +%F).tgz"
