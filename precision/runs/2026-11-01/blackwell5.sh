#!/bin/bash
# After blackwell4.sh: CuPy as a control. Does cuBLAS 13.8 emulate float64 on this card when called
# directly, through the environment variable or through the math mode?
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$HOME/out
export PATH="$HOME/.local/bin:$PATH"
while pgrep -f "bash blackwell4.sh" > /dev/null; do sleep 15; done
CUPY=$HOME/venv-cupy
rm -rf "$CUPY"
uv venv -q -p 3.12 "$CUPY" && uv pip install -q --python "$CUPY/bin/python" "cupy-cuda13x[ctk]" numpy \
  || { echo "CuPy: install failed"; exit 1; }
uv pip freeze --python "$CUPY/bin/python" > "$OUT/environment-cupy.txt"
run() {  # <label> <dot64 args> [VAR=value]...
  echo "-- $1"
  env "${@:3}" "$CUPY/bin/python" "$HERE/dot64.py" "$1" --lib cupy $2 >> "$OUT/dot64.jsonl" 2> "$OUT/dot64-$1.err" || echo "$1: exit $?"
}
echo "== 4. CuPy, as a control"
run cupy-native ""
run cupy-emulate "" CUBLAS_EMULATE_DOUBLE_PRECISION=1
run cupy-emulate-eager "" CUBLAS_EMULATE_DOUBLE_PRECISION=1 CUBLAS_EMULATION_STRATEGY=eager
run cupy-mathmode "--math-mode 8"
run cupy-mathmode-eager "--math-mode 8" CUBLAS_EMULATION_STRATEGY=eager
CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG="$OUT/cublas-cupy-mathmode.log" "$CUPY/bin/python" "$HERE/dot64.py" logged-cupy-mathmode \
  --lib cupy --math-mode 8 --sizes 1024 --kinds normal --reps 1 > /dev/null 2>&1 || echo "logged: exit $?"
python3 - "$OUT/dot64.jsonl" <<'PY' | tee "$OUT/dot64-summary.txt"
import json, sys
print(f"{'label':<22} {'kind':<10} {'n':>6} {'TFLOPS':>7} {'error/u':>8}  first_s  targets")
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['label']:<22} {r['kind']:<10} {r['n']:>6} {r['tflops']:7.2f} {r['error_u']:8.2f} {r['first_s']:8.1f}  {','.join(r['targets'])}")
PY
tar czf "$HOME/blackwell-$(date -u +%F).tgz" -C "$HOME" out
echo "== Done (5): ~/blackwell-$(date -u +%F).tgz"
