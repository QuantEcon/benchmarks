#!/bin/bash
# QuantEcon/project-benchmarks-jax#60 on the pod, phase B: items 1, 3, 4 (if CAPS=1) and 6, in order
set -uo pipefail
H=runpod-rtx5090; D=$(date -u +%F)
mark() { echo "$(date -u +%FT%TZ) $*" >> ~/status.txt; }
cd ~/benchmarks
mark "B start"
bash problems/run_by_hand.sh $H; mark "B problems-full exit $?"
source .venv-problems/bin/activate
JAX_DEFAULT_MATMUL_PRECISION=highest python problems/run_suite.py --backends jax-gpu --host $H --dtypes float32 --label highest 2>&1 \
  | tee problems/results/$H/$D/run-highest.log; mark "B highest exit ${PIPESTATUS[0]}"
deactivate
tar czf $H-$D-problems.tgz problems/results/$H; mark "B problems-tgz exit $?"
bash precision/run_by_hand.sh $H gpu; mark "B precision-full exit $?"
bash precision/sustained.sh $H; mark "B sustained exit $?"
if [ "${CAPS:-0}" = 1 ]; then
  bash precision/sustained.sh $H --minutes 10 --idle 5 --cap 450; mark "B cap450 exit $?"
  bash precision/sustained.sh $H --minutes 10 --idle 5 --cap 400; mark "B cap400 exit $?"
fi
mkdir -p ~/dot64
uv venv -q -p 3.12 ~/venv-cupy && uv pip install -q --python ~/venv-cupy/bin/python "cupy-cuda13x[ctk]" numpy
uv pip freeze --python ~/venv-cupy/bin/python > ~/dot64/environment-cupy.txt
DOT=precision/runs/2026-11-01/dot64.py
~/venv-cupy/bin/python $DOT cupy-native --lib cupy >> ~/dot64/dot64.jsonl 2> ~/dot64/dot64-cupy-native.err; mark "B dot64-native exit $?"
CUBLAS_EMULATE_DOUBLE_PRECISION=1 ~/venv-cupy/bin/python $DOT cupy-emulate --lib cupy >> ~/dot64/dot64.jsonl 2> ~/dot64/dot64-cupy-emulate.err; mark "B dot64-emulate exit $?"
SWEEP="--lib cupy --sizes 8192 --kinds lognormal0.5,lognormal1,lognormal2,lognormal3,lognormal4"
~/venv-cupy/bin/python $DOT cupy-native-sigma $SWEEP >> ~/dot64/dot64.jsonl 2> ~/dot64/dot64-cupy-native-sigma.err; mark "B dot64-native-sigma exit $?"
CUBLAS_EMULATE_DOUBLE_PRECISION=1 ~/venv-cupy/bin/python $DOT cupy-emulate-sigma $SWEEP >> ~/dot64/dot64.jsonl 2> ~/dot64/dot64-cupy-emulate-sigma.err; mark "B dot64-emulate-sigma exit $?"
mark "B done"
