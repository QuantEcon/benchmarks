#!/bin/bash
# highest.sh <host> [label]: the organiser's highest pass after the default problem-type run (#34, #37), then
# a re-pack with run_by_hand.sh's own tar line, so the tarball carries both passes. [label] is highest, or
# cuda12-highest where the default run was labelled cuda12 (a driver older than 580). <date> is the directory
# the default run wrote, its UTC date at start, not the local date.
set -euo pipefail
H=$1; LABEL=${2:-highest}
cd ~/benchmarks
D=$(basename "$(ls -d problems/results/$H/20*/ | sort | tail -1)")
[ -f "$H-$D-problems.tgz" ] || { echo "no $H-$D-problems.tgz: the default run's date isn't $D"; exit 1; }
echo "default run's date: $D; label $LABEL"
export PATH="$HOME/.local/bin:$PATH"
source .venv-problems/bin/activate
JAX_DEFAULT_MATMUL_PRECISION=highest python problems/run_suite.py --backends jax-gpu --host $H --dtypes float32 --label $LABEL 2>&1 \
  | tee problems/results/$H/$D/run-highest.log
tar czf "$H-$D-problems.tgz" "problems/results/$H"
echo "re-packed $H-$D-problems.tgz"
