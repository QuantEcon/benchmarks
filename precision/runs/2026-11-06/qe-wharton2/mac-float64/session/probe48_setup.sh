#!/bin/bash
# probe48_setup.sh: QuantEcon/project-benchmarks-jax#48's probe for the M5 Pro, after the suite (Matt,
# 2026-11-06: "You may queue work for M5PRO"). The probe's commit goes into a worktree beside the suite's
# checkout, and MLX into a venv of its own, both under ~/bench/probe48 so that the copy-back of
# ~/bench/pbj-m5pro stays small. Then the imports are warmed, since the endpoint agents scan new files.
set -e
REF=${PROBE_REF:?the probe commit, set in chain.sh}
P=~/bench/probe48
mkdir -p $P
cd ~/bench/benchmarks
git fetch -q origin
[ -d $P/src ] || git worktree add -q --detach $P/src "$REF"
git -C $P/src log -1 --format='probe at %h: %s'
[ -x $P/mlx-venv/bin/python ] || uv venv -q --python 3.12 $P/mlx-venv
uv pip install -q --python $P/mlx-venv/bin/python -r $P/src/precision/probes/mac_float64/requirements-mlx.txt
$P/mlx-venv/bin/python -m compileall -q $P/mlx-venv/lib/python3.12/site-packages > /dev/null || true
for i in 1 2; do /usr/bin/time -p $P/mlx-venv/bin/python -c 'import mlx.core as mx, numpy; print("mlx", mx.__version__, "numpy", numpy.__version__)'; done
