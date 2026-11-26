#!/bin/bash
# setup.sh <ref>: a clone of QuantEcon/benchmarks at <ref> in ~/bench/benchmarks, and its .venv at the
# problem-type suite's pinned releases; then warm the imports, since the endpoint agents scan new files
# (a fresh venv's first imports once took ~48 s).
set -e
ref=$1
[ "$(command -v git)" = "$HOME/bench/tools/env/bin/git" ] || { echo "git on PATH isn't ours: $(command -v git)"; exit 1; }
[ -d ~/bench/benchmarks/.git ] || git clone -q https://github.com/QuantEcon/benchmarks.git ~/bench/benchmarks
cd ~/bench/benchmarks
git fetch -q origin
git checkout -q --detach "$ref"
git log --oneline -1
git status --short | head -20
[ -x .venv/bin/python ] || uv venv -q --python 3.12 .venv
uv pip install -q --python .venv/bin/python jax numba scipy -c problems/constraints.txt
.venv/bin/python -m compileall -q .venv/lib/python3.12/site-packages > /dev/null || true
for i in 1 2; do
  /usr/bin/time -p .venv/bin/python -c 'import numpy, scipy, numba, jax; print(numpy.__version__, scipy.__version__, numba.__version__, jax.__version__)'
done
