#!/bin/bash
# setup.sh: bring ~/bench/benchmarks to main and its .venv to the problem-type suite's pinned releases, then warm
# the imports: the machine's endpoint agents scan new files, and a fresh venv's first imports once took ~48 s.
set -e
cd ~/bench/benchmarks
which git; git --version
git status --short | head -20
git fetch origin
git checkout main
git pull --ff-only origin main
git log --oneline -1
uv pip install --python .venv/bin/python jax numba scipy -c problems/constraints.txt
.venv/bin/python -m compileall -q .venv/lib/python3.12/site-packages > /dev/null || true
for i in 1 2; do /usr/bin/time -p .venv/bin/python -c 'import numpy, scipy, numba, jax; print(numpy.__version__, scipy.__version__, numba.__version__, jax.__version__)'; done
