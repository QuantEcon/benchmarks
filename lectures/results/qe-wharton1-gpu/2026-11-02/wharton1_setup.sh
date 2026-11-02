#!/bin/bash
# Set-up for wharton1_run.sh on qe-wharton1-gpu, under econ-ai's machine lock, which the caller takes:
# ~/bench/benchmarks at 5fbf033 with Numba added, a quick check of the problem-type suite on every
# backend (kept out of the results), and ~/bench/lectures-python for lecture-python.myst at 087518ec0,
# with the T4 run's exact packages. Bytecode is compiled at install, so no run pays for it.
set -eu
export PATH=$HOME/.local/bin:$PATH
[ -d ~/econ-ai/.lock-machine ] || { echo "take the lock first"; exit 1; }

cd ~/bench/benchmarks
git fetch -q origin
git checkout -q --detach 5fbf033
git log -1 --format='benchmarks at %h: %s'
uv pip install --compile-bytecode --python .venv/bin/python numba scipy -c problems/constraints.txt
.venv/bin/python -c "import jax, numba, numpy, scipy; print('jax', jax.__version__, 'numba', numba.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, jax.devices())"
rm -rf ~/bench/smoke/problems
.venv/bin/python problems/run_suite.py --quick --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu \
  --gpus 0 --host qe-wharton1-gpu --out ~/bench/smoke/problems
.venv/bin/python problems/summarise.py ~/bench/smoke/problems

L=~/bench/lectures-python
mkdir -p $L/lectures
cp lectures/whole/lectures_set.sh lectures/run_lecture.py $L/
cp lectures/whole/lectures-python.txt $L/lectures.txt
base=https://raw.githubusercontent.com/QuantEcon/lecture-python.myst/087518ec0d1a73807fb7cd77c6f7b59d882fb3bc/lectures
for lec in $(cat $L/lectures.txt); do curl -fsSL -o $L/lectures/$lec.md $base/$lec.md; done
echo "$(ls $L/lectures | wc -l) lectures fetched"
[ -x $L/.venv/bin/python ] || uv venv -q --python 3.12 $L/.venv
uv pip install --compile-bytecode --python $L/.venv/bin/python \
  -r lectures/results/runson-g4dn.2xlarge/2026-11-01/lecture-python/environment.txt
uv pip freeze --python $L/.venv/bin/python > $L/environment.txt
cd $L
JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c "import jax; print(jax.devices())"
JAX_PLATFORMS=cpu .venv/bin/python -c "import jax; print(jax.devices())"
echo SETUP OK
