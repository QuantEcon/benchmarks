#!/bin/bash
# run_repro.sh: ROCm/jax#895's and #896's reproducers and the argsort check, as #34 gives them
source ~/benchmarks/.venv-phase2/bin/activate
export JAX_PLATFORMS=rocm HIP_VISIBLE_DEVICES=0
mkdir -p ~/repro && cd ~/repro
for n in 895 896; do
  curl -fsSL "https://api.github.com/repos/ROCm/jax/issues/$n" \
    | python -c 'import json, sys; print(json.load(sys.stdin)["body"])' \
    | awk '/^```python/ {f = 1; next} /^```/ {f = 0} f' > "rocm-jax-$n.py"
done
python rocm-jax-895.py 2>&1 | tee rocm-jax-895.txt
python argsort.py 2>&1 | tee argsort.txt
python rocm-jax-896.py 2>&1 | tee rocm-jax-896-default.txt
XLA_FLAGS=--xla_gpu_enable_scatter_determinism_expander=true python rocm-jax-896.py 2>&1 | tee rocm-jax-896-expander.txt
XLA_FLAGS=--xla_gpu_deterministic_ops=true python rocm-jax-896.py 2>&1 | tee rocm-jax-896-deterministic.txt
