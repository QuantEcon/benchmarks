#!/bin/bash
# probe_rocm.sh: #38's tpu_f64_probe.py, verbatim, on the MI300X in the precision run's environment, with XLA's defaults.
# JAX_PLATFORMS lists cpu too, because the probe compares each result with jax.devices("cpu").
source ~/benchmarks/.venv-phase2/bin/activate
export PATH="$HOME/.local/bin:$PATH"
cd ~/extras
sha256sum tpu_f64_probe.py | tee probe-sha256.txt
uv pip freeze > probe-environment.txt
JAX_PLATFORMS=rocm,cpu HIP_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  timeout 15m python -u tpu_f64_probe.py 2>&1 | tee probe-mi300x.txt
exit ${PIPESTATUS[0]}
