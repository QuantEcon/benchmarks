#!/bin/bash
# One ifp_egm_jax run: $1 = cuda|cpu, $2 = float64|float32, $3 = output name
cd ~/bench/lectures
if [ $1 = cuda ]; then DEV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0"; else DEV=""; fi
env JAX_PLATFORMS=$1 $DEV XLA_PYTHON_CLIENT_PREALLOCATE=false timeout ${4:-3600} .venv-ifp/bin/python run_lecture.py ifp_egm_jax.md \
  --dtype $2 --out $3 --env-info ~/bench/benchmarks/precision \
  --record "jax_time_without_compile" "numba_time_without_compile" "np.asarray(σ_star_egm_jax).ravel().tolist()" "np.asarray(σ_star_egm_nb).ravel().tolist()" "str(σ_star_egm_jax.dtype)"
echo "$(date +%T) done $1 $2 $3 (exit $?)"
