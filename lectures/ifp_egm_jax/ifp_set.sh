#!/bin/bash
# ifp_egm_jax: the JAX solve and the EGM operator, card 0b:00.0 and host CPU, both precisions; then as used in float64
cd ~/bench/lectures
run() { # platform, then the rest of the command
  local plat=$1; shift
  if [ $plat = cuda ]; then env JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg "$@"
  else env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg "$@"; fi
}
for rep in 1 2 3; do for plat in cuda cpu; do for d in float64 float32; do
  run $plat timeout 1500 .venv-ifp/bin/python ifp_check.py ifp_egm_jax.md --dtype $d --sizes 200 --out ifp/check/$plat-$d-$rep.json
  echo "$(date +%T) check $plat $d $rep (exit $?)"
done; done; done
for rep in 1 2 3; do for plat in cuda cpu; do for d in float64 float32; do
  run $plat timeout 1500 .venv-ifp/bin/python ifp_scale.py ifp_egm_jax.md --dtype $d --sizes 200,1000,5000,20000,50000 --out ifp/scale/$plat-$d-$rep.json
  echo "$(date +%T) scale $plat $d $rep (exit $?)"
done; done; done
for rep in 1 2 3; do for plat in cuda cpu; do
  ./ifp_one.sh $plat float64 ifp/as-used/$plat-float64-$rep.json 1800
  nvidia-smi --id=0 --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader
done; done
echo ALLDONE
