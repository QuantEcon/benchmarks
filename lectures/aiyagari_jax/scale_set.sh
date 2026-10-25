#!/bin/bash
# aiyagari_jax operators at larger grids: card 0b:00.0 and the host CPU, both precisions
cd ~/bench/lectures
for plat in cuda cpu; do
  for dtype in float64 float32; do
    env JAX_PLATFORMS=$plat CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg \
      timeout 1800 .venv/bin/python aiyagari_scale.py aiyagari_jax.md --dtype $dtype \
      --sizes 200,1000,2000,5000,10000 --out scale/$plat-$dtype.json
    echo "$(date +%T) finished $plat $dtype (exit $?)"
    nvidia-smi --id=0 --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader
  done
done
echo ALLDONE
