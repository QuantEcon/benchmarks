#!/bin/bash
# aiyagari_jax as used: card 0b:00.0 and the host CPU, float64 and float32, three of each, interleaved
cd ~/bench/lectures
REC=(--record "K_star" "K_vals.tolist()" "np.asarray(σ_star).ravel().tolist()" "str(household.a_grid.dtype)")
for rep in 1 2 3; do
  for plat in cuda cpu; do
    for dtype in float64 float32; do
      if [ $plat = cuda ]; then
        env JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
          .venv/bin/python run_lecture.py aiyagari_jax.md --dtype $dtype --out runs/$plat-$dtype-$rep.json --env-info ~/bench/benchmarks/precision "${REC[@]}"
      else
        env JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
          .venv/bin/python run_lecture.py aiyagari_jax.md --dtype $dtype --out runs/$plat-$dtype-$rep.json --env-info ~/bench/benchmarks/precision "${REC[@]}"
      fi
      echo "$(date +%T) done $plat $dtype $rep (exit $?)"
      nvidia-smi --id=0 --query-gpu=clocks.sm,temperature.gpu,power.draw,clocks_event_reasons.active --format=csv,noheader
    done
  done
done
echo ALLDONE
