#!/bin/bash
# Phase 1 of QuantEcon/benchmarks#1 on qe-wharton1-gpu: the full precision matrix from main
cd ~/bench/benchmarks
H=qe-wharton1-gpu
echo "== start $(date -u +%FT%TZ) at $(git log -1 --format=%h)"; uptime; nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
.venv/bin/python precision/run_matrix.py --host $H --devices cpu,gpu,gpu:2 --gpus 0,1
echo "== main matrix done $(date -u +%T)"
.venv/bin/python precision/run_matrix.py --host $H --devices cpu --prefix "numactl --interleave=all" --label interleave
echo "== interleave done $(date -u +%T)"
.venv-cuda12/bin/python precision/run_matrix.py --host $H --devices cpu,gpu,gpu:2 --gpus 0,1 --label cuda12
echo "== cuda12 done $(date -u +%T)"
echo ALLDONE
