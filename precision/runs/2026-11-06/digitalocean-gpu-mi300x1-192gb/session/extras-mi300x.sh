#!/bin/bash
# extras-mi300x.sh: Matt's additions on the MI300X, after the chain: (1) the precision set's GPU cells with XLA's
# autotuner off, label autotune0 as in benchmarks#21; (2) the matmul at n = 16,384, same flag and label;
# (3) #38's tpu_f64_probe.py, standalone, with XLA's defaults. Then re-pack the precision tarball with
# run_by_hand.sh's own tar line, so it carries the autotune0 rows.
H=digitalocean-gpu-mi300x1-192gb
S=~/session/step.sh
until [ -f ~/session/chain.done ]; do sleep 30; done
cd ~/benchmarks
D=$(date -u +%F)
sha256sum "$H-$D.tgz" > ~/session/precision-tgz-before-extras.sha256
export PATH="$HOME/.local/bin:$PATH"
source .venv-phase2/bin/activate
bash $S autotune0 env XLA_FLAGS=--xla_gpu_autotune_level=0 python precision/run_matrix.py --host $H --devices gpu --gpus 0 --label autotune0
bash $S matmul16k env XLA_FLAGS=--xla_gpu_autotune_level=0 python precision/run_matrix.py --host $H --devices gpu --gpus 0 --workloads matmul --sizes 16384 --label autotune0
bash $S probe bash ~/extras/probe_rocm.sh
tar czf "$H-$D.tgz" "results/$H" "lectures/results/$H" && echo "re-packed $H-$D.tgz" | tee -a ~/session/autotune0.log
sha256sum "$H-$D.tgz" > ~/session/precision-tgz-after-extras.sha256
echo done > ~/session/extras.done
