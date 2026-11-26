#!/bin/bash
# chain34-mi300x.sh: #34 on the MI300X: both quick checks, then full.sh (precision, problems, highest, repro) if both pass.
H=digitalocean-gpu-mi300x1-192gb
export ROCM_MAJOR=10
S=~/session/step.sh
ok() { read rc w < ~/session/$1.done; [ "$rc" = 0 ] || { echo "$1 exited $rc: stopping"; echo stopped > ~/session/chain.done; exit 1; }; }
bash $S precision-quick bash precision/run_by_hand.sh $H cpu,gpu --quick; ok precision-quick
bash $S problems-quick bash problems/run_by_hand.sh $H --quick; ok problems-quick
bash ~/session/full.sh $H
echo done > ~/session/chain.done
