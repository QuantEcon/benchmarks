#!/bin/bash
# tpu_go.sh <ip>: open the SSH master to the TPU VM, send the probe, its setup and the constraints, start them in tmux.
set -eu
IP=$1; T=ubuntu@$IP; D=$HOME/work/quantecon/_runs/tpu-38; CM=$D/cmt
bash $HOME/work/quantecon/_runs/benchmarks-jax-34/wait_ssh.sh $T $CM 'echo READY; hostname'
S="ssh -n -o ControlPath=$CM $T"
scp -o ControlPath=$CM -q $D/setup_and_probe.sh $D/tpu_f64_probe.py $D/constraints.txt "$T:"
$S 'sha256sum tpu_f64_probe.py; command -v tmux || echo "no tmux"'
$S 'if command -v tmux > /dev/null; then tmux new -d -s probe "bash ~/setup_and_probe.sh"; else nohup setsid bash ~/setup_and_probe.sh > ~/run.log 2>&1 < /dev/null & fi; sleep 2; date -u "+started %F %T UTC"'
