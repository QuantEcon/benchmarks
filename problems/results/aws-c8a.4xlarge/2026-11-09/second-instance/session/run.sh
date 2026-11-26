#!/bin/bash
# On the second c8a.4xlarge: record the machine, make the problems' venv with numpy and Numba at
# the harness's pins, then run the one-core probe and the triad. Run from the clone, by start.sh.
set -uo pipefail
O=~/out; mkdir -p $O
python3 ~/clock_sampler.py $O/clock.jsonl ~/run.log 2 &
{ cat /etc/os-release; uname -a; lscpu; echo "SMT active: $(cat /sys/devices/system/cpu/smt/active 2>/dev/null)"
  echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
  numactl --hardware 2>/dev/null || echo "numactl not installed"; grep -E "MemTotal|MemAvailable" /proc/meminfo; } > $O/machine.txt
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv venv -q -p 3.12 .venv-problems && source .venv-problems/bin/activate && uv pip install -q numpy numba -c problems/constraints.txt || exit 1
uv pip freeze > $O/environment.txt
echo "== $(date -u +%T) one core's probe"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 0 python ~/bw_probe.py $O/bandwidth.jsonl
echo "== $(date -u +%T) triad"
NUMBA_NUM_THREADS=16 python ~/stream_triad.py $O/triad.jsonl
