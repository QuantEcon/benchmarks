#!/bin/bash
# On qe-wharton1-gpu, under project-econ-on-ai-hardware's machine lock, which the caller has taken and
# this releases on exit: ar1_bayes again, three times on the card at 0b:00.0 and then the Threadripper,
# as run.sh ran every lecture. Its runs there failed from cell 8 on, since PyTensor's C compile needs
# Python's headers and Ubuntu's python3.12 has none installed. This uses a venv, .venv-h, on uv's own
# CPython 3.12.3, which ships its headers, with the same packages (its `uv pip freeze` must match
# environment.txt), and a fresh PyTensor cache, so the first run pays its compile as the first would.
set -u
export PATH=$HOME/.local/bin:$PATH
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
D=$(date -u +%F)
R=~/bench/run-$D-ar1
mkdir -p $R
state() {
  echo "== $1: $(date "+%F %T %Z") ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,stat,pcpu,comm --sort=-pcpu | head -8
}
sample() { while sleep 600; do state sample >> $R/state.txt; done; }
state before > $R/state.txt
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> $R/state.txt; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> $R/state.txt' EXIT

L=~/bench/lectures-python
O=$R/lecture-python
mkdir -p $O
uv pip freeze --python $L/.venv-h/bin/python > $O/environment.txt 2>/dev/null
cmp -s $L/environment.txt $O/environment.txt || { echo "== the packages differ; not running" >> $R/state.txt; exit 1; }
echo ar1_bayes > $R/list.txt
export PYTENSOR_FLAGS=base_compiledir=$R/pytensor
echo "== start $(date -u +%FT%TZ); $($L/.venv-h/bin/python -c 'import sys; print(sys.version.split()[0], sys.base_prefix)')" >> $R/state.txt
cd $L && ROOT=$L CAP=7200 PY=.venv-h/bin/python LIST=$R/list.txt ./lectures_set.sh $O 3 cuda cpu > $O/lectures-set.log 2>&1
s=$?; echo "== ar1_bayes done $(date -u +%T), exit $s" >> $R/state.txt
find $O -name '*.err' -empty -delete
echo ALLDONE >> $R/state.txt
