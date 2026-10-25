#!/bin/bash
# QuantEcon/benchmarks#1 and #4 on qe-wharton2's M5 Pro CPU: the precision matrix, then the
# lecture-jax set three times, from main. A guest machine: nothing is paused, and the state
# samples record what else ran, including the endpoint agents and how long the console was idle.
# git comes from ~/bench/tools, since the machine has no command-line tools and /usr/bin/git
# would ask the console to install them.
export PATH=$HOME/bench/tools/env/bin:$HOME/.local/bin:$PATH
H=qe-wharton2
DAY=$(date -u +%F)
OUT=~/bench/run-$DAY; mkdir -p $OUT
LEC=results/$H/$DAY/whole
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  echo "console idle: $(ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {printf "%d s", $NF/1e9; exit}')"
  ps -Ao pid,stat,pcpu,user,comm -r | head -10
  pmset -g therm | grep -v '^Note'; pmset -g | grep -E 'powermode|lowpowermode'; pmset -g batt | sed -n 2p
  ioreg -r -k AppleClamshellState -d 4 | grep '"AppleClamshellState"'; memory_pressure | tail -1
}
sample() { while sleep 600; do state sample >> $OUT/state.txt; done; }
state before > $OUT/state.txt
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> $OUT/state.txt' EXIT

cd ~/bench/benchmarks
echo "== precision start $(date -u +%FT%TZ) at $(git log -1 --format=%h)"
.venv/bin/python precision/run_matrix.py --host $H --devices cpu
echo "== precision done $(date -u +%T)"
state "between" >> $OUT/state.txt

cd ~/bench/lectures-all
echo "== lectures start $(date -u +%FT%TZ)"
./lectures_set.sh $LEC 3 cpu > $OUT/lectures-set.log 2>&1
uv pip freeze -p .venv/bin/python > $LEC/environment.txt
cp $OUT/lectures-set.log $LEC/
echo "== lectures done $(date -u +%T)"
echo ALLDONE
