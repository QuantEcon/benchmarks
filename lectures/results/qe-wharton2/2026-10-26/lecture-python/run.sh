#!/bin/bash
# QuantEcon/benchmarks#4: the 33 lecture-python.myst lectures that import JAX (087518ec0), as written,
# three times, on the qe-wharton2 CPU. A guest machine: nothing is paused; the state samples record
# what else ran. git from ~/bench/tools, as in wharton2_run.sh.
export PATH=$HOME/bench/tools/env/bin:$HOME/.local/bin:$PATH
cd ~/bench/lectures-python
O=results/qe-wharton2/2026-10-26/lecture-python
mkdir -p $O
state() {
  echo "== $1: $(date "+%F %T %Z") ($(date -u +%T) UTC)"; uptime
  echo "console idle: $(ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {printf "%d s", $NF/1e9; exit}')"
  ps -Ao pid,stat,pcpu,user,comm -r | head -10
  pmset -g therm | grep -v "^Note"; pmset -g | grep -E "powermode|lowpowermode"; pmset -g batt | sed -n 2p
  ioreg -r -k AppleClamshellState -d 4 | grep '"AppleClamshellState"'; memory_pressure | tail -1
}
sample() { while sleep 600; do state sample >> $O/state.txt; done; }
state before > $O/state.txt
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> $O/state.txt' EXIT
echo "== start $(date -u +%FT%TZ) at $(git -C ~/bench/benchmarks log -1 --format=%h); lecture-python.myst at 087518ec0" >> $O/state.txt
cp environment.txt $O/environment.txt
ROOT=$HOME/bench/lectures-python ./lectures_set.sh $O 3 cpu > $O/lectures-set.log 2>&1
