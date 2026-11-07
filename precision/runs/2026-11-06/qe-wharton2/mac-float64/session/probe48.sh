#!/bin/bash
# probe48.sh: the probe from its worktree, twice, then its step phase alone as a control, as on the M6, where
# mediaanalysisd took a core mid-run after the cliff wrote its XLA dump. Each folder's state.txt samples the
# machine every 5 s: redact the console user's and the managed agents' names before publishing it. The output
# goes to ~/bench/pbj-m5pro/probe48-out, so the usual copy-back carries it.
P=~/bench/probe48
O=~/bench/pbj-m5pro/probe48-out
PY=$HOME/bench/benchmarks/.venv/bin/python
MLXPY=$P/mlx-venv/bin/python
cd $P/src
sampled() {  # sampled <folder> <command...>
  local f=$1 s rc; shift; mkdir -p $f
  ( while sleep 5; do echo "== $(date -u +%T)"; ps -Ao pid,pcpu,comm -r | sed -n '2,6p'; done ) > $f/state.txt &
  s=$!
  "$@"; rc=$?
  kill $s; return $rc
}
control() {
  local k SZ
  for k in bellman pv; do
    if [ $k = bellman ]; then SZ=2000,8000,14000; else SZ=1000,4000; fi
    PROBE_HOST=qe-wharton2 $PY precision/probes/mac_float64/probe.py step jax $k $SZ >> $O/stepcheck/step.jsonl
    PROBE_HOST=qe-wharton2 $MLXPY precision/probes/mac_float64/probe.py step mlx $k $SZ >> $O/stepcheck/step.jsonl
  done
}
for r in run1 run2; do
  sampled $O/$r env PROBE_HOST=qe-wharton2 PY=$PY MLXPY=$MLXPY bash precision/probes/mac_float64/run.sh $O/$r \
    || echo "$r: run.sh exited $?"
done
sampled $O/stepcheck control
