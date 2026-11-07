#!/bin/bash
# chain.sh: the problem-type suite's CPU backends on qe-wharton2's M5 Pro, for QuantEcon/benchmarks#5. In screen
# `pbj-m5pro` under caffeinate: record the machine, set up, a --quick check, then every size in both precisions.
# Each step runs through step.sh, and the chain stops if a step before the full run fails.
d=~/bench/pbj-m5pro
S=$d/step.sh
H=qe-wharton2
ok() { read rc w < $d/$1.done; [ "$rc" = 0 ] || { echo "$1 exited $rc: stopping"; echo stopped > $d/chain.done; exit 1; }; }
bash $S machine bash $d/machine.sh; ok machine
bash $S setup bash $d/setup.sh; ok setup
bash $S quick .venv/bin/python problems/run_suite.py --host $H --quick; ok quick
bash $S state-before bash $d/state.sh
bash $S full .venv/bin/python problems/run_suite.py --host $H
bash $S state-after bash $d/state.sh
# Then QuantEcon/project-benchmarks-jax#48's probe in the same window (Matt, 2026-11-06: "You may queue work
# for M5PRO"; econ-ai agreed 1 h 50 min), whatever the full run's exit, since its steps record their own. It is
# pinned to QuantEcon/benchmarks#44's merge commit, which the M6 ran too, so that both Macs run the same probe
export PROBE_REF=3806db6
bash $S probe48-setup bash $d/probe48_setup.sh
read rc w < $d/probe48-setup.done
if [ "$rc" = 0 ]; then
  bash $S probe48 bash $d/probe48.sh
  bash $S state-after-probe bash $d/state.sh
fi
echo done > $d/chain.done
