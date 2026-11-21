#!/bin/bash
# chain.sh: Phase 2's problem types on qe-wharton2's M5 Pro, CPU backends, for QuantEcon/benchmarks#5. In screen
# `pbj-p2` under caffeinate, with PROBLEMS (e.g. stationary,hpi), REF (the commit) and SLOT (1 or 2) from the
# environment. It takes econ-ai's lock, an empty directory, first, and stops at once if someone holds it; it
# releases the lock last, before writing chain<SLOT>.done ("done", "stopped" or "locked").
d=~/bench/pbj-p2
S=$d/step.sh
H=qe-wharton2
LOCK=~/econ-ai/.lock-machine
: "${PROBLEMS:?}" "${REF:?}" "${SLOT:?}"
took=0
finish() { [ "$took" = 1 ] && rmdir "$LOCK"; echo "$1" > $d/chain$SLOT.done; exit 0; }
if mkdir "$LOCK" 2>/dev/null; then took=1; else echo "the lock is held: not starting"; finish locked; fi
ok() { read rc w < $d/$1.done; [ "$rc" = 0 ] || { echo "$1 exited $rc: stopping"; finish stopped; }; }
bash $S machine$SLOT bash $d/machine.sh; ok machine$SLOT
bash $S tools$SLOT bash $d/tools.sh; ok tools$SLOT
bash $S setup$SLOT bash $d/setup.sh "$REF"; ok setup$SLOT
bash $S quick$SLOT .venv/bin/python problems/run_suite.py --host $H --problems "$PROBLEMS" --quick --out $d/quick$SLOT
ok quick$SLOT
bash $S state-before$SLOT bash $d/state.sh
bash $S full$SLOT .venv/bin/python problems/run_suite.py --host $H --problems "$PROBLEMS"
bash $S state-after$SLOT bash $d/state.sh
finish done
