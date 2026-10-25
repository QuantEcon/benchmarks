#!/bin/bash
# Phase 0 of QuantEcon/benchmarks#1 on qe-mini1: the full precision matrix on the M6 CPU, from main,
# with the agents and the animated wallpaper paused (SIGSTOP) and resumed (SIGCONT) on exit
cd ~/bench/benchmarks
OUT=~/bench/phase0; mkdir -p $OUT
state() {
  echo "== $1: $(date "+%F %T %Z")"; uptime
  ps -Ao pid,stat,pcpu,comm -r | head -10
  pmset -g therm | grep -v "^Note"; pmset -g | grep lowpowermode; pmset -g batt | head -1
}
PAUSED="$(pgrep -f "claude remote-control") $(pgrep -f "claude --print --sdk-url") $(pgrep -f WallpaperAerialsExtension) $(pgrep -f VTDecoderXPCService)"
resume() {
  kill -CONT $PAUSED 2>/dev/null; echo "== resumed: $PAUSED at $(date +%T)"
  for p in $(pgrep -f "claude remote-control"); do
    case "$(ps -o stat= -p $p)" in *+*) ;; *) tmux send-keys -t rc fg Enter ;; esac
  done
  state after >> $OUT/state.txt
}
trap resume EXIT
state before > $OUT/state.txt
echo "== pausing: $PAUSED" | tee -a $OUT/state.txt
kill -STOP $PAUSED
sleep 30
state paused >> $OUT/state.txt
echo "== start $(date -u +%FT%TZ) at $(git log -1 --format=%h)"
.venv/bin/python precision/run_matrix.py --host qe-mini1 --devices cpu
echo ALLDONE
