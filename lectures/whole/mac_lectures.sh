#!/bin/bash
# QuantEcon/benchmarks#4 on qe-mini1's CPU: every lecture as used, three times, with the agents and the
# console's animated wallpaper paused (SIGSTOP) and resumed (SIGCONT) on exit, even after a failure.
cd ~/bench/lectures-all
state() {
  echo "== $1: $(date '+%F %T %Z')"; uptime
  ps -Ao pid,stat,pcpu,comm -r | head -10
  pmset -g therm | grep -v '^Note'; pmset -g | grep lowpowermode; pmset -g batt | head -1
}
# Anchored to the start of each target's command line, so no shell that mentions them is paused
RC='^claude remote-control'
PAUSED="$(pgrep -f "$RC") $(pgrep -f '^[^ ]*/claude --print --sdk-url') \
  $(pgrep -f '^/System/Library/ExtensionKit/Extensions/WallpaperAerialsExtension') \
  $(pgrep -f '^/System/Library/Frameworks/VideoToolbox.framework/.*/VTDecoderXPCService')"
resume() {
  kill -CONT $PAUSED 2>/dev/null; echo "== resumed: $PAUSED at $(date '+%T')"
  for p in $(pgrep -f "$RC"); do
    case "$(ps -o stat= -p $p)" in *+*) ;; *) tmux send-keys -t rc fg Enter ;; esac
  done
  state after >> state.txt
}
trap resume EXIT
state before > state.txt
echo "== pausing: $PAUSED" | tee -a state.txt
kill -STOP $PAUSED
sleep 30
state paused >> state.txt
./lectures_set.sh results 3 cpu
