#!/bin/bash
# QuantEcon/benchmarks#4 on qe-mini1's CPU: every lecture as used, three times, with the console's
# animated wallpaper and its video decoder (VTDecoderXPCService) paused (SIGSTOP) and resumed
# (SIGCONT) on exit, even after a failure. qe-mini1's
# Claude sessions are left running: stopped for long, they lose Remote Control and exit, so agree a
# quiet window with them before the run instead.
# Arguments go to lectures_set.sh, `results 3 cpu` by default; so do DTYPES and LIST.
cd ~/bench/lectures-all
state() {
  echo "== $1: $(date '+%F %T %Z')"; uptime
  ps -Ao pid,stat,pcpu,comm -r | head -10
  pmset -g therm | grep -v '^Note'; pmset -g | grep lowpowermode; pmset -g batt | head -1
}
# Anchored to the start of each target's command line, so no shell that mentions them is paused
PAUSED="$(pgrep -f '^/System/Library/ExtensionKit/Extensions/WallpaperAerialsExtension') \
  $(pgrep -f '^/System/Library/Frameworks/VideoToolbox.framework/.*/VTDecoderXPCService')"
resume() {
  kill -CONT $PAUSED 2>/dev/null; echo "== resumed: $PAUSED at $(date '+%T')"
  state after >> state.txt
}
trap resume EXIT
state before > state.txt
echo "== pausing: $PAUSED" | tee -a state.txt
kill -STOP $PAUSED
sleep 30
state paused >> state.txt
if [ $# -gt 0 ]; then ./lectures_set.sh "$@"; else ./lectures_set.sh results 3 cpu; fi
