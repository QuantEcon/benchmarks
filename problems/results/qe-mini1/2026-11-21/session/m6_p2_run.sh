#!/bin/bash
# On qe-mini1 (the M6), in a quiet window agreed with every session there (Q26, 2026-11-21), with
# the sessions left running (Matt, 2026-11-01): QuantEcon/benchmarks#5's Phase 2 problem types,
# every size, on the four CPU backends, at ~/bench/benchmarks's checkout. As m6_run.sh, only the
# console's animated wallpaper and macOS's Photos analysis are paused, and only while busy.
# cputimes.txt records every process's CPU time at the start and the end, so the sessions' share of
# the machine can be stated; samples.txt has the busiest processes each minute. The run folder is a
# .noindex one, so Spotlight leaves its files alone (QuantEcon/benchmarks#59).
# Run as: caffeinate -i -s bash m6_p2_run.sh
set -u
PROBLEMS=stationary,hpi,time_iter,mle,mle_hand,mle_onepass,mle_blocked
D=$(date -u +%F)
R=~/bench/run-$D-m6-p2.noindex
mkdir -p $R
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -Ao pid,stat,pcpu,comm -r | head -10
  pmset -g therm | grep -v '^Note'; pmset -g | grep lowpowermode; pmset -g batt | head -1
}
cputimes() { echo "== $1: $(date -u +%FT%TZ)"; ps -Ao pid,ppid,time,comm; }
sample() { while sleep 60; do { echo "== $(date -u +%T)"; uptime; ps -Ao pid,pcpu,comm -r | sed -n 2,7p; } >> $R/samples.txt; done; }

# The wallpaper and its video decoder only, and only the processes busy playing it, above 1% of a
# core: anchored to the start of each command line, as in mac_lectures.sh
busy() { for p in "$@"; do ps -o pcpu= -p $p | awk -v p=$p '$1 > 1 {print p}'; done; }
PAUSED=$(busy $(pgrep -f '^/System/Library/ExtensionKit/Extensions/WallpaperAerialsExtension'))
[ -n "$PAUSED" ] && PAUSED="$PAUSED $(busy $(pgrep -f '^/System/Library/Frameworks/VideoToolbox.framework/.*/VTDecoderXPCService'))"
PAUSED=$(echo $PAUSED)
# macOS's Photos analysis, checked before the run and paused if busy, as in m6_run.sh
photos() {
  local p; p=$(busy $(pgrep -f '^/System/Library/PrivateFrameworks/(Media|Photo)Analysis.framework/.*/(mediaanalysisd|photoanalysisd)$'))
  p=$(echo $p)
  [ -z "$p" ] && return
  kill -STOP $p; PAUSED=$(echo $PAUSED $p)
  echo "== pausing Photos analysis: $p at $(date -u +%T)" >> $R/state.txt
}
finish() {
  kill $SAMPLER 2>/dev/null
  if [ -n "$PAUSED" ]; then kill -CONT $PAUSED; echo "== resumed: $PAUSED at $(date '+%T')" >> $R/state.txt; fi
  cputimes end >> $R/cputimes.txt
  state after >> $R/state.txt
  echo "== exit $(date -u +%FT%TZ)" >> $R/state.txt
}
SAMPLER=""
trap finish EXIT
state before > $R/state.txt
cputimes start > $R/cputimes.txt
if [ -n "$PAUSED" ]; then
  echo "== pausing the wallpaper: $PAUSED" >> $R/state.txt
  kill -STOP $PAUSED
else
  echo "== the wallpaper is not playing" >> $R/state.txt
fi
photos
[ -n "$PAUSED" ] && { sleep 30; state paused >> $R/state.txt; }
sample & SAMPLER=$!

cd ~/bench/benchmarks
echo "== start $(date -u +%FT%TZ) at $(git log -1 --format=%h)" >> $R/state.txt
echo "== problems start $(date -u +%T): $PROBLEMS" >> $R/state.txt
.venv/bin/python problems/run_suite.py --problems $PROBLEMS --host qe-mini1 --out $R/results \
  > $R/problems.log 2>&1
s=$?; echo "== problems done $(date -u +%T), exit $s" >> $R/state.txt
echo ALLDONE >> $R/state.txt
