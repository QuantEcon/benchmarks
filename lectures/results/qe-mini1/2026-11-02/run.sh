#!/bin/bash
# On qe-mini1 (the M6), with its long-running Claude sessions left running (Matt, 2026-11-02): they
# keep a quiet window, with no builds, tests or Python runs here. Only the console's animated
# wallpaper and macOS's Photos analysis are paused, and only while they are busy. Nothing else is.
#  0. A check against 2026-10-25's runs, which were made with the agents paused: precision's Phase 0
#     matrix as it ran then (CPU, every workload, both precisions) and six lecture-jax lectures whose
#     three runs then were within 2% of each other, three times each.
#  1. QuantEcon/benchmarks#5's problem-type suite at 5fbf033, every size, on the CPU backends.
#  2. QuantEcon/benchmarks#4: lecture-python.myst's 33 JAX lectures at 087518ec0, as written, three
#     times, on the CPU, with the M5 Pro run's exact packages.
# cputimes.txt records every process's CPU time at the start and the end, so the sessions' share of
# the machine can be stated; samples.txt has the busiest processes each minute.
# Run as: START="YYYY-MM-DD HH:MM" caffeinate -i -s bash m6_run.sh   (START is local time)
set -u
if [ -n "${START:-}" ]; then
  t=$(date -j -f '%Y-%m-%d %H:%M' "$START" +%s)
  while [ "$(date +%s)" -lt "$t" ]; do sleep 30; done
fi
D=$(date -u +%F)
R=~/bench/run-$D-m6
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
# macOS's Photos analysis, mediaanalysisd and photoanalysisd, which start by themselves on a Mac
# nobody is using (Matt, 2026-11-02): checked before each step, and paused if busy. The video
# services mediaanalysisd drives go quiet once it stops, so they are left alone.
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
echo "== start $(date -u +%FT%TZ) at $(git -C ~/bench/benchmarks log -1 --format=%h)" >> $R/state.txt

cd ~/bench/benchmarks
echo "== check: precision start $(date -u +%T)" >> $R/state.txt
.venv/bin/python precision/run_matrix.py --host qe-mini1 --devices cpu --label sessions-running \
  --out $R/check/precision > $R/check-precision.log 2>&1
printf '%s\n' kesten_processes_jax ifp_egm_jax hopenhayn jax_intro_jax markov_asset_jax inventory_dynamics_jax > $R/check-lectures.txt
photos
echo "== check: lectures start $(date -u +%T)" >> $R/state.txt
cd ~/bench/lectures-all && LIST=$R/check-lectures.txt ./lectures_set.sh $R/check/lecture-jax 3 cpu > $R/check-lectures.log 2>&1

cd ~/bench/benchmarks
photos
echo "== problems start $(date -u +%T)" >> $R/state.txt
.venv/bin/python problems/run_suite.py --host qe-mini1 > $R/problems.log 2>&1
s=$?; echo "== problems done $(date -u +%T), exit $s" >> $R/state.txt

L=~/bench/lectures-python
O=$L/results/qe-mini1/$D/lecture-python
mkdir -p $O
cp $L/environment.txt $O/environment.txt
photos
echo "== lectures start $(date -u +%T); lecture-python.myst at 087518ec0" >> $R/state.txt
cd $L && ROOT=$L ./lectures_set.sh $O 3 cpu > $O/lectures-set.log 2>&1
s=$?; echo "== lectures done $(date -u +%T), exit $s" >> $R/state.txt
find $O -name '*.err' -empty -delete
echo ALLDONE >> $R/state.txt
