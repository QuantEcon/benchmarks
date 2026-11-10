#!/bin/bash
# Does a `.noindex` output folder stop mediaanalysisd waking when XLA writes its dump? qe-mini1's M6,
# macOS 27.2.1 (26B131), 2026-11-10, at Matt's go-ahead ("Please proceed"), in a quiet window agreed
# with the sessions here, which stay running. On 2026-11-06 and 11-09, mediaanalysisd woke within ~3 s
# of #48's cliff step, which writes 32 XLA dump files. Here the same step (probe.py from QuantEcon/
# benchmarks at ee5ddfa: `cliff <variant> 16000,18000` for its four variants, as run.sh runs them,
# about 4 s in all) runs three ways, three rounds, alternating:
#   plain    the dump into an ordinary folder under ~/bench, as run.sh does
#   noindex  the dump into a folder whose name ends in .noindex
#   nodump   no dump at all, so the step's compute and memory alone
# A 45 s rest before each lets the daemons settle. samples.txt has, every second, the CPU of the
# media and Spotlight daemons; state.txt the machine before and after.
set -u
B=~/work/quantecon/benchmarks
PY=~/bench/benchmarks/.venv/bin/python
OUT=~/bench/noindex-$(date -u +%F)-qe-mini1
mkdir -p "$OUT/plain" "$OUT/dump.noindex"
git -C "$B" show ee5ddfa:precision/probes/mac_float64/probe.py > "$OUT/probe.py"
cp "$0" "$OUT/"
shasum -a 256 "$OUT/probe.py" > "$OUT/sha256.txt"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC), console $(stat -f %Su /dev/console)"; uptime
  ps -Ao pid,pcpu,pmem,comm -r | head -12
  memory_pressure | tail -1
}
D='mediaanalysisd|photoanalysisd|ANECompilerService|mds_stores|mdworker|/mds$|mdbulkimport|corespotlightd|fseventsd'
sample() {
  while sleep 1; do
    ps -Ao pcpu=,comm= | grep -E "$D" | awk -v t="$(date -u +%T)" '$1 > 0 {sub(".*/", "", $2); printf "%s %s %s\n", t, $1, $2}'
  done >> "$OUT/samples.txt"
}
log() { echo "== $* $(date -u +%T)" | tee -a "$OUT/run.log" >> "$OUT/samples.txt"; }
{ sw_vers; state before; } > "$OUT/state.txt"
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; echo "== ended $(date -u +%T)" >> "$OUT/state.txt"' EXIT
log "rest"; sleep 45
for round in 1 2 3; do
  for way in plain noindex nodump; do
    case $way in
      plain) DUMP="$OUT/plain/xla-dump-$round" ;;
      noindex) DUMP="$OUT/dump.noindex/xla-dump-$round" ;;
      nodump) DUMP="" ;;
    esac
    log "start $way $round"
    for v in default no-reduce-fusion add rowmax; do
      PROBE_DUMP="${DUMP:+$DUMP/$v}" PROBE_HOST=qe-mini1 "$PY" -I "$OUT/probe.py" cliff $v 16000,18000 \
        >> "$OUT/cliff-$way.jsonl" 2>> "$OUT/run.log" || echo "failed: $way $round $v" >> "$OUT/run.log"
    done
    log "end $way $round ($( [ -n "$DUMP" ] && find "$DUMP" -type f | wc -l | tr -d ' ' || echo 0) dump files)"
    sleep 45
  done
done
log "done"
