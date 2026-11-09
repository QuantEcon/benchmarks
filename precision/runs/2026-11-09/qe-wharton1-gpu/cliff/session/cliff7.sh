#!/bin/bash
# QuantEcon/project-benchmarks-jax#7 on qe-wharton1-gpu's Threadripper 2990WX, 2026-11-09 (Matt: "Agree
# please run"): does the float64 row reduction's cliff past ~2 GiB come from YNNPACK's reduce, as on
# both Macs? It runs #48's probe's cliff step, probe.py from QuantEcon/benchmarks at ba58764 (unchanged
# since 3806db6), in its four variants, at 16,000, 18,000 and 20,000, at the default placement, on the
# CPU only. Run in tmux under econ-on-ai-hardware's lock, which the launcher takes after their OK; this
# script releases it at the end. Both cards stay idle.
set -u
HERE=~/bench/cliff7
PY=~/bench/benchmarks/.venv/bin/python
OUT=$HERE/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  ps -eo pid,pcpu,pmem,comm --sort=-pcpu | head -10
  nvidia-smi --query-gpu=pci.bus_id,memory.used,utilization.gpu --format=csv,noheader
}
log() { echo "== $* $(date -u +%T)" | tee -a "$OUT/run.log"; }
{ uname -a; lscpu | grep -E 'Model name|^CPU\(s\)|Thread|NUMA node[0-9]'
  echo "boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>&1)"
  echo "driver: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>&1), max $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>&1) kHz"
  free -g | head -2
  "$PY" -c 'import jax, jaxlib, numpy; print("jax", jax.__version__, "| jaxlib", jaxlib.__version__, "| numpy", numpy.__version__)'
  sha256sum "$HERE/probe.py"; } > "$OUT/machine.txt" 2>&1
state before > "$OUT/state.txt"
trap 'state after >> "$OUT/state.txt"; rmdir ~/econ-ai/.lock-machine && log "lock released"' EXIT
NFAIL=0
for v in default no-reduce-fusion add rowmax; do
  log cliff $v
  PROBE_DUMP="$OUT/xla-dump/$v" "$PY" "$HERE/probe.py" cliff $v 16000,18000,20000 >> "$OUT/cliff.jsonl" 2>> "$OUT/run.log" \
    || { NFAIL=$((NFAIL + 1)); log "failed: cliff $v"; }
done
# As run.sh does: of the dump, only the optimised modules say which ops YNNPACK took
find "$OUT/xla-dump" -type f ! -name '*after_optimizations.txt' -delete
log "done, $NFAIL failed"
