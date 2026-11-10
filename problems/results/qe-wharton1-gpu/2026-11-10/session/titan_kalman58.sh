#!/bin/bash
# QuantEcon/benchmarks#58's TITAN half on qe-wharton1-gpu (Matt, 2026-11-10: "Agree"): P7's five
# sweeps (batch, state, batch20, batch50, batch100), jax-gpu only, both precisions, on the card at
# 0b:00.0 (the suite's --gpus 0, in PCI order), from the runs checkout at #60's merge commit and its
# venv (jax[cuda13] 0.12.0.1), as on 2026-11-02. `--quick` first, kept apart in quick/. Under
# econ-on-ai-hardware's lock, which the caller takes and this releases on exit. gpu.csv samples
# both cards every 10 s.
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
B=~/bench/benchmarks
OUT=~/bench/kalman58/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT/session"
cp "$0" "$OUT/session/"
log() { echo "== $* $(date -u +%T)" | tee -a "$OUT/run.log"; }
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,user,stat,pcpu,comm --sort=-pcpu | head -8
}
state before > "$OUT/state.txt"
( while sleep 10; do
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  done >> "$OUT/gpu.csv" ) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> "$OUT/state.txt"' EXIT
cd "$B"
{ git log -1 --format='benchmarks at %h: %s'; git status --porcelain -- problems ':!problems/results'
  nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version --format=csv,noheader
  .venv/bin/python -c "import jax, jaxlib, numpy; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__, '| numpy', numpy.__version__)"
  .venv/bin/python -c "import importlib.metadata as m; [print(d.metadata['Name'], d.version) for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()) if d.metadata['Name'].lower().startswith(('jax', 'nvidia'))]"
  sha256sum "$0"; } > "$OUT/setup.txt" 2>&1
suite() {  # suite <part> <arguments...>
  local part=$1; shift
  log "$part start"
  .venv/bin/python problems/run_suite.py --problems kalman --backends jax-gpu --gpus 0 \
    --host qe-wharton1-gpu "$@" >> "$OUT/run.log" 2>&1
  log "$part done, exit $?"
}
suite quick --quick --sweeps batch20,batch50,batch100 --label kalman-grid-quick --out "$OUT/quick"
suite full --sweeps batch,state,batch20,batch50,batch100 --timeout 1800 --label kalman-grid --out "$OUT/results"
log done
