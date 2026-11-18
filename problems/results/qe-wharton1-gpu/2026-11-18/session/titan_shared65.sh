#!/bin/bash
# Two jobs on qe-wharton1-gpu's TITAN RTX at 0b:00.0, then #65's sustained load, in one window
# (Matt, 2026-11-18: "Please proceed with TITAN, run all tests"). The plan is PLAN.md beside this.
#   0. trap:   two processes with JAX's defaults (75% preallocated). Does the second start?
#   1. solo:   the problem types' 50 jax-gpu cells, one job (label shared-solo)
#   2. shared: two such jobs started together (shared-a, shared-b)
#   3. solo2:  one job again, to bound drift (shared-solo2)
#   4. busy:   the 9 largest cells while precision/sustained.py holds the card with matmul (neighbour-busy)
#   5. #65:    GPU=0 bash precision/sustained.sh qe-wharton1-gpu
# Steps 1 to 4 give every JAX process 45% of the card: XLA_PYTHON_CLIENT_MEM_FRACTION=0.45, which
# run_suite.py's setdefault keeps, and --memory-fraction 0.45, which skips the same cells in each.
# From ~/bench/benchmarks at f085c6d, with .venv (jax[cuda13] 0.12.0.1) for steps 0 to 4; step 5
# makes .venv-phase2. Under econ-on-ai-hardware's lock, which the caller takes and this releases
# on exit. gpu.csv samples both cards every 5 s until step 5, which logs the card once a second.
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
B=~/bench/benchmarks
OUT=~/bench/shared65/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT/session"
cp "$0" "$OUT/session/"
log() { echo "== $* $(date -u +%FT%TZ)" | tee -a "$OUT/run.log"; }
state() {
  echo "== $1: $(date '+%F %T %Z') ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  ps -eo pid,user,stat,pcpu,comm --sort=-pcpu | head -10
}
sampler() {
  while sleep 5; do
    nvidia-smi --query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  done >> "$OUT/gpu.csv"
}
state before > "$OUT/state.txt"
sampler & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> "$OUT/state.txt"; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> "$OUT/state.txt"' EXIT

cd "$B"
git fetch -q && git checkout -q f085c6d || { log "checkout failed"; exit 1; }
PY=$B/.venv/bin/python
{ git log -1 --format='benchmarks at %h: %s'; git status --porcelain -- problems precision ':!problems/results' ':!precision/runs'
  nvidia-smi --query-gpu=index,pci.bus_id,name,driver_version,memory.total,power.limit --format=csv,noheader
  uname -r; dpkg-query -W libc6 2>/dev/null
  "$PY" -c "import jax, jaxlib, numpy; print('jax', jax.__version__, '| jaxlib', jaxlib.__version__, '| numpy', numpy.__version__)"
  "$PY" -c "import importlib.metadata as m; [print(d.metadata['Name'], d.version) for d in sorted(m.distributions(), key=lambda d: d.metadata['Name'].lower()) if d.metadata['Name'].lower().startswith(('jax', 'nvidia'))]"
  sha256sum "$0"; } > "$OUT/setup.txt" 2>&1

export CUDA_DEVICE_ORDER=PCI_BUS_ID
SHARE=0.45
suite() {  # suite <label> <arguments...>: one job of the jax-gpu problem types on card 0
  local label=$1; shift
  XLA_PYTHON_CLIENT_MEM_FRACTION=$SHARE "$PY" problems/run_suite.py --backends jax-gpu --gpus 0 \
    --host qe-wharton1-gpu --memory-fraction $SHARE --label "$label" "$@" >> "$OUT/$label.log" 2>&1
}

log "0 trap start"
mkdir -p "$OUT/trap"
cat > "$OUT/trap/hold.py" <<'EOF'
import time, jax, jax.numpy as jnp
x = jnp.ones((1024, 1024)); x.block_until_ready()
print("A ready:", jax.devices()[0], "bytes_limit", jax.devices()[0].memory_stats().get("bytes_limit"), flush=True)
time.sleep(60)
print("A done", flush=True)
EOF
cat > "$OUT/trap/second.py" <<'EOF'
import time
t = time.time()
try:
    import jax, jax.numpy as jnp
    d = jax.devices()[0]
    print("B device:", d, "bytes_limit", d.memory_stats().get("bytes_limit"), flush=True)
    jnp.ones((1024, 1024)).block_until_ready(); print("B: 4 MiB ok", flush=True)
    jnp.ones((1024, 1024, 1024), jnp.float32).block_until_ready(); print("B: 4 GiB ok", flush=True)
except Exception as e:
    print("B failed:", type(e).__name__, str(e)[:800], flush=True)
print(f"B: {time.time() - t:.1f} s", flush=True)
EOF
( CUDA_VISIBLE_DEVICES=0 "$PY" "$OUT/trap/hold.py" > "$OUT/trap/a.txt" 2>&1 ) & HOLD=$!
for i in $(seq 60); do grep -q "A ready" "$OUT/trap/a.txt" 2>/dev/null && break; sleep 1; done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > "$OUT/trap/apps-a.txt" 2>&1
CUDA_VISIBLE_DEVICES=0 timeout 120 "$PY" "$OUT/trap/second.py" > "$OUT/trap/b-default.txt" 2>&1
echo "exit $?" >> "$OUT/trap/b-default.txt"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 120 "$PY" "$OUT/trap/second.py" > "$OUT/trap/b-nopreallocate.txt" 2>&1
echo "exit $?" >> "$OUT/trap/b-nopreallocate.txt"
wait $HOLD
log "0 trap done"
sleep 30

log "1 solo start"; suite shared-solo --out "$OUT/solo"; log "1 solo done, exit $?"
sleep 30
log "2 shared start"
suite shared-a --out "$OUT/shared-a" & A=$!
suite shared-b --out "$OUT/shared-b" & Bj=$!
wait $A; echo "shared-a exit $? $(date -u +%FT%TZ)" >> "$OUT/run.log"
wait $Bj; echo "shared-b exit $? $(date -u +%FT%TZ)" >> "$OUT/run.log"
log "2 shared done"
sleep 30
log "3 solo2 start"; suite shared-solo2 --out "$OUT/solo2"; log "3 solo2 done, exit $?"
sleep 30

log "4 busy start"
( CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=$SHARE \
  "$PY" precision/sustained.py run --workload matmul --dtype float32 --size 8192 --minutes 15 \
  --out "$OUT/neighbour" > "$OUT/neighbour.log" 2>&1 ) & NB=$!
sleep 120   # the neighbour alone first, for its own baseline
for cell in vfi:grid:1500 egm:grid:50000 wealth:households:10000000 kalman:batch:10000 kalman:state:200; do
  IFS=: read -r p s n <<< "$cell"
  suite neighbour-busy --problems "$p" --sweeps "$s" --sizes "$n" --out "$OUT/busy"
  echo "busy $cell exit $? $(date -u +%FT%TZ)" >> "$OUT/run.log"
done
log "4 busy cells done; the neighbour runs on alone"
wait $NB
log "4 busy done, neighbour exit $?"
kill $SAMPLER 2>/dev/null

log "5 sustained start"
GPU=0 bash precision/sustained.sh qe-wharton1-gpu > "$OUT/sustained.log" 2>&1
log "5 sustained done, exit $?"
cp -p qe-wharton1-gpu-*-sustained.tgz "$OUT/" 2>/dev/null
log done
