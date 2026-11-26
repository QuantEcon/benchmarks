#!/bin/bash
# tpu_cells.sh, on the TPU VM: the precision set's cells through the harness's own precision/run_cell.py at 5cd9d3d,
# unchanged, one fresh process per cell as run_matrix.py runs them (run_matrix.py accepts only cpu and gpu, so this
# loop stands in for it). It adds the driver's fields: host, label, run_id, timestamp, and the cell's environment.
# Quick sizes first, so a reclaim still leaves the core cells; then the rest; then float32 matmul at highest.
# Waits for the probe to finish first: one process at a time can hold the TPU.
until [ -f ~/session/all.done ]; do sleep 10; done
source ~/venv/bin/activate
cd ~/benchmarks
OUT=~/out/tpu-cells.jsonl
RUN=$(date -u +%Y%m%dT%H%M%SZ)
cell() {
  w=$1; n=$2; d=$3; label=$4; shift 4
  x64=0; [ $d = float64 ] && x64=1
  line=$(env JAX_PLATFORMS=tpu JAX_ENABLE_X64=$x64 "$@" timeout 240 python precision/run_cell.py \
    --workload $w --size $n --dtype $d --reps 5 2> ~/out/cell-stderr.txt | tail -1)
  python - "$line" "$w" "$n" "$d" "$label" "$RUN" "$x64 $*" >> $OUT <<'PY'
import json, sys, datetime
line, w, n, d, label, run, extra = sys.argv[1:]
try:
    row = json.loads(line)
except Exception:
    row = {"status": "error", "error": "no row: timeout or crash",
           "stderr_tail": open("/home/ubuntu/out/cell-stderr.txt").read()[-1500:]}
row.update(host="gcp-ct6e-standard-1t", workload=w, size=int(n), dtype=d, device="tpu", label=label,
           run_id=run, cell_env="JAX_PLATFORMS=tpu JAX_ENABLE_X64=" + extra,
           timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat())
print(json.dumps(row))
PY
  tail -1 $OUT | python -c 'import json, sys; r = json.loads(sys.stdin.read()); print(r["workload"], r["size"], r["dtype"], r["label"] or "-", r["status"], "warm", r.get("warm_median_s"), "compile", r.get("compile_s"), (r.get("error") or "")[:160])'
}
echo "== cells: start $(date -u '+%F %T UTC'), run $RUN"
for d in float32 float64; do
  for n in 256 1024; do cell matmul $n $d ""; done
  for w in bellman_stored bellman_recompute; do for n in 1000 5000; do cell $w $n $d ""; done; done
  cell kalman 1000 $d ""
done
echo "== quick sizes done $(date -u '+%T UTC')"
for d in float32 float64; do
  for n in 512 2048 4096 8192; do cell matmul $n $d ""; done
  for w in bellman_stored bellman_recompute; do for n in 2000 10000 20000 40000 60000; do cell $w $n $d ""; done; done
  for n in 10000 100000; do cell kalman $n $d ""; done
done
echo "== full sizes done $(date -u '+%T UTC')"
for n in 256 512 1024 2048 4096 8192; do cell matmul $n float32 highest JAX_DEFAULT_MATMUL_PRECISION=highest; done
echo "== cells: end $(date -u '+%F %T UTC')"
echo "$(date -u '+%F %T UTC')" > ~/session/cells.done
