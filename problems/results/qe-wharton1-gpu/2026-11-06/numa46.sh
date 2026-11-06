#!/bin/bash
# QuantEcon/project-benchmarks-jax#46: why qe-wharton1-gpu's Threadripper 2990WX runs the CPU
# stacks slowly. Four problem-type cells, in float64, each under three numactl placements, from
# QuantEcon/benchmarks' main, with nothing in the suite changed. Run it in tmux under econ-ai's
# machine lock, which the caller takes and this releases on exit:
#   1. set-up: the checkout moved to main, the environment, and the machine's layout;
#   2. a check: each cell's --quick size under each placement, into this run's own folder,
#      stopping unless every row is ok;
#   3. the run: each cell under each placement, cell by cell so that any drift falls on all
#      three alike, into problems/results/qe-wharton1-gpu/;
#   4. the single-threaded stacks on a node without memory;
#   5. last: an optional EGM run for project-econ-on-ai-hardware, cut from this copy.
# Nodes 0 and 2 have the memory, 64 GB each; nodes 1 and 3 have none. OpenBLAS, Numba and XLA
# all size their thread pools to the bound CPUs (checked on 2026-11-06), so no thread count is set.
set -u
LOCK=~/econ-ai/.lock-machine
[ -d "$LOCK" ] || { echo "the lock is not held; not starting"; exit 1; }
B=~/bench/benchmarks
R=~/bench/numa46/run-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p $R

# label|numactl arguments
PLACEMENTS=("numa-default|" "numa-one-node|--cpunodebind=0 --membind=0" "numa-memory-nodes|--cpunodebind=0,2 --membind=0,2")
# numactl won't take a node without memory for --cpunodebind ("<1> is invalid"), so node 1's CPUs
# are named instead
REMOTE="numa-remote-memory|--physcpubind=16-23,48-55 --membind=0"
# problem|sweep|size|backends: the three largest gaps to the M6 and the one win, cheapest first.
# Left out, as slowest on the Threadripper and never its fastest: vfi's numpy, wealth's
# numba and egm's jax-cpu. vfi runs in float64 only.
CELLS=("vfi|grid|1500|numba,numba-parallel,jax-cpu"
       "kalman|batch|10000|numpy,numba,numba-parallel,jax-cpu"
       "wealth|households|10000000|numpy,numba-parallel,jax-cpu"
       "egm|grid|50000|numpy,numba,numba-parallel")
REMOTE_CELLS=("vfi|grid|1500|numba" "kalman|batch|10000|numpy" "wealth|households|10000000|numpy" "egm|grid|50000|numpy")

state() {
  echo "== $1: $(date "+%F %T %Z") ($(date -u +%T) UTC)"; uptime
  nvidia-smi --query-gpu=index,pci.bus_id,clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader
  ps -eo pid,user,stat,pcpu,comm --sort=-pcpu | head -8
  numactl --hardware | grep free
}
sample() { while sleep 600; do state sample >> $R/state.txt; done; }
state before > $R/state.txt
sample & SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null; state after >> $R/state.txt; rmdir "$LOCK"; echo "== lock released $(date -u +%T)" >> $R/state.txt' EXIT
cd $B

step() {  # step <name> <command...>: log its start, end and exit status
  local n=$1; shift
  echo "== $n start $(date -u +%T)" >> $R/state.txt
  "$@" > $R/$n.log 2>&1; local s=$?
  echo "== $n done $(date -u +%T), exit $s" >> $R/state.txt
  return $s
}

setup() {
  # The run checkout holds an untracked copy of 2026-11-02.jsonl, byte-identical to main's
  # tracked file; it would block the checkout, so it moves aside rather than being deleted
  local f=problems/results/qe-wharton1-gpu/2026-11-02.jsonl
  git fetch -q origin || return 1
  if [ -f $f ] && ! git ls-files --error-unmatch $f >/dev/null 2>&1; then
    cmp -s $f <(git show origin/main:$f) || { echo "$f differs from main's; stopping"; return 1; }
    mv $f $R/2026-11-02.jsonl.untracked-copy
  fi
  git checkout -q --detach origin/main || return 1
  git log -1 --format='benchmarks at %h: %s'
  git status --porcelain
  ~/.local/bin/uv pip freeze --python .venv/bin/python > $R/environment.txt
  .venv/bin/python -c "import jax, numba, numpy, scipy; print('jax', jax.__version__, 'numba', numba.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
}

layout() {
  numactl --hardware; echo; lscpu; echo; free -g; echo
  echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing)"
  echo "zone_reclaim_mode: $(cat /proc/sys/vm/zone_reclaim_mode)"
  echo "THP: enabled $(cat /sys/kernel/mm/transparent_hugepage/enabled), defrag $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
  echo "cpufreq: governor $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null), boost $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null)"
  echo "BIOS memory interleave: not readable without root; four nodes, two without memory, means it isn't interleaved across dies"
  uname -a
  for p in "${PLACEMENTS[@]}" "$REMOTE"; do
    local a=${p#*|}; echo; echo "== ${p%%|*}: numactl ${a:-(none)}"
    [ -n "$a" ] && numactl $a numactl --show
  done
}

suite() {  # suite <label> <numactl arguments> <run_suite.py arguments...>
  local label=$1 args=$2; shift 2
  ${args:+numactl $args} .venv/bin/python problems/run_suite.py --dtypes float64 \
    --label $label --host qe-wharton1-gpu "$@"
}

check() {
  local expected=0 c p sw n be pl
  for c in "${CELLS[@]}"; do IFS='|' read -r p sw n be <<< "$c"
    for pl in "${PLACEMENTS[@]}"; do
      step "check-${pl%%|*}-$p" suite "${pl%%|*}" "${pl#*|}" --problems $p --sweeps $sw --quick \
        --backends $be --out $R/check || return 1
      expected=$((expected + $(echo $be | tr ',' ' ' | wc -w)))
    done
  done
  .venv/bin/python - "$R/check" $expected <<'PY'
import json, pathlib, sys
rows = [json.loads(l) for f in pathlib.Path(sys.argv[1]).rglob('*.jsonl') for l in open(f)]
bad = sorted({(r['label'], r['problem'], r['backend'], r['status']) for r in rows if r['status'] != 'ok'})
print(f"{len(rows)} rows, {sys.argv[2]} expected; not ok: {bad}")
sys.exit(0 if not bad and len(rows) == int(sys.argv[2]) else 1)
PY
}

run() {
  local c p sw n be pl
  for c in "${CELLS[@]}"; do IFS='|' read -r p sw n be <<< "$c"
    for pl in "${PLACEMENTS[@]}"; do
      step "${pl%%|*}-$p" suite "${pl%%|*}" "${pl#*|}" --problems $p --sweeps $sw --sizes $n \
        --backends $be --timeout 1800
    done
  done
}

remote() {
  local c p sw n be
  for c in "${REMOTE_CELLS[@]}"; do IFS='|' read -r p sw n be <<< "$c"
    step "${REMOTE%%|*}-$p" suite "${REMOTE%%|*}" "${REMOTE#*|}" --problems $p --sweeps $sw \
      --sizes $n --backends $be --timeout 1800
  done
}

# 5. Cut from this copy: an optional EGM run for project-econ-on-ai-hardware, at their
# request, which reports it there. It ran from their checkout after step 4, and wrote
# nothing to problems/results/.

step setup setup || { echo "== set-up failed; stopping" >> $R/state.txt; exit 1; }
step layout layout
step check check || { echo "== the check failed; stopping" >> $R/state.txt; exit 1; }
run
remote
echo ALLDONE >> $R/state.txt
