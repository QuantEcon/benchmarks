#!/bin/bash
# Sustained load on one NVIDIA card: does it hold its clocks through an hour of steady work?
#
#   git clone https://github.com/QuantEcon/benchmarks.git && cd benchmarks
#   bash precision/sustained.sh <host> [--minutes 60] [--idle 10] [--cap <watts>]
#
# <host> names the machine, as in run_by_hand.sh. The schedule, with nvidia-smi logging the card
# once a second throughout, in its own tmux session:
#   1. --idle minutes with the card idle;
#   2. --minutes of bellman_stored at m = 40,000 in float64, bound by memory bandwidth;
#   3. --idle minutes again;
#   4. --minutes of matmul at n = 8,192 in float32, at JAX's default precision, which on Ampere
#      and later cards is TF32 on the tensor cores.
# The idle periods let each workload start from the card's idle temperature, and the log shows
# whether it did. WORKLOADS replaces the pair, as workload:dtype:size separated by spaces.
# Each workload runs in a fresh process through precision/sustained.py, which times every step.
# precision/sustained.py summarise then compares the first 5 minutes with the last 30: step time,
# clocks, power, temperature and the clock-event reasons. That comparison is scaled down in
# proportion for runs shorter than an hour.
#
# --cap first sets the card's power limit with nvidia-smi -pl. That needs root on the host, which
# a rented container usually lacks, so the script stops if it's refused. On exit the script sets
# the limit back to what it was, since the card may be someone else's.
#
# It uses the precision suite's environment, .venv-phase2, and installs it as run_by_hand.sh
# does if it isn't there. Results go to precision/runs/<date>/<host>/sustained/
# (sustained-<cap>W with --cap): the card's record, nvidia-smi.csv, phases.csv, each workload's
# per-second rows and its record, and summary.md. They're packed into
# <host>-<date>-sustained.tgz. They stay out of results/, whose .jsonl files are the precision
# matrix's cells.
#
# JAX_PLATFORMS=cpu skips the check that JAX sees the card, to test the script on a machine
# without one; such a run measures nothing.
set -euo pipefail
USAGE="usage: bash precision/sustained.sh <host> [--minutes M] [--idle M] [--cap W]"
HOST=${1:?$USAGE}; shift
MINUTES=60; IDLE=10; CAP=""
while [ $# -gt 0 ]; do
  case $1 in
    --minutes) MINUTES=${2:?$USAGE}; shift 2 ;;
    --idle) IDLE=${2:?$USAGE}; shift 2 ;;
    --cap) CAP=${2:?$USAGE}; shift 2 ;;
    *) echo "$USAGE"; exit 1 ;;
  esac
done
WORKLOADS=${WORKLOADS:-"bellman_stored:float64:40000 matmul:float32:8192"}
GPU=${GPU:-0}
DATE=$(date -u +%F)
OUT="precision/runs/$DATE/$HOST/sustained${CAP:+-${CAP}W}"
TARBALL="$HOST-$DATE-sustained${CAP:+-${CAP}W}.tgz"
SESSION="sustained-smi-$$"
command -v nvidia-smi > /dev/null || { echo "no nvidia-smi: this script is for an NVIDIA card"; exit 1; }
command -v tmux > /dev/null || { echo "tmux is needed, to run the card's log apart from this script"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: move it, or use another host name"; exit 1; }
mkdir -p "$OUT"
seconds() { awk -v m="$1" 'BEGIN { printf "%d", m * 60 + 0.5 }'; }
stamp() { date -u +%FT%TZ; }

echo "== The card"
nvidia-smi -q -i "$GPU" > "$OUT/card-q.txt" 2>&1 || true
CARD=name,pci.bus_id,pci.sub_device_id,vbios_version,driver_version,memory.total,power.limit
CARD=$CARD,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm,clocks.max.mem
CARD=$CARD,pcie.link.gen.max,pcie.link.width.max
nvidia-smi -i "$GPU" --query-gpu=$CARD --format=csv 2>&1 | tee "$OUT/card.csv" || true
{ lscpu | head -25; free -g; } > "$OUT/host.txt" 2>&1 || true

echo "== Python 3.12 and JAX"
export PATH="$HOME/.local/bin:$PATH"
if [ ! -f .venv-phase2/bin/activate ]; then
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  rm -rf .venv-phase2 && uv venv -q -p 3.12 .venv-phase2
  source .venv-phase2/bin/activate
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if [ "${driver%%.*}" -ge 580 ]; then CUDA=cuda13; else CUDA=cuda12; fi
  uv pip install -q "jax[$CUDA]" -c precision/constraints.txt
fi
source .venv-phase2/bin/activate
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
if [ "${JAX_PLATFORMS:-cuda}" = cuda ]; then
  export JAX_PLATFORMS=cuda
  python -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}'); assert jax.devices()[0].platform == 'gpu', 'JAX sees no GPU'"
else
  echo "JAX_PLATFORMS=$JAX_PLATFORMS: a test of the script, not a measurement of the card"
fi

RESTORE=""
cleanup() {
  tmux kill-session -t "$SESSION" 2> /dev/null || true
  if [ -n "$RESTORE" ]; then
    if nvidia-smi -i "$GPU" -pl "$RESTORE"; then
      echo "== The power limit is back at $RESTORE W"
    else
      echo "== SET THE POWER LIMIT BACK BY HAND: nvidia-smi -i $GPU -pl $RESTORE"
    fi
  fi
}
trap cleanup EXIT
if [ -n "$CAP" ]; then
  echo "== A power limit of $CAP W"
  RESTORE=$(nvidia-smi -i "$GPU" --query-gpu=power.limit --format=csv,noheader,nounits | awk 'NR == 1 { printf "%d", $1 + 0.5 }')
  if ! nvidia-smi -i "$GPU" -pl "$CAP"; then
    RESTORE=""
    echo "nvidia-smi -pl was refused: this host doesn't allow a power limit to be set"
    exit 1
  fi
  nvidia-smi -i "$GPU" --query-gpu=power.limit,power.default_limit --format=csv | tee "$OUT/cap.csv"
fi

echo "== The card's log, once a second"
QUERY=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active
# Drivers before clocks_event_reasons was named call it clocks_throttle_reasons
nvidia-smi --id="$GPU" --query-gpu=$QUERY --format=csv > /dev/null 2>&1 \
  || QUERY=${QUERY/clocks_event_reasons/clocks_throttle_reasons}
echo "nvidia-smi --id=$GPU --query-gpu=$QUERY --format=csv -l 1" > "$OUT/nvidia-smi.cmd"
tmux new-session -d -s "$SESSION" \
  "TZ=UTC nvidia-smi --id=$GPU --query-gpu=$QUERY --format=csv -l 1 > '$PWD/$OUT/nvidia-smi.csv' 2>&1"

echo "phase,start,end" > "$OUT/phases.csv"
idle() {
  local start; start=$(stamp)
  echo "== Idle, $IDLE minutes, from $start"
  sleep "$(seconds "$IDLE")"
  echo "idle,$start,$(stamp)" >> "$OUT/phases.csv"
}
for workload in $WORKLOADS; do
  IFS=: read -r name dtype size <<< "$workload"
  idle
  start=$(stamp)
  echo "== $name, $dtype, $size: $MINUTES minutes, from $start"
  python precision/sustained.py run --workload "$name" --dtype "$dtype" --size "$size" \
    --minutes "$MINUTES" --out "$OUT" || echo "$name $dtype $size: exit $?"
  echo "$name-$dtype-$size,$start,$(stamp)" >> "$OUT/phases.csv"
done
tmux kill-session -t "$SESSION" 2> /dev/null || true

echo "== First minutes against last"
python precision/sustained.py summarise "$OUT" | tee "$OUT/summary.md"
tar czf "$TARBALL" "$OUT"
echo "== Done: $TARBALL. Copy it off the machine."
