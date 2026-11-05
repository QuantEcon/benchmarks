#!/bin/bash
# The problem-type suite on a rented GPU machine, NVIDIA's or AMD's, run by hand.
#
#   git clone https://github.com/QuantEcon/benchmarks.git && cd benchmarks
#   bash problems/run_by_hand.sh <host> [--quick]
#
# <host> names the machine in the results as <provider>-<instance>, e.g. aws-g7e.2xlarge.
# --quick runs each sweep's small sizes, to check the machine; run it first.
#
# It records the machine, installs a Python 3.12 environment with uv in .venv-problems, and
# stops if JAX sees no GPU. On NVIDIA it installs jax[cuda13] on a driver of 580 or later, and
# otherwise jax[cuda12], labelled cuda12. On AMD it installs jax 0.12.0 with the plugin for the
# ROCm in /opt/rocm, as precision/run_by_hand.sh does, whose header says how to set ROCM_INDEX
# and ROCM_MAJOR on a machine whose ROCm is neither 7 nor 10. Then it runs every problem on the
# card alone (--backends jax-gpu), in both precisions. The card needn't run the CPU backends:
# summarise.py checks each answer against NumPy's float64 answer from the CPU runs already in
# problems/results/. Results go to problems/results/<host>/, packed into <host>-<date>-problems.tgz.
set -euo pipefail
HOST=$1; QUICK=${2:-}
case $QUICK in ""|--quick) ;; *) echo "usage: bash problems/run_by_hand.sh <host> [--quick]"; exit 1 ;; esac
DATE=$(date -u +%F)
OUT="problems/results/$HOST/$DATE"
mkdir -p "$OUT"
if [ -n "$(nvidia-smi --query-gpu=index --format=csv,noheader 2> /dev/null)" ]; then PLATFORM=cuda; else PLATFORM=rocm; fi

echo "== The machine"
if [ $PLATFORM = cuda ]; then
  {
    nvidia-smi
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,pci.bus_id,power.limit --format=csv
    lscpu | head -25
    free -g
  } > "$OUT/machine.txt" 2>&1
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  if [ "${driver%%.*}" -ge 580 ]; then CUDA=cuda13; LABEL=""; else CUDA=cuda12; LABEL=cuda12; fi
  echo "driver $driver: jax[$CUDA]"
else
  rocm=$(cat /opt/rocm/.info/version 2>/dev/null || echo none)
  ROCM_MAJOR=${ROCM_MAJOR:-${rocm%%.*}}; LABEL=""
  {
    head -4 /etc/os-release
    echo "kernel $(uname -r), amdgpu module $(cat /sys/module/amdgpu/version 2>/dev/null || echo 'built in'), ROCm in /opt/rocm: $rocm"
    amd-smi version
    amd-smi static
    lscpu | head -25
    free -g
  } > "$OUT/machine.txt" 2>&1 || true
  case $ROCM_MAJOR in
    7|10) echo "ROCm $rocm: jax-rocm$ROCM_MAJOR-plugin" ;;
    *) echo "ROCm $rocm has no jax 0.12.0 plugin: set ROCM_MAJOR to 7 or 10, and ROCM_INDEX for ROCm's own packages"
       exit 1 ;;
  esac
fi

echo "== Python 3.12, JAX, Numba and SciPy"
export PATH="$HOME/.local/bin:$PATH"
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -q -p 3.12 .venv-problems
source .venv-problems/bin/activate
if [ $PLATFORM = cuda ]; then
  uv pip install -q "jax[$CUDA]" numba scipy -c problems/constraints.txt
else
  if [ -n "${ROCM_INDEX:-}" ]; then
    # ROCm's packages carry device code for the cards named, by gfx target; GFX overrides amd-smi's
    GFX=${GFX:-$(python -c 'import sys; sys.path.insert(0, "precision"); import env_info; print(next((c["gfx_target"] for c in env_info.amd_gpus() if c["gfx_target"]), ""))')}
    [ -n "$GFX" ] || { echo "amd-smi gave no gfx target: set GFX, e.g. GFX=gfx942"; exit 1; }
    uv pip install -q --index-url "$ROCM_INDEX" "rocm[libraries,device-$GFX]"
  fi
  uv pip install -q jax "jax-rocm$ROCM_MAJOR-plugin" "jax-rocm$ROCM_MAJOR-pjrt" numba scipy \
    -c problems/constraints-rocm.txt
fi
uv pip freeze > "$OUT/environment.txt"
JAX_PLATFORMS=$PLATFORM python -c "import jax; print(f'JAX {jax.__version__}: {jax.devices()}'); assert jax.devices()[0].platform == 'gpu', 'JAX sees no GPU'"

echo "== The suite on the card"
python problems/run_suite.py --backends jax-gpu --host "$HOST" ${QUICK:+--quick} ${LABEL:+--label "$LABEL"} \
  | tee -a "$OUT/run.log"
# This host's tables, with any answer that differs from NumPy's in float64 listed below each
python problems/summarise.py problems/results/ | awk -v h="on $HOST" '/^### /{keep = index($0, h) > 0} keep' \
  > "$OUT/summary.md" || true
tar czf "$HOST-$DATE-problems.tgz" "problems/results/$HOST"
echo "== Done: $HOST-$DATE-problems.tgz. Copy it off the machine, then terminate the machine."
