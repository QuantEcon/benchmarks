#!/bin/bash
# QuantEcon/project-benchmarks-jax#63 on the pod: the slice's record, JAX's view, both --quick checks,
# then, if every quick cell is ok, the full problem types, highest, and the precision suite
set -uo pipefail
H=runpod-rtxpro6000-mig1g24; D=$(date -u +%F)
mark() { echo "$(date -u +%FT%TZ) $*" >> ~/status.txt; }
mark "M start"
mkdir -p ~/card
nvidia-smi -L > ~/card/nvidia-smi-L.txt 2>&1
nvidia-smi -q > ~/card/nvidia-smi-q.txt 2>&1; nvidia-smi -q -x > ~/card/nvidia-smi-q.xml 2>&1
nvidia-smi --query-gpu=name,pci.bus_id,pci.sub_device_id,vbios_version,driver_version,memory.total,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max,mig.mode.current --format=csv > ~/card/card.csv 2>&1
{ head -4 /etc/os-release; echo "kernel $(uname -r)"; echo "nproc $(nproc)"; lscpu | head -20; free -g; } > ~/card/host.txt 2>&1
env | cut -d= -f1 | grep -i -E 'cuda|nvidia|mig' > ~/card/env-names.txt
cd ~ && rm -rf benchmarks && git clone -q https://github.com/QuantEcon/benchmarks.git && cd benchmarks && git fetch -q origin env-info-mig && git checkout -q 8d777d6
git log --oneline -1 > ~/card/commit.txt
bash problems/run_by_hand.sh $H --quick; mark "M problems-quick exit $?"
source .venv-problems/bin/activate
for f in default 0.95; do
  if [ $f = default ]; then unset XLA_PYTHON_CLIENT_MEM_FRACTION; else export XLA_PYTHON_CLIENT_MEM_FRACTION=$f; fi
  XLA_PYTHON_CLIENT_PREALLOCATE=false python -c "import jax; d = jax.devices()[0]; print('$f:', jax.devices(), d.device_kind, 'allocator limit GiB', round(d.memory_stats()['bytes_limit'] / 2**30, 2))" >> ~/card/jax-view.txt 2>&1
done
unset XLA_PYTHON_CLIENT_MEM_FRACTION
bad=$(python -c "
import json, glob
rows = [json.loads(l) for f in glob.glob('problems/results/$H/*.jsonl') for l in open(f)]
print(sum(r.get('status') != 'ok' for r in rows) if rows else 'none')")
deactivate
bash precision/run_by_hand.sh $H gpu --quick; mark "M precision-quick exit $?"
if [ "$bad" != 0 ]; then mark "M stop: problems quick has $bad cells not ok"; exit 1; fi
mark "M quick done"
bash problems/run_by_hand.sh $H; mark "M problems-full exit $?"
source .venv-problems/bin/activate
JAX_DEFAULT_MATMUL_PRECISION=highest python problems/run_suite.py --backends jax-gpu --host $H --dtypes float32 --label highest 2>&1 \
  | tee problems/results/$H/$D/run-highest.log; mark "M highest exit ${PIPESTATUS[0]}"
deactivate
tar czf $H-$D-problems.tgz problems/results/$H
bash precision/run_by_hand.sh $H gpu; mark "M precision-full exit $?"
mark "M done"
