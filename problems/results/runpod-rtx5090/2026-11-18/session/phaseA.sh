#!/bin/bash
# QuantEcon/project-benchmarks-jax#60 on the pod, phase A: the card's record, the clone, the cap test and both --quick checks
set -uo pipefail
H=runpod-rtx5090
mark() { echo "$(date -u +%FT%TZ) $*" >> ~/status.txt; }
mark "A start"
mkdir -p ~/card
nvidia-smi -q > ~/card/nvidia-smi-q.txt 2>&1
nvidia-smi --query-gpu=name,pci.bus_id,pci.sub_device_id,vbios_version,driver_version,memory.total,power.limit,power.default_limit,power.min_limit,power.max_limit,clocks.max.sm,clocks.max.mem,pcie.link.gen.max,pcie.link.width.max --format=csv > ~/card/card.csv 2>&1
{ head -4 /etc/os-release; echo "kernel $(uname -r)"; echo "nproc $(nproc)"; lscpu | head -20; free -g; df -h /; } > ~/card/host.txt 2>&1
cur=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
{ echo "nvidia-smi -pl ${cur%.*} (the current limit, which changes nothing):"; nvidia-smi -pl "${cur%.*}"; echo "exit $?"; } > ~/card/pl-test.txt 2>&1
cd ~ && rm -rf benchmarks && git clone -q https://github.com/QuantEcon/benchmarks.git && cd benchmarks && git checkout -q f085c6d
git log --oneline -1 > ~/card/commit.txt
bash problems/run_by_hand.sh $H --quick; mark "A problems-quick exit $?"
bash precision/run_by_hand.sh $H gpu --quick; mark "A precision-quick exit $?"
mark "A done"
