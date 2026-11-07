#!/bin/bash
# machine.sh: record the machine, and stop the chain if it is on battery or in Low Power Mode,
# since a timed run there wouldn't compare with the M6's on mains power.
sw_vers
sysctl -n machdep.cpu.brand_string hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.memsize 2>/dev/null
bash ~/bench/pbj-m5pro/state.sh
if pmset -g batt | head -1 | grep -q 'Battery Power'; then echo "on battery: stopping"; exit 2; fi
if pmset -g | grep -q -E 'lowpowermode +1'; then echo "Low Power Mode is on: stopping"; exit 3; fi
echo "on AC power, Low Power Mode off"
