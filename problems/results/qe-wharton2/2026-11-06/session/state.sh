#!/bin/bash
# state.sh: the machine's state now: load, power, thermal, the busiest processes. ps output is redacted before
# publishing (the console user's name and the managed agents' names), as in QuantEcon/benchmarks#13.
date -u '+%F %T UTC'
uptime
pmset -g batt | head -2
pmset -g | grep -i -E 'lowpowermode|powermode' || true
pmset -g therm 2>/dev/null | grep -v '^$' || true
screen -ls || true
ps -axo pid,pcpu,pmem,rss,comm -r | head -15
