#!/bin/bash
cd ~
git clone -q https://github.com/QuantEcon/benchmarks.git && cd benchmarks && git checkout -q 6b42b04 && git log --oneline -1
start=$(date -u +%s); echo "quick start $(date -u +%FT%TZ)"
bash problems/run_by_hand.sh aws-g7e.2xlarge --quick
rc=$?
echo "quick end $(date -u +%FT%TZ), wall $(( $(date -u +%s) - start )) s, exit $rc"
