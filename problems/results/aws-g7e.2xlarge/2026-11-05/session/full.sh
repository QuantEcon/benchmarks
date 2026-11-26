#!/bin/bash
cd ~/benchmarks && git log --oneline -1
start=$(date -u +%s); echo "full start $(date -u +%FT%TZ)"
UV_VENV_CLEAR=1 bash problems/run_by_hand.sh aws-g7e.2xlarge
rc=$?
echo "full end $(date -u +%FT%TZ), wall $(( $(date -u +%s) - start )) s, exit $rc"
