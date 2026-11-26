#!/bin/bash
# step.sh <name> <command...>: run one step from ~/benchmarks, log it to ~/session/<name>.log,
# and write ~/session/<name>.done ("<exit> <wall s>") when it ends, for the laptop to wait on.
name=$1; shift
cd ~/benchmarks
start=$(date +%s)
echo "== $name: start $(date -u '+%F %T UTC'); cmd: $*" | tee -a ~/session/$name.log
"$@" 2>&1 | tee -a ~/session/$name.log
rc=${PIPESTATUS[0]}
end=$(date +%s)
echo "== $name: end $(date -u '+%F %T UTC'); exit $rc; wall $((end - start)) s" | tee -a ~/session/$name.log
echo "$rc $((end - start))" > ~/session/$name.done
