#!/bin/bash
# step.sh <name> <command...>: run one step from ~/bench/benchmarks, log it to ~/bench/pbj-m5pro/<name>.log,
# and write ~/bench/pbj-m5pro/<name>.done ("<exit> <wall s>") when it ends.
# The real git comes first on PATH: /usr/bin/git is the Xcode stub, which would open a dialog on the console.
export PATH="$HOME/bench/tools/env/bin:$HOME/.local/bin:$PATH"
name=$1; shift
d=~/bench/pbj-m5pro
cd ~/bench/benchmarks
start=$(date +%s)
echo "== $name: start $(date -u '+%F %T UTC'); cmd: $*" | tee -a $d/$name.log
"$@" 2>&1 | tee -a $d/$name.log
rc=${PIPESTATUS[0]}
end=$(date +%s)
echo "== $name: end $(date -u '+%F %T UTC'); exit $rc; wall $((end - start)) s" | tee -a $d/$name.log
echo "$rc $((end - start))" > $d/$name.done
