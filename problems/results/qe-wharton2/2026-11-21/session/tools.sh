#!/bin/bash
# tools.sh: a real git, from conda-forge through micromamba, in ~/bench/tools, as before 2026-11-18:
# /usr/bin/git is the Xcode stub, which would open a dialog on the console, and precision/env_info.py
# calls git for every cell.
set -e
T=~/bench/tools
if [ ! -x $T/env/bin/git ]; then
  mkdir -p $T
  curl -fsSL https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xj -C $T bin/micromamba
  $T/bin/micromamba create -y -q -r $T/mamba -p $T/env -c conda-forge git
fi
[ "$(command -v git)" = "$T/env/bin/git" ] || { echo "git on PATH isn't ours: $(command -v git)"; exit 1; }
git --version
