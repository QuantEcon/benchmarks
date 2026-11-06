#!/bin/bash
# setup_and_probe.sh, on the TPU VM as ubuntu: JAX 0.12.0.1 for TPU in ~/venv, then #38's probe in a fresh process.
# Each step logs to ~/session/<step>.log and writes ~/session/<step>.done ("<exit> <wall s>"); results go to ~/out.
mkdir -p ~/session ~/out
step() {
  name=$1; shift; start=$(date +%s)
  echo "== $name: start $(date -u '+%F %T UTC'); cmd: $*" | tee -a ~/session/$name.log
  "$@" 2>&1 | tee -a ~/session/$name.log; rc=${PIPESTATUS[0]}; end=$(date +%s)
  echo "== $name: end $(date -u '+%F %T UTC'); exit $rc; wall $((end - start)) s" | tee -a ~/session/$name.log
  echo "$rc $((end - start))" > ~/session/$name.done; return $rc
}
machine() {
  { date -u; hostname; cat /etc/os-release; uname -a; nproc; free -g; df -h /; lscpu; ls -l /dev/vfio /dev/accel* 2>&1; } | tee ~/out/machine.txt
}
setup() {
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH=$HOME/.local/bin:$PATH
  uv venv -q -p 3.12 ~/venv && source ~/venv/bin/activate || return 1
  uv pip install "jax[tpu]" -c ~/constraints.txt || return 1
  uv pip freeze > ~/out/environment.txt
  df -h / | tail -1
  python -c 'import jax; print(f"JAX {jax.__version__}: {jax.devices()}"); assert jax.devices()[0].platform == "tpu", "JAX sees no TPU"' | tee ~/out/devices.txt
  return ${PIPESTATUS[0]}
}
probe() {
  source ~/venv/bin/activate && cd ~/out
  timeout 35m python -u ~/tpu_f64_probe.py 2>&1 | tee ~/out/probe.txt; return ${PIPESTATUS[0]}
}
step machine machine
step setup setup && step probe probe
echo "$(date -u '+%F %T UTC')" > ~/session/all.done
