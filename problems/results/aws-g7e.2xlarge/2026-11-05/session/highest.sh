#!/bin/bash
cd ~/benchmarks && source .venv-problems/bin/activate
echo "highest start $(date -u +%FT%TZ)"
JAX_DEFAULT_MATMUL_PRECISION=highest python problems/run_suite.py --backends jax-gpu --host aws-g7e.2xlarge --dtypes float32 --label highest 2>&1 | tee problems/results/aws-g7e.2xlarge/2026-11-05/run-highest.log
echo "highest end $(date -u +%FT%TZ), exit ${PIPESTATUS[0]}"
