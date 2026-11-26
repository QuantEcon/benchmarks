# What ran on aws-c8a.4xlarge on 2026-11-09, for QuantEcon/project-benchmarks-jax#47, in order.
# Each step ran in tmux from a clone of QuantEcon/benchmarks at 1874f1c, with its output in
# ~/run.log, which is session.log here. The scripts in ~ are the files now in problems/checks/.

# 03:11 UTC: the runs, as one chain
bash problems/run_by_hand.sh aws-c8a.4xlarge --quick --cpu \
  && bash problems/run_by_hand.sh aws-c8a.4xlarge --cpu \
  && .venv-problems/bin/python precision/run_matrix.py --host aws-c8a.4xlarge --devices cpu \
  && XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=-reduce .venv-problems/bin/python \
       problems/run_suite.py --problems vfi --sizes 1000,1500 --backends jax-cpu --dtypes float64 \
       --label ynn-reduce-off --host aws-c8a.4xlarge

# 03:21: beside it, in a tmux session of its own, until the chain's log ended
python3 ~/clock_sampler.py ~/benchmarks/problems/results/aws-c8a.4xlarge/2026-11-09/clock.jsonl ~/run.log 2

# 05:10: once the chain had ended, with the sampler started again beside it, writing to the same
# clock.jsonl. This step's output was logged apart, then added to the end of ~/run.log
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 0 \
  .venv-problems/bin/python ~/bw_probe.py problems/results/aws-c8a.4xlarge/2026-11-09/bandwidth.jsonl
