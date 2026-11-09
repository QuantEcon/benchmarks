# `aws-c7i.8xlarge`, 2026-11-09: the CPU stacks on 16 cores of a Xeon Platinum 8488C

The problem-type suite's CPU stacks (`numpy`, `numba`, `numba-parallel` and `jax-cpu`) and the precision suite's CPU column, on 16 cores of Intel's Xeon Platinum 8488C (Sapphire Rapids), run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh) `--cpu`, for QuantEcon/project-benchmarks-jax#47. [`aws-c8a.4xlarge`](../../aws-c8a.4xlarge/2026-11-09/README.md) and [`aws-c8i.8xlarge`](../../aws-c8i.8xlarge/2026-11-09/README.md) ran the same runs that day, and the c8a's notes describe the runs and files that all three share.

| | |
|---|---|
| Instance | EC2 `c7i.8xlarge`, on demand, in us-east-1, launched with 16 cores and one thread per core (the type's default is 16 cores with two threads each), 64 GiB. It was launched from a laptop's AWS CLI, with a key pair and an SSH-only security group made for it and deleted afterwards, and was set to end itself after 300 minutes |
| Image | Canonical's Ubuntu 24.04 LTS, `ubuntu-noble-24.04-amd64-server-20261023` (`ami-0d6dd9e8265d48906`), kernel 7.0.0-1014-aws |
| CPU | Intel Xeon Platinum 8488C under KVM: one NUMA node, 2 MiB of L2 per core, and 105 MiB of L3 (`machine.txt`) |
| Software | JAX 0.12.0.1, NumPy 2.5.4, Numba 0.68.1 and SciPy 1.18.1, on Python 3.12.3 (`environment.txt`) |
| Harness | `1874f1c` |
| On the machine (UTC) | 03:10 to 06:28, launch to termination |

## The runs

As on the c8a, from `session/run.sh`, with the session's whole output in `session/session.log`:

- **The `--quick` check,** 03:10 to 03:13, `run_id` 20261109T031047Z: 36 cells, all `ok`, left out of the `.jsonl`.
- **The full run,** 03:13 to 06:00, `run_id` 20261109T031338Z: 199 cells `ok` and one `timeout`. P1's `numpy` cell at 1,500 passed the suite's 900 s limit for a cell, since its cold run and one warm run take about 16 minutes here.
- **The precision suite's CPU column,** 06:00 to 06:05, `run_id` 20261109T060022Z, in [`results/aws-c7i.8xlarge/2026-11-09.jsonl`](../../../../results/aws-c7i.8xlarge/2026-11-09.jsonl): 46 cells, all `ok`. With 64 GiB, `bellman_stored` at 60,000 in float64 fits.
- **P1's `jax-cpu` cells at 1,000 and 1,500 with XLA's YNNPACK reduce fusion off,** 06:05 to 06:09, `run_id` 20261109T060442Z, labelled `ynn-reduce-off`: 2 cells, both `ok`.
- **The timed-out cell again, with an 1,800 s limit,** 06:10 to 06:26, `run_id` 20261109T060946Z, labelled `timeout-1800`: `ok`, with a cold run of 481.5 s and one warm run of 481.1 s, 963 s in all. It took 572 iterations, as on the c8a and the c8i, and its answer is the same as theirs at every sampled position.
- **Each CPU's clock, every 2 s,** from 03:21 to 06:26, in `clock.jsonl`. It misses P1's cells up to `numpy` at 700.
- **One core's bandwidth,** 06:26, in `bandwidth.jsonl`, from [`problems/checks/bw_probe.py`](../../../checks/bw_probe.py) as at `51b698f`.
- **The triad,** 06:26, in `triad.jsonl`, from [`problems/checks/stream_triad.py`](../../../checks/stream_triad.py) as at `a17671a` (sha256 `e73525d3…`), at 1, 2, 4, 8 and 16 threads. The arrays were first touched on all 16 threads, in the instance's one NUMA node.

`summary.md` is this host's tables, from `summarise.py` at `2f66a4c` once the `--quick` rows were left out. The `timeout-1800` and `ynn-reduce-off` cells have tables of their own. The run's own summary step failed on `clock.jsonl`, as on the c8a.
