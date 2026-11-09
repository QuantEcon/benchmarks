# `aws-c8i.8xlarge`, 2026-11-09: the CPU stacks on 16 cores of a Xeon 6975P-C

The problem-type suite's CPU stacks (`numpy`, `numba`, `numba-parallel` and `jax-cpu`) and the precision suite's CPU column, on 16 cores of Intel's Xeon 6975P-C (Granite Rapids), run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh) `--cpu`, for QuantEcon/project-benchmarks-jax#47. [`aws-c8a.4xlarge`](../../aws-c8a.4xlarge/2026-11-09/README.md) and [`aws-c7i.8xlarge`](../../aws-c7i.8xlarge/2026-11-09/README.md) ran the same runs that day, and the c8a's notes describe the runs and files that all three share.

| | |
|---|---|
| Instance | EC2 `c8i.8xlarge`, on demand, in us-east-1, launched with 16 cores and one thread per core (the type's default is 16 cores with two threads each), 64 GiB. It was launched from a laptop's AWS CLI, with a key pair and an SSH-only security group made for it and deleted afterwards, and was set to end itself after 300 minutes |
| Image | Canonical's Ubuntu 24.04 LTS, `ubuntu-noble-24.04-amd64-server-20261023` (`ami-0d6dd9e8265d48906`), kernel 7.0.0-1014-aws |
| CPU | Intel Xeon 6975P-C under KVM: one NUMA node, 2 MiB of L2 per core, and 480 MiB of L3 (`machine.txt`) |
| Software | JAX 0.12.0.1, NumPy 2.5.4, Numba 0.68.1 and SciPy 1.18.1, on Python 3.12.3 (`environment.txt`) |
| Harness | `1874f1c` |
| On the machine (UTC) | 03:11 to 05:51, launch to termination |

## The runs

As on the c8a, from `session/run.sh`, with the session's whole output in `session/session.log`:

- **The `--quick` check,** 03:12 to 03:14, `run_id` 20261109T031154Z: 36 cells, all `ok`, left out of the `.jsonl`.
- **The full run,** 03:14 to 05:40, `run_id` 20261109T031409Z: 200 cells, all `ok`.
- **The precision suite's CPU column,** 05:40 to 05:43, `run_id` 20261109T053949Z, in [`results/aws-c8i.8xlarge/2026-11-09.jsonl`](../../../../results/aws-c8i.8xlarge/2026-11-09.jsonl): 46 cells, all `ok`. With 64 GiB, `bellman_stored` at 60,000 in float64 fits.
- **P1's `jax-cpu` cells at 1,000 and 1,500 with XLA's YNNPACK reduce fusion off,** 05:43 to 05:49, `run_id` 20261109T054318Z, labelled `ynn-reduce-off`: 2 cells, both `ok`.
- **Each CPU's clock, every 2 s,** from 03:21 to 05:49, in `clock.jsonl`. It misses P1's cells up to `numba` at 700.
- **One core's bandwidth,** 05:49, in `bandwidth.jsonl`, from [`problems/checks/bw_probe.py`](../../../checks/bw_probe.py) as at `51b698f`.
- **The triad,** 05:49, in `triad.jsonl`, from [`problems/checks/stream_triad.py`](../../../checks/stream_triad.py) as at `a17671a` (sha256 `e73525d3…`), at 1, 2, 4, 8 and 16 threads. The arrays were first touched on all 16 threads, in the instance's one NUMA node.

`summary.md` is this host's tables, from `summarise.py` at `2f66a4c` once the `--quick` rows were left out. The run's own summary step failed on `clock.jsonl`, as on the c8a.
