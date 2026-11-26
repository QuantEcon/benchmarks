# `aws-c8a.4xlarge`, 2026-11-09: the CPU stacks on 16 cores of an EPYC 9R45

The problem-type suite's CPU stacks (`numpy`, `numba`, `numba-parallel` and `jax-cpu`) and the precision suite's CPU column, on 16 cores of AMD's EPYC 9R45 (Zen 5), run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh) `--cpu`, for QuantEcon/project-benchmarks-jax#47. Two Intel instances ran the same runs that day, [`aws-c7i.8xlarge`](../../aws-c7i.8xlarge/2026-11-09/README.md) and [`aws-c8i.8xlarge`](../../aws-c8i.8xlarge/2026-11-09/README.md).

| | |
|---|---|
| Instance | EC2 `c8a.4xlarge`, on demand, in us-east-1: 16 cores with one thread each (the type has no SMT), 32 GiB. It was launched from a laptop's AWS CLI, with a key pair and an SSH-only security group made for it and deleted afterwards, and was set to end itself after 300 minutes |
| Image | Canonical's Ubuntu 24.04 LTS, `ubuntu-noble-24.04-amd64-server-20261023` (`ami-0d6dd9e8265d48906`), kernel 7.0.0-1014-aws |
| CPU | AMD EPYC 9R45 under KVM: one NUMA node, 1 MiB of L2 per core, and 64 MiB of L3 in two 32 MiB slices (`machine.txt`) |
| Software | JAX 0.12.0.1, NumPy 2.5.4, Numba 0.68.1 and SciPy 1.18.1, on Python 3.12.3 (`environment.txt`) |
| Harness | `1874f1c` |
| On the machine (UTC) | 03:10 to 05:11, launch to termination |

## The runs

Each ran in one `tmux` session, in turn, from `session/run.sh`. The session's whole output is `session/session.log`.

- **The `--quick` check,** 03:11 to 03:13, `run_id` 20261109T031121Z: 36 cells, all `ok`. Its rows are left out of the `.jsonl`, since the full run repeats each of their cells; its output is in `run.log`.
- **The full run,** 03:13 to 05:04, `run_id` 20261109T031311Z: every size in both precisions, 200 cells, all `ok`.
- **The precision suite's CPU column,** 05:04 to 05:06, `run_id` 20261109T050348Z, in [`results/aws-c8a.4xlarge/2026-11-09.jsonl`](../../../../results/aws-c8a.4xlarge/2026-11-09.jsonl): 45 cells `ok`. `bellman_stored` at 60,000 in float64 was `skipped`, since it needs about 26.8 GiB and the limit here was 24.6 GiB.
- **P1's `jax-cpu` cells at 1,000 and 1,500 again with XLA's YNNPACK reduce fusion off,** 05:06 to 05:10, `run_id` 20261109T050613Z, labelled `ynn-reduce-off`: 2 cells, both `ok`. That is `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=-reduce`, the fusion that slows once a float64 input passes 2 GiB (QuantEcon/project-benchmarks-jax#7). P1's stored rewards are 1.8 GB at 1,500.
- **Each CPU's clock, every 2 s,** from 03:21 to 05:10, in `clock.jsonl`, from [`problems/checks/clock_sampler.py`](../../../checks/clock_sampler.py). Each line has every CPU's MHz from `/proc/cpuinfo` (on this kernel, the clock it actually ran at) and its busy share since the last line, so a serial cell's clock is the busy CPU's. It started 8 minutes into the full run, so it misses P1's cells up to `numpy` at 1,000.
- **One core's bandwidth,** 05:10, in `bandwidth.jsonl`, from [`problems/checks/bw_probe.py`](../../../checks/bw_probe.py) as at `51b698f`, pinned to CPU 0 with every thread pool at one. The sampler ran beside it.

`summary.md` is this host's tables, from `python problems/summarise.py problems/results/` at `2f66a4c` once the `--quick` rows were left out. The run's own summary step failed, as `session.log` shows: before `2f66a4c`, `summarise.py` read every `.jsonl` under `problems/results/`, `clock.jsonl` among them.

## A second instance, for the triad

The triad, [`problems/checks/stream_triad.py`](../../../checks/stream_triad.py), was asked for after this instance had ended. So a second `c8a.4xlarge`, launched the same way with the same image and CPU, ran it and the bandwidth probe again. That instance was up from 05:16 to 05:31 UTC, and its files are in [`second-instance/`](second-instance/): `machine.txt`, `environment.txt`, `bandwidth.jsonl`, `triad.jsonl`, `clock.jsonl`, and `session/`, with the script it ran and its log.

- **The probe** ran at 05:16:53 and **the triad** at 05:16:59, at 1, 2, 4, 8 and 16 threads, best of 10. The arrays were first touched on all 16 threads. The instance has one NUMA node, so the lower thread counts read pages in the same node.
- **The triad that ran** was the draft of `stream_triad.py` with sha256 `8620dd65…`. It differs from the committed file (`e73525d3…`, `a17671a`) only in its docstring and in also running at the full thread count, which on 16 CPUs gives the same counts.
- **Its probe agrees with the first instance's** to within 5% on every operation.
- **It sat idle from 05:17 to 05:30.** The script started the clock sampler as a background child of the logged run, so the log's pipe stayed open, and the end-of-run line the sampler waits for never came. The sampler was stopped by hand at 05:30, after everything timed had finished.
