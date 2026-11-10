# P7's Kalman filters by filters and states on an RTX PRO 6000, 2026-11-10

For [#58](https://github.com/QuantEcon/benchmarks/issues/58): the TITAN RTX's half of #58 on the card QuantEcon/project-compute#114's workstation is built around. Its float64 rate lies between Turing's and the H100's. It's QuantEcon/project-benchmarks-jax#58's handover, run by a session on Matt's laptop. The rows are in [`../2026-11-10.jsonl`](../2026-11-10.jsonl), labelled `kalman-grid`, and `summary.md` tables them. The CPU side is [the M6's](../../qe-mini1/2026-11-10/README.md) and [the M5 Pro's](../../qe-wharton2/2026-11-10/README.md) rows of the same night.

| | |
|---|---|
| Host | AWS's `g7e.2xlarge` in us-east-1, on the Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04), 2026-11-06: an RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 595.112.07, power limit 600 W, with 8 vCPUs (`machine.txt`) |
| Software | QuantEcon/benchmarks at `314b061`, through [`run_by_hand.sh`](../../../run_by_hand.sh), which installed `jax[cuda13]` 0.12.0.1 (`environment.txt`) |
| When | 2026-11-10: the check 20:48:28 to 20:49:55 UTC, the full run 20:50:19 to 21:05:27 |
| Commands | As on the H100 (its [notes](../../lambda-gpu_1x_h100_sxm5/2026-11-10/README.md)), with the host `aws-g7e.2xlarge` |

`run.log` has both runs' output. The `.jsonl` keeps the full run's 66 rows (`20261110T205020Z`) and leaves out the check's 6. `session/notes.md` is the laptop session's record.

## Checks

- **All 66 cells ran,** status `ok`. The longest was 10,000 filters at 100 states in float64, 189 s for the process.
- **Every answer agrees with NumPy's in float64** at the same size, on the M6 and the M5 Pro: `summary.md` lists no exceptions.
- **`batch` and `state` agree with 2026-11-05's** ([its run](../2026-11-05/README.md)) within 6% warm, at a median 1.00 of that day's times, and within 5% cold.
- **Its fresh processes compile slowly, as on 2026-11-05.** At 10 filters, a cold cell took 1.0 s to 1.2 s at 10 states, 5.3 s and 10.5 s at 50 (float32 and float64), and 9.3 s and 17.4 s at 100. On the H100 the same cells took 0.9 s to 2.1 s.
- **The card ran unthrottled:** no row's `gpu_telemetry` shows a clock event reason. The SM clock was 2,272 MHz to 2,422 MHz, 2,332 at the median.
