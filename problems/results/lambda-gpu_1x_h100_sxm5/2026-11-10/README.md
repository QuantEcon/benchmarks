# P7's Kalman filters by filters and states on an H100, 2026-11-10

For [#58](https://github.com/QuantEcon/benchmarks/issues/58): the TITAN RTX's half of #58 on a card with full float64, to see whether the crossover rule found there carries to float64. It's QuantEcon/project-benchmarks-jax#58's handover, run by a session on Matt's laptop. The rows are in [`../2026-11-10.jsonl`](../2026-11-10.jsonl), labelled `kalman-grid`, and `summary.md` tables them. The CPU side is [the M6's](../../qe-mini1/2026-11-10/README.md) and [the M5 Pro's](../../qe-wharton2/2026-11-10/README.md) rows of the same night.

| | |
|---|---|
| Host | Lambda's `gpu_1x_h100_sxm5` in us-southeast-1, Lambda's default image (Ubuntu 24.04.4): an H100 80GB HBM3, driver 580.105.08, power limit 700 W, beside 26 vCPUs of a Xeon Platinum 8480+ (`machine.txt`) |
| Software | QuantEcon/benchmarks at `314b061`, through [`run_by_hand.sh`](../../../run_by_hand.sh), which installed `jax[cuda13]` 0.12.0.1 (`environment.txt`) |
| When | 2026-11-10: the check 20:45:58 to 20:46:21 UTC, the full run 20:46:57 to 20:51:15 |
| Commands | `PROBLEMS=kalman SWEEPS=batch20,batch50,batch100 RUN_LABEL=kalman-grid bash problems/run_by_hand.sh lambda-gpu_1x_h100_sxm5 --quick`, then `PROBLEMS=kalman SWEEPS=batch,state,batch20,batch50,batch100 TIMEOUT=1800 RUN_LABEL=kalman-grid bash problems/run_by_hand.sh lambda-gpu_1x_h100_sxm5` |

`run.log` has both runs' output. The `.jsonl` keeps the full run's 66 rows (`20261110T204658Z`) and leaves out the check's 6. `session/notes.md` is the laptop session's record.

## Checks

- **All 66 cells ran,** status `ok`. The longest was 10,000 filters at 100 states in float64, 39 s for the process.
- **Every answer agrees with NumPy's in float64** at the same size, on the M6 and the M5 Pro: `summary.md` lists no exceptions.
- **`batch` and `state` agree with 2026-11-06's** ([its run](../2026-11-06/README.md), `jax[cuda12]` on driver 570.148.08) within 7% warm, at a median 0.97 of that day's times. Cold they're 0 to 12% faster.
- **The card's clock:** in 64 of the 66 rows, `gpu_telemetry` shows no clock event reason, at 1,980 MHz. The two largest float64 cells, 3,000 and 10,000 filters at 100 states, reached the 700 W limit (694 W to 697 W) and show the software power cap, at 1,860 MHz and 1,815 MHz. That's the card at full load.
- **Cold,** a fresh process at 10 filters took 0.9 s to 2.1 s, from 10 to 100 states.
