# The RTX PRO 6000 held to 300 W, on aws-g7e.2xlarge, 2026-11-18

For QuantEcon/project-benchmarks-jax#64: the problem types and the precision matrix on an RTX PRO 6000 Blackwell Server Edition, with its power limit set to 300 W, the limit of the card's Max-Q Workstation Edition. It ran by hand from a laptop session, from QuantEcon/benchmarks at `f085c6d`, from 01:18 UTC. The rows are labelled `pl300`, and `pl300-highest` for the float32 runs again under `JAX_DEFAULT_MATMUL_PRECISION=highest`.

| File | What it holds |
|---|---|
| `../2026-11-18.jsonl` | The problem types at 300 W: 50 `pl300` cells and 22 `pl300-highest`, all ok |
| `../../../../results/aws-g7e.2xlarge/2026-11-18.jsonl` | The precision matrix at 300 W: 46 `pl300` rows and 12 `pl300-highest`, all ok |
| `run.log`, `run-pl300-highest.log`, `environment.txt`, `machine.txt` | The problem types' log, the `highest` pass's, the packages, and `nvidia-smi` at the start, under the cap |
| `summary.md` | `problems/summarise.py`'s tables for the `pl300` labels |
| `card/` | The card's record (`card.csv`, `nvidia-smi-q.txt`, `host.txt`), the commit, the cap test (`pl-test.txt`), `nvidia-smi` once a second under the cap (`smi-pl300.csv`) and the limit read back afterwards (`power-after.txt`). The card's serial number and UUID are redacted in `nvidia-smi-q.txt` |
| `session/run.log` | The whole session's log, the checks at 600 W included |

**The card:** an NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, VBIOS 98.02.9E.00.03, on driver 595.112.07 with CUDA 13.2. Its limit can be set from 300 to 600 W, and is 600 W by default. JAX 0.12.0.1 with `jax[cuda13]`, Python 3.12.3.

**The cap:** `nvidia-smi -pl 300` was allowed, and the limit read 300 W throughout. `smi-pl300.csv` samples the card once a second from 01:20:39 to 01:35:39 UTC:
- a mean draw of 124 W, and a maximum of 317 W;
- 18 seconds at 290 W or more;
- 27 seconds with `SwPowerCap` among the clock-event reasons.

So the cap binds only on the heaviest cells, and each row's `gpu_telemetry` shows which.

Afterwards the limit was set back to 600 W. `power-after.txt`, read straight after `-pl 600`, still shows 300 W, but a read a second later shows 600 W (`session/run.log`). So read a limit back after a short pause.

**Left out of the `.jsonl` files:** the 600 W `--quick` checks run first: 9 problem cells, and 18 precision rows, 4 of them `highest`. They're in `session/run.log`. The suite keeps the latest row for each cell, host and label, so these would have displaced the full 600 W runs below in those cells. The quick check's lecture and scale runs, under `lectures/results/`, are left out too.

## What they show

The rows at 600 W that they're compared with come from the same instance type, but on another day and so a different card: the problem types of 2026-11-05 and the precision matrix of 2026-10-26, on driver 595.91.07.

- **The problem types barely notice the cap.** At 300 W, 48 of the 50 warm medians are within 5% of the 600 W ones, and the median ratio is 0.997. The two exceptions are `vfi` at 1,500 in float64, 7% slower, and `wealth` at 10,000,000 in float32, 16% slower but taking only 17 ms. Under `highest`, 21 of 22 are within 5%.
- **Neither do float64 matrix products.** At n = 4,096 and 8,192, `matmul` in float64 runs at 1,744–1,758 GFLOPS at both limits.
- **Large float32 matrix products do.** At JAX's default precision, `matmul` falls from 166 to 131 TFLOPS at n = 4,096, and from 186 to 147 at 8,192, a fifth less. Under `highest` it falls from 68 to 54 and from 83 to 55, a fifth to a third less.
- **Across all 46 of the precision matrix's cells,** the median ratio of warm times is 0.999.
