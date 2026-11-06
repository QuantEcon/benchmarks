# `digitalocean-gpu-mi350x1-288gb-spot`, 2026-11-05: the problem-type suite on an MI350X

This is the suite's `jax-gpu` backend on one AMD Instinct MI350X (`gfx950`), under ROCm 10.1. It was run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh), for QuantEcon/project-benchmarks-jax#32. The same session ran Phase 2's precision set. Its run notes, [`precision/runs/2026-11-05/`](../../../../precision/runs/2026-11-05/README.md), cover:
- the droplet, the image and the card;
- the `gfx950` libraries the image lacked, and how they were added;
- the session's scripts and logs.

## The runs

- **The `--quick` check,** 23:12 to 23:13 UTC. Its nine rows (`run_id` 20261105T231248Z) are left out of the `.jsonl`, since the full run repeats each of their cells. Its output is in the session's logs.
- **The full run,** 23:34 to 23:38. Its `run_id` is 20261105T233419Z: every size in both precisions, 50 cells, all `ok`, with output in `run.log`.
- **Every float32 sweep again with `JAX_DEFAULT_MATMUL_PRECISION=highest`,** 23:38 to 23:40, labelled `highest` as on the RTX PRO 6000. Its `run_id` is 20261105T233840Z: 22 cells, all `ok` (`run-highest.log`).

`summary.md` is this host's tables, from `python problems/summarise.py problems/results/` once the `--quick` rows were left out.

## What they show

- **Every answer agrees with NumPy's float64 answer,** at JAX's default precision and under `highest`.
    - The only lines under the tables are P3's float32 iteration counts: 2,204 to 2,212, against NumPy float64's 2,190 to 2,192.
    - The TITAN RTX's counts were 14 to 26 more than float64's. Here they're 14 to 22 more.
- **At the default, a float32 matrix product on this card is full float32.**
    - P3's float32 iteration counts are the same under `highest`, size by size.
    - Its warm times agree to within 4%: 0.36 s to 0.99 s.
    - On the RTX PRO 6000, TF32 cut the counts to 809 to 1,381, and the answers were 11% to 12% off.
- **The Kalman filter is slow here.** P7 takes 0.24 s to 0.33 s warm for a filter of 1,000 steps, at every state size and for batches up to 1,000, and 0.49 s to 0.58 s for 10,000 filters.
