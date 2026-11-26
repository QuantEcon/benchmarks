# `digitalocean-gpu-mi300x1-192gb`, 2026-11-06: the problem-type suite on an MI300X

This is the suite's `jax-gpu` backend on one AMD Instinct MI300X (`gfx942`), under ROCm 10.1. It was run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh), for QuantEcon/project-benchmarks-jax#32. The same session ran Phase 2's precision set. Its run notes, [`precision/runs/2026-11-06/digitalocean-gpu-mi300x1-192gb/`](../../../../precision/runs/2026-11-06/digitalocean-gpu-mi300x1-192gb/README.md), cover the droplet, the card and the session's scripts and logs.

## The runs

- **The `--quick` check,** 02:59 to 03:01 UTC. Its nine rows (`run_id` 20261106T025951Z) are left out of the `.jsonl`, since the full run repeats each of their cells.
- **The full run,** 03:17 to 03:25, `run_id` 20261106T031739Z: every size in both precisions, 50 cells, all `ok` (`run.log`).
- **Every float32 sweep again with `JAX_DEFAULT_MATMUL_PRECISION=highest`,** 03:25 to 03:27, `run_id` 20261106T032508Z: 22 cells, all `ok`, labelled `highest` (`run-highest.log`).
- **Each step's exit:** each step ran through the session's `step.sh`, whose log for the step ends with its exit status and wall time. Every step exited 0.

`summary.md` is this host's tables, from `python problems/summarise.py problems/results/` once the `--quick` rows were left out. It adds the `highest` tables to what the run wrote.

## What they show

- **At JAX's default precision, P3's float32 answers are wrong.**
    - They're 0.25 to 0.26 off NumPy's float64 answer, after 494 iterations at 200 points and 629 to 901 at 1,000 to 50,000.
    - On this card, float32 matrix products are rounded to TF32's 10 fraction bits by truncation. The counts and errors match `problems/checks/tf32_egm.py`'s truncated case, not the case rounded to nearest that matched the NVIDIA cards.
    - With `highest` they agree, after 2,204 to 2,212 iterations, as on the MI350X, which does full float32 by default.
- **Everything else agrees with NumPy:** P1, which runs in float64 only, P3 in float64, and P4 and P7 in both precisions. P7's float32 answers lose accuracy at the default but stay within the check, at up to 7.1e-4 off.
- **Its float64 compiles are slow:** a Kalman filter takes 14 s to 15 s cold, against about 3 s on the MI350X.
