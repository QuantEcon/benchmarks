# `digitalocean-gpu-h200x1-141gb`, 2026-11-06: the problem-type suite on an H200

This is the suite's `jax-gpu` backend on one NVIDIA H200, 141 GB. It was run by hand through [`problems/run_by_hand.sh`](../../../run_by_hand.sh), for QuantEcon/project-benchmarks-jax#37. The same session then ran Phase 2's precision set. Its run notes, [`precision/runs/2026-11-06/`](../../../../precision/runs/2026-11-06/README.md), cover the droplet and the card, with the session's scripts and logs.

## The runs

- **The `--quick` check,** 01:26 to 01:27 UTC. Its nine rows (`run_id` 20261106T012644Z) are left out of the `.jsonl`, since the full run repeats each of their cells.
- **The full run,** 01:27 to 01:29, `run_id` 20261106T012730Z: every size in both precisions, 50 cells, all `ok` (`run.log`). The driver is 580.196.07, so `jax[cuda13]` installed, and the rows carry no label.
- **Every float32 sweep again with `JAX_DEFAULT_MATMUL_PRECISION=highest`,** 01:29 to 01:30, `run_id` 20261106T012931Z: 22 cells, all `ok`, labelled `highest` (`run-highest.log`).
- **Each step's exit:** each step ran through the session's `step.sh`, whose log for the step ends with its exit status and wall time. Every step exited 0.

`summary.md` is this host's tables, from `python problems/summarise.py problems/results/` once the `--quick` rows were left out. It adds the `highest` tables to what the run wrote.

## What they show

- **TF32, as on the H100 and the RTX PRO 6000.**
    - At XLA's default precision, P3's float32 answers differ from NumPy's float64 answer by a relative 0.11 to 0.12, after 809 to 1,381 iterations. Those are the other two cards' counts, size by size.
    - With `highest` they agree, after 2,208 to 2,214 iterations.
- **Everything else agrees with NumPy:** P1, which runs in float64 only, P3 in float64, and P4 and P7 in both precisions.
- **Against the H100,** the H200 is 7% to 14% faster at each problem's largest size, in float64, and no faster where each step's fixed cost dominates: a single small filter, and P3 up to 5,000 points.
- **Compiles are quick:** no cell's cold time passed 1.95 s.
