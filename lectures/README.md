# Lectures, run as used

Whole QuantEcon lectures, run the way a reader or a CI build runs them: every code cell in order, in one fresh Python process, with compile time counted. `run_lecture.py` times each cell, so a lecture's cost can be traced to the cells that carry it. It can also switch a float64 lecture to float32, to price the precision.

This is the as-used method of [meta#335](https://github.com/QuantEcon/meta/issues/335), applied to a whole lecture. The suite in `precision/` measures one kernel at a time; this measures what the reader waits for.

## Running a lecture

Fetch the lecture at a fixed commit, then run it in an environment that has the lecture's imports:

```bash
gh api 'repos/QuantEcon/lecture-jax/contents/lectures/aiyagari_jax.md?ref=4b939ec' \
    -H 'Accept: application/vnd.github.raw' > aiyagari_jax.md
JAX_PLATFORMS=cuda CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python lectures/run_lecture.py aiyagari_jax.md --dtype float64 --out run.json \
    --env-info precision --record "K_star"
```

- **`--dtype float32`** rewrites the lecture's `jax.config.update("jax_enable_x64", True)` line to `False`. The lecture must have exactly one such line.
- **`--record`** evaluates expressions in the lecture's namespace after the last cell, for comparing the two precisions.
- **`--env-info`** points at `precision/`, whose `env_info.py` writes the same environment block as the precision benchmarks.
- **Cells that only `!pip install` are skipped** and listed under `skipped`, so a run neither changes its environment nor times a download. Install those packages beforehand.
- **Plots use matplotlib's Agg backend**, so `plt.show()` does nothing.

Each run writes one JSON file: the time and output tail of each cell, the total, the recorded values, the device and the flags.

## What is here

| Lecture | The question | Results |
|---|---|---|
| [`aiyagari_jax`](aiyagari_jax/) | project-compute's: on a TITAN RTX, is float64 bound by bandwidth or by FP64 arithmetic? Also, where does the card beat an Apple CPU? | [`qe-wharton1-gpu`](results/qe-wharton1-gpu/2026-10-25/aiyagari_jax/), [`qe-mini1`](results/qe-mini1/2026-10-25/aiyagari_jax/) |
| [`ifp_egm_jax`](ifp_egm_jax/) | The same questions, for a lecture that interpolates instead of searching a dense choice grid | [`qe-wharton1-gpu`](results/qe-wharton1-gpu/2026-10-25/ifp_egm_jax/), [`qe-mini1`](results/qe-mini1/2026-10-25/ifp_egm_jax/) |

Results live under `lectures/results/<host>/<date>/<lecture>/`, apart from the top-level `results/`, which holds the precision benchmarks' JSON lines.

## Comparing machines

[`compare_hosts.py`](compare_hosts.py) sets the TITAN RTX, its host's Threadripper and `qe-mini1`'s M6 side by side for both lectures:

```bash
python lectures/compare_hosts.py lectures/results/qe-wharton1-gpu/2026-10-25 lectures/results/qe-mini1/2026-10-25
```

**`qe-mini1` runs agents around the clock.** [`mac_set.sh`](mac_set.sh) runs its CPU-only set with the agents and the console's animated wallpaper paused (SIGSTOP), and resumes them (SIGCONT) on exit, even after a failure. `state.txt` beside the results records the machine before, during and after the run. On 2026-10-25 the Mac idled at under 2% CPU while paused.
