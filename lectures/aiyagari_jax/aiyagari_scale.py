"""
Time aiyagari_jax's household operators at larger asset grids, in one
precision, in one fresh process.

    python aiyagari_scale.py aiyagari_jax.md --dtype float64 --sizes 200,1000 --out s.json

It runs the lecture's own definition cells (up to `howard_policy_iteration`),
then for each grid size times:

- `hpi`: the lecture's Howard policy iteration, cold (compile included) and
  warm, with its loop capped at HPI_CAP. In float32 the lecture's stopping
  rule (max |v_new - v| <= 1e-4) can be tighter than float32 resolves, and
  the loop then runs to its own cap of 10,000; `hpi_capped` records that;
- `greedy`: the lecture's `get_greedy`, which recomputes log utility over
  every (a, z, a') on each call: FP64 arithmetic on a weak-FP64 card;
- `greedy_stored`: not in the lecture. The same step over a stored reward
  array, so it reads memory instead of computing logs;
- `value`: the lecture's `get_value`, a bicgstab solve.
"""
import time

T0 = time.perf_counter()

import argparse
import contextlib
import io
import json
import re
import statistics

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'
HPI_CAP = 200


def timed(fn, min_total=1.0, min_reps=5, max_reps=50):
    """Median of warm calls, after one untimed call."""
    import jax
    jax.block_until_ready(fn())
    times = []
    while len(times) < min_reps or (sum(times) < min_total and len(times) < max_reps):
        t = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t)
    return statistics.median(times), len(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lecture")
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--sizes", default="200,1000,2000,5000,10000")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cells = [m.group(1) for m in CELL.finditer(open(args.lecture).read())]
    last = next(i for i, c in enumerate(cells) if "def howard_policy_iteration" in c)
    cells = [c for c in cells[:last + 1] if "!nvidia-smi" not in c]
    if args.dtype == "float32":
        cells = [c.replace(X64_ON, X64_OFF) for c in cells]
    ns = {}
    for c in cells:
        exec(compile(c, "<lecture>", "exec"), ns)

    import jax
    import jax.numpy as jnp
    B, get_greedy, get_value = ns["B"], ns["get_greedy"], ns["get_value"]
    hpi, create_household, create_prices = (ns["howard_policy_iteration"],
                                            ns["create_household"], ns["create_prices"])

    @jax.jit
    def rewards(household, prices):
        """u(c) at every (a, z, a'), as B computes it with v = 0."""
        a_size, z_size = len(household.a_grid), len(household.z_grid)
        return B(jnp.zeros((a_size, z_size)), household, prices)

    @jax.jit
    def greedy_stored(R, v, household):
        β, a_grid, z_grid, Π = household
        a_size, z_size = len(a_grid), len(z_grid)
        EV = jnp.sum(jnp.reshape(v, (1, 1, a_size, z_size))
                     * jnp.reshape(Π, (1, z_size, 1, z_size)), axis=-1)
        return jnp.argmax(R + β * EV, axis=-1)

    devices = jax.devices()
    row = {"dtype": args.dtype, "platform": devices[0].platform,
           "device_kind": devices[0].device_kind, "jax": jax.__version__,
           "default_float": jnp.zeros(()).dtype.name, "sizes": []}
    for n in map(int, args.sizes.split(",")):
        household, prices = create_household(a_size=n), create_prices()
        cell = {"a_size": n, "cells": n * n * len(household.z_grid)}
        t = time.perf_counter()
        σ = jax.block_until_ready(hpi(household, prices, max_iter=HPI_CAP))
        cell["hpi_cold_s"] = time.perf_counter() - t
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            t = time.perf_counter()
            σ = jax.block_until_ready(hpi(household, prices, max_iter=HPI_CAP, verbose=True))
            cell["hpi_warm_s"] = time.perf_counter() - t
        cell["hpi_iterations"] = out.getvalue().count("Concluded loop")
        cell["hpi_capped"] = cell["hpi_iterations"] >= HPI_CAP
        v = get_value(σ, household, prices)
        cell["greedy_s"], cell["greedy_reps"] = timed(lambda: get_greedy(v, household, prices))
        R = jax.block_until_ready(rewards(household, prices))
        cell["greedy_stored_s"], _ = timed(lambda: greedy_stored(R, v, household))
        cell["stored_matches"] = bool(jnp.all(greedy_stored(R, v, household)
                                              == get_greedy(v, household, prices)))
        del R
        cell["value_s"], _ = timed(lambda: get_value(σ, household, prices))
        cell["policy"] = jnp.ravel(σ).tolist()
        cell["v_sum"] = float(jnp.sum(v))
        row["sizes"].append(cell)
        row["process_s"] = time.perf_counter() - T0
        with open(args.out, "w") as f:          # after each size, so a slow one loses nothing
            json.dump(row, f)
        print(f"{args.dtype} {row['platform']} a_size={n}: hpi warm {cell['hpi_warm_s']:.3f} s "
              f"({cell['hpi_iterations']} it), greedy {cell['greedy_s'] * 1e3:.2f} ms, "
              f"stored {cell['greedy_stored_s'] * 1e3:.2f} ms, value {cell['value_s'] * 1e3:.2f} ms",
              flush=True)


if __name__ == "__main__":
    main()
