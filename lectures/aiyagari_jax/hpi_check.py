"""
How many Howard policy iterations does aiyagari_jax's G take, by precision?

    python hpi_check.py aiyagari_jax.md --dtype float32

For the lecture's 50 capital values (its "visual inspection" cell), run the
lecture's howard_policy_iteration with verbose=True, and count its loops and
the error it stopped at. The loop stops when max |v_new - v| <= 1e-4, or after
10,000 loops.
"""
import argparse
import contextlib
import io
import re
import statistics
import time

import numpy as np

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'

parser = argparse.ArgumentParser()
parser.add_argument("lecture")
parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
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

firm, household = ns["create_firm"](), ns["create_household"]()
loops, last_errors, seconds = [], [], []
for K in np.linspace(4, 12, 50):
    r = ns["r_given_k"](K, firm)
    prices = ns["create_prices"](r=r, w=ns["r_to_w"](r, firm))
    out = io.StringIO()
    t = time.perf_counter()
    with contextlib.redirect_stdout(out):
        σ = jax.block_until_ready(ns["howard_policy_iteration"](household, prices, verbose=True))
    seconds.append(time.perf_counter() - t)
    lines = [l for l in out.getvalue().splitlines() if l.startswith("Concluded loop")]
    loops.append(len(lines))
    last_errors.append(float(lines[-1].rsplit("error", 1)[1].strip(" .")) if lines else float("nan"))

print(f"{args.dtype} on {jax.devices()[0].platform}: loops per solve min {min(loops)}, "
      f"median {statistics.median(loops):.0f}, max {max(loops)}; "
      f"{sum(n >= 10_000 for n in loops)} of 50 hit the 10,000 cap; "
      f"total {sum(seconds):.1f} s, {statistics.median(seconds) * 1e3:.0f} ms median per solve")
print(f"  loops at each K: {loops}")
print(f"  error at stop, median {statistics.median(last_errors):.1e}, max {max(last_errors):.1e}")
