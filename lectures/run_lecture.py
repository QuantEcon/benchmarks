"""
Run a MyST lecture's code cells in one fresh process, as a notebook kernel
would, and time each cell.

    python run_lecture.py aiyagari_jax.md --dtype float32 --out run.json \
        --record "K_star" "K_vals.tolist()"

Cells run through IPython, so `%time` and `!cmd` work, with matplotlib's Agg
backend. --dtype as-written runs the lecture untouched, whatever precision
it sets. --dtype float32 rewrites the lecture's `jax_enable_x64` line to
False, and --dtype float64 runs a float64 lecture as written; both require
exactly one such line. --record evaluates expressions
in the lecture's namespace after the last cell, for comparing precisions.

Cells that only `!pip install` are skipped and listed in the output, so a run
neither changes its environment nor times a download. Install what they
install beforehand.
"""
import time

T0 = time.perf_counter()

import argparse
import contextlib
import io
import json
import os
import re
import sys
import traceback

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'
TAGS = re.compile(r"tags:\s*\[([^\]]*)\]")


def code_cells(text):
    """Each code cell's source, with MyST cell options removed, and its tags."""
    cells = []
    for match in CELL.finditer(text):
        lines = match.group(1).splitlines()
        options = []
        if lines and lines[0].strip() == "---":           # a YAML options block
            end = lines.index("---", 1) if "---" in lines[1:] else 0
            options, lines = lines[1:end], lines[end + 1:]
        while lines and lines[0].startswith(":"):          # `:tags:` style options
            options.append(lines.pop(0))
        tags = [tag.strip().strip("'\"") for m in map(TAGS.search, options) if m
                for tag in m.group(1).split(",") if tag.strip()]
        cells.append(("\n".join(lines), tags))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lecture")
    parser.add_argument("--dtype", choices=["as-written", "float32", "float64"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--record", nargs="*", default=[])
    parser.add_argument("--env-info", help="path to benchmarks' precision/ directory")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")

    cells = code_cells(open(args.lecture).read())
    installs = [s for s, _ in cells if s.strip() and all(
        line.strip().startswith("!pip install") for line in s.strip().splitlines())]
    cells = [(s, tags) for s, tags in cells if s not in installs]
    if args.dtype != "as-written" and sum(X64_ON in s for s, _ in cells) != 1:
        raise SystemExit(f"expected one `{X64_ON}` line in {args.lecture}")
    if args.dtype == "float32":
        cells = [(s.replace(X64_ON, X64_OFF), tags) for s, tags in cells]

    from IPython.core.interactiveshell import InteractiveShell
    shell = InteractiveShell.instance()
    row = {"lecture": os.path.basename(args.lecture), "dtype": args.dtype,
           "flags": {k: v for k, v in sorted(os.environ.items())
                     if k.startswith(("JAX_", "XLA_", "CUDA_"))},
           "skipped": installs, "cells": []}
    row["startup_s"] = time.perf_counter() - T0
    for i, (source, tags) in enumerate(cells):
        out = io.StringIO()
        t = time.perf_counter()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            result = shell.run_cell(source)
        elapsed = time.perf_counter() - t
        first = next((line for line in source.splitlines() if line.strip()), "")
        cell = {"index": i, "first_line": first[:90], "seconds": elapsed,
                "ok": result.success, "output": out.getvalue()[-3000:], "tags": tags,
                # The lecture marks a cell that raises on purpose, to show an error
                "expected_error": "raises-exception" in tags}
        if not result.success:
            error = result.error_in_exec or result.error_before_exec
            cell["error"] = repr(error)[:2000]
        row["cells"].append(cell)

    row["lecture_s"] = sum(c["seconds"] for c in row["cells"])
    row["all_ok"] = all(c["ok"] or c["expected_error"] for c in row["cells"])
    row["record"] = {}
    for expr in args.record:
        try:
            row["record"][expr] = shell.ev(expr)
        except Exception as e:
            row["record"][expr] = f"{type(e).__name__}: {e}"
    import jax
    import jax.numpy as jnp
    devices = jax.devices()
    row.update(jax=jax.__version__, platform=devices[0].platform,
               device_kind=devices[0].device_kind,
               default_float=jnp.zeros(()).dtype.name,
               x64=bool(jax.config.jax_enable_x64))
    if args.env_info:
        sys.path.insert(0, args.env_info)
        import env_info
        row["env"] = env_info.collect()
    row["process_s"] = time.perf_counter() - T0
    with open(args.out, "w") as f:
        json.dump(row, f, default=str)
    print(f"{args.dtype} {row['platform']}: lecture {row['lecture_s']:.1f} s, "
          f"all cells ok: {row['all_ok']}, default float {row['default_float']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
