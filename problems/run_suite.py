"""
Run the problem-type suite, one fresh process per cell.

    python problems/run_suite.py --quick
    python problems/run_suite.py --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu

Each cell (problem x size x backend x precision) runs in its own Python
process, so compile time is counted and nothing carries over between cells.
Each appends one JSON line to problems/results/<host>/<date>.jsonl.

At each size every backend runs before the next size starts, so a slow
spell on the machine lands on one size rather than on one backend. Once a
backend times out, its larger sizes in that sweep are recorded as skipped.
"""
import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from spec import BACKENDS, GIB, ITEMSIZE, PROBLEMS

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "precision"))
import env_info  # noqa: E402

MEMORY_FRACTION = 0.8
CELL_FLAGS = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "CUDA_VISIBLE_DEVICES",
              "CUDA_DEVICE_ORDER", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
              "XLA_PYTHON_CLIENT_MEM_FRACTION", "NUMBA_NUM_THREADS")


def cell_env(backend, dtype, gpus):
    """The environment a cell's process starts with: precision and device, for JAX."""
    env = dict(os.environ)
    if backend.startswith("jax"):
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
    if backend == "jax-cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif backend == "jax-gpu":
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = gpus[0]
        # As in precision/: JAX's allocator otherwise stops at 75% of the card
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    return env


def capacity(backend, gpus, environment):
    """Bytes available to a cell, for skipping cells that cannot fit."""
    if backend != "jax-gpu":
        return environment["memory_bytes"]
    cards = {str(c["index"]): c for c in environment["gpus"]}
    if gpus[0] not in cards:
        raise SystemExit(f"jax-gpu asks for GPU {gpus[0]}, "
                         f"but nvidia-smi lists {sorted(cards) or 'none'}")
    return cards[gpus[0]]["memory_total_mib"] * 2**20


def numpy_blas():
    """The BLAS NumPy was built against, which its linear algebra runs on."""
    try:
        return np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except Exception:
        return None


def run_cell(prefix, cell, reps, budget, env, timeout):
    cmd = prefix + [sys.executable, str(HERE / "run_cell.py"),
                    "--problem", cell["problem"], "--sweep", cell["sweep"],
                    "--size", str(cell["size"]), "--backend", cell["backend"],
                    "--dtype", cell["dtype"], "--reps", str(reps), "--budget", str(budget)]
    t = time.perf_counter()
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout,
                             cwd=HERE)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "process_s": time.perf_counter() - t}
    elapsed = time.perf_counter() - t
    lines = [line for line in out.stdout.splitlines() if line.strip()]
    try:
        row = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        row = {"status": "error", "error": f"exit {out.returncode}, no result",
               "stderr": out.stderr[-4000:]}
    row["process_s"] = elapsed
    return row


def describe(row):
    if row["status"] != "ok":
        detail = row.get("reason") or row.get("error") or ""
        return row["status"] + (f" ({detail})"[:120] if detail else "")
    return (f"warm {row['warm_median_s']:9.3f} s  cold {row['cold_s']:9.3f} s  "
            f"{row['iterations']:>6d} iterations")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", default=",".join(PROBLEMS))
    parser.add_argument("--sweeps", help="only these sweeps, e.g. batch,state")
    parser.add_argument("--backends", default="numpy,numba,numba-parallel,jax-cpu")
    parser.add_argument("--dtypes", default="float64,float32",
                        help="each problem runs only in the precisions it allows")
    parser.add_argument("--sizes", help="override every selected sweep's sizes")
    parser.add_argument("--quick", action="store_true", help="the small sizes only")
    parser.add_argument("--reps", type=int, default=5, help="warm runs at most")
    parser.add_argument("--budget", type=float, default=60,
                        help="seconds of warm runs a cell may take, after at least one")
    parser.add_argument("--gpus", default="0", help="the GPU index, in PCI bus order, for jax-gpu")
    parser.add_argument("--prefix", default="",
                        help="command to wrap each cell in, e.g. 'numactl --interleave=all'")
    parser.add_argument("--label", default="", help="tag for this run")
    parser.add_argument("--host", default=socket.gethostname().split(".")[0])
    parser.add_argument("--out", default=str(HERE / "results"))
    parser.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    parser.add_argument("--memory-fraction", type=float, default=MEMORY_FRACTION,
                        help="skip cells needing more than this share of a device's memory")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    problems = args.problems.split(",")
    backends = args.backends.split(",")
    for names, known in ((problems, PROBLEMS), (backends, BACKENDS)):
        unknown = [n for n in names if n not in known]
        if unknown:
            raise SystemExit(f"unknown {unknown}; choose from {list(known)}")
    dtypes = args.dtypes.split(",")
    gpus = args.gpus.split(",")
    prefix = shlex.split(args.prefix)

    environment = env_info.collect()
    environment["numpy_blas"] = numpy_blas()
    # Check every device before running anything, so a missing GPU fails fast
    available = {b: capacity(b, gpus, environment) for b in backends}
    now = datetime.now(timezone.utc)
    path = Path(args.out) / args.host / f"{now:%Y-%m-%d}.jsonl"
    base = {"run_id": now.strftime("%Y%m%dT%H%M%SZ"), "host": args.host,
            "label": args.label, "prefix": args.prefix, "env": environment}
    if not args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {path}")

    for name in problems:
        spec = PROBLEMS[name]
        for sweep, plan in spec["sweeps"].items():
            if args.sweeps and sweep not in args.sweeps.split(","):
                continue
            sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
                     else plan["quick"] if args.quick else plan["sizes"])
            for dtype in (d for d in dtypes if d in spec["dtypes"]):
                timed_out = set()
                for size in sizes:
                    for backend in backends:
                        cell = dict(problem=name, sweep=sweep, size=size,
                                    backend=backend, dtype=dtype)
                        row = dict(base, timestamp=datetime.now(timezone.utc).isoformat(),
                                   problem_type=spec["label"], **cell)
                        env = cell_env(backend, dtype, gpus)
                        row["cell_flags"] = {k: env[k] for k in CELL_FLAGS if k in env}
                        need = spec["memory"](sweep, size, ITEMSIZE[dtype])
                        limit = args.memory_fraction * available[backend]
                        if need > limit:
                            row.update(status="skipped",
                                       reason=f"needs ~{need / GIB:.1f} GiB, "
                                              f"limit {limit / GIB:.1f} GiB")
                        elif backend in timed_out:
                            row.update(status="skipped", reason="timed out at a smaller size")
                        elif args.dry_run:
                            row["status"] = "planned"
                        else:
                            row.update(run_cell(prefix, cell, args.reps, args.budget, env,
                                                args.timeout))
                            if row["status"] == "timeout":
                                timed_out.add(backend)
                        print(f"{spec['label']} {name:7s} {sweep:10s} {size:>9d} {dtype:8s} "
                              f"{backend:15s} {describe(row)}", flush=True)
                        if not args.dry_run:
                            with path.open("a") as f:
                                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
