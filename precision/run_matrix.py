"""
Run the precision benchmark matrix, one fresh process per cell.

    python precision/run_matrix.py --devices cpu,gpu --quick

Each cell (workload x size x precision x device) runs in its own Python
process, so compile time is counted and nothing carries over between cells.
Each appends one JSON line to results/<host>/<date>.jsonl.

Devices are `cpu`, `gpu` (one card) and `gpu:N` (N cards, sharded; only
for the workloads that shard). `cpu:N` forces N host devices, which tests
the sharded code path on a laptop but measures nothing useful.
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

import env_info
from spec import GIB, ITEMSIZE, WORKLOADS

HERE = Path(__file__).resolve().parent
MEMORY_FRACTION = 0.8
CELL_FLAGS = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "CUDA_VISIBLE_DEVICES",
              "CUDA_DEVICE_ORDER", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE",
              "XLA_PYTHON_CLIENT_MEM_FRACTION")


def parse_device(device):
    kind, _, count = device.partition(":")
    if kind not in ("cpu", "gpu"):
        raise SystemExit(f"unknown device {device!r}; use cpu, gpu or gpu:N")
    return kind, int(count or 1)


def cell_env(kind, count, dtype, gpus):
    """The environment a cell's process starts with: precision and device."""
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
    if kind == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env.pop("CUDA_VISIBLE_DEVICES", None)
        if count > 1:
            flag = f"--xla_force_host_platform_device_count={count}"
            env["XLA_FLAGS"] = f"{env.get('XLA_FLAGS', '')} {flag}".strip()
    else:
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus[:count])
        # Even with preallocation off, JAX's allocator stops at 75% of the card by
        # default, below --memory-fraction; this leaves that check as the only limit
        env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    return env


def capacity(kind, count, gpus, environment):
    """Bytes available per device, for skipping cells that cannot fit."""
    if kind == "cpu":
        return environment["memory_bytes"] / count
    cards = {str(c["index"]): c for c in environment["gpus"]}
    used = [cards[i] for i in gpus[:count] if i in cards]
    if len(used) < count:
        raise SystemExit(f"asked for {count} GPU(s) from --gpus {','.join(gpus)}, "
                         f"but nvidia-smi lists {sorted(cards) or 'none'}")
    return min(c["memory_total_mib"] for c in used) * 2**20


def run_cell(prefix, workload, size, dtype, reps, env, timeout):
    cmd = prefix + [sys.executable, str(HERE / "run_cell.py"), "--workload", workload,
                    "--size", str(size), "--dtype", dtype, "--reps", str(reps)]
    t = time.perf_counter()
    try:
        out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
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


def describe(row, unit, work):
    if row["status"] != "ok":
        return row["status"] + (f" ({row.get('reason') or row.get('error', '')})"[:120]
                                if row.get("reason") or row.get("error") else "")
    t = row["warm_median_s"]
    return (f"warm {t * 1e3:9.2f} ms  {work / t:9.1f} {unit}  "
            f"compile {row['compile_s']:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workloads", default=",".join(WORKLOADS))
    parser.add_argument("--devices", default="cpu")
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument("--sizes", help="override every selected workload's sizes")
    parser.add_argument("--quick", action="store_true", help="the small sizes only")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--gpus", default="0",
                        help="GPU indices in PCI bus order; gpu uses the first, gpu:N the first N")
    parser.add_argument("--prefix", default="",
                        help="command to wrap each cell in, e.g. 'numactl --interleave=all'")
    parser.add_argument("--label", default="", help="tag for this run, e.g. interleave")
    parser.add_argument("--host", default=socket.gethostname().split(".")[0])
    parser.add_argument("--out", default=str(HERE.parent / "results"))
    parser.add_argument("--timeout", type=int, default=900, help="seconds per cell")
    parser.add_argument("--memory-fraction", type=float, default=MEMORY_FRACTION,
                        help="skip cells needing more than this share of a device's memory; "
                             "lower it on a machine that is in use")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workloads = args.workloads.split(",")
    unknown = [w for w in workloads if w not in WORKLOADS]
    if unknown:
        raise SystemExit(f"unknown workload(s) {unknown}; choose from {list(WORKLOADS)}")
    devices = [parse_device(d) for d in args.devices.split(",")]
    dtypes = args.dtypes.split(",")
    gpus = args.gpus.split(",")
    prefix = shlex.split(args.prefix)

    environment = env_info.collect()
    # Check every device before running anything, so a missing GPU fails fast
    available = {(kind, count): capacity(kind, count, gpus, environment)
                 for kind, count in devices}
    now = datetime.now(timezone.utc)
    path = Path(args.out) / args.host / f"{now:%Y-%m-%d}.jsonl"
    base = {"run_id": now.strftime("%Y%m%dT%H%M%SZ"), "host": args.host,
            "label": args.label, "prefix": args.prefix, "env": environment}
    if not args.dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing to {path}")

    for name in workloads:
        spec = WORKLOADS[name]
        sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes
                 else spec["quick"] if args.quick else spec["sizes"])
        for kind, count in devices:
            if count > 1 and not spec["shardable"]:
                continue
            device = kind if count == 1 else f"{kind}:{count}"
            # A workload can say that the CPU needs more than a GPU does
            memory = spec["memory_cpu"] if kind == "cpu" and "memory_cpu" in spec else spec["memory"]
            for size in sizes:
                for dtype in dtypes:
                    s = ITEMSIZE[dtype]
                    need = memory(size, s, count)
                    row = dict(base, timestamp=datetime.now(timezone.utc).isoformat(),
                               workload=name, bound=spec["bound"], size=size, dtype=dtype,
                               device=device, unit=spec["unit"], work=spec["work"](size, s))
                    env = cell_env(kind, count, dtype, gpus)
                    row["cell_flags"] = {k: env[k] for k in CELL_FLAGS if k in env}
                    limit = args.memory_fraction * available[(kind, count)]
                    if need > limit:
                        row.update(status="skipped",
                                   reason=f"needs ~{need / GIB:.1f} GiB per device, "
                                          f"limit {limit / GIB:.1f} GiB")
                    elif args.dry_run:
                        row["status"] = "planned"
                    else:
                        row.update(run_cell(prefix, name, size, dtype, args.reps, env,
                                            args.timeout))
                    print(f"{name:18s} {size:>7d} {dtype:8s} {device:6s} "
                          f"{describe(row, spec['unit'], row['work'])}", flush=True)
                    if not args.dry_run:
                        with path.open("a") as f:
                            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
