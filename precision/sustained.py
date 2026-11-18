"""
Hold one of the precision suite's kernels on a card for a set time, and time every step.

precision/sustained.sh drives it, with nvidia-smi logging the card once a second meanwhile.

    python precision/sustained.py run --workload bellman_stored --dtype float64 --size 40000 \
        --minutes 60 --out DIR
    python precision/sustained.py summarise DIR

run builds the kernel with kernels.py's builder and compiles it once. Then it calls it back to
back for --minutes, waiting for each step to finish before starting the next. For each second in
which steps ended, it writes a row to DIR/<workload>-<dtype>-<size>.csv: the UTC second, the
seconds since the start, the number of steps, their median, minimum and maximum times in ms, and
the rate at the median, in the workload's unit (spec.py). DIR/<workload>-<dtype>-<size>.json
records the build, the compile, the first run, the allocator's counts and the machine
(env_info.py). Rows are flushed as they're written, so a run that's stopped keeps what it timed.

summarise compares each workload's first minutes with its last, in its steps and in
nvidia-smi.csv's samples over the same seconds. That's the first 5 minutes against the last 30 for
an hour's run, scaled down in proportion for a shorter one. It also reports the card in the
last minute of the idle period before each workload, from phases.csv. It prints markdown and
writes DIR/summary.json.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# NVML's clock-event reasons, by bit (nvmlClocksEventReason* in nvml.h)
REASONS = {
    0x1: "GpuIdle",
    0x2: "ApplicationsClocksSetting",
    0x4: "SwPowerCap",
    0x8: "HwSlowdown",
    0x10: "SyncBoost",
    0x20: "SwThermalSlowdown",
    0x40: "HwThermalSlowdown",
    0x80: "HwPowerBrakeSlowdown",
    0x100: "DisplayClockSetting",
}
# The columns of sustained.sh's nvidia-smi query, in its order
SMI_FIELDS = ["timestamp", "sm_mhz", "mem_mhz", "power_w", "temp_c", "reasons"]
# The card's fields in the summary, with the decimals each is shown to
CARD = [("sm_mhz", "SM clock, MHz", 0), ("mem_mhz", "Memory clock, MHz", 0),
        ("power_w", "Power, W", 1), ("temp_c", "Temperature, °C", 0)]


def utc(t):
    """An epoch time as ISO 8601 in UTC, to the second."""
    return dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch(text):
    """ISO 8601 in UTC, as utc() and `date -u +%FT%TZ` write it, as an epoch time."""
    return dt.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()


def run(args):
    # The precision is set before JAX starts, as run_matrix.py sets it for a cell
    os.environ["JAX_ENABLE_X64"] = "1" if args.dtype == "float64" else "0"
    sys.path.insert(0, str(HERE))
    import jax
    import jax.numpy as jnp
    import env_info
    import kernels
    from spec import ITEMSIZE, WORKLOADS

    spec = WORKLOADS[args.workload]
    label = f"{args.workload}-{args.dtype}-{args.size}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = jax.devices()[:1]
    meta = {
        "workload": args.workload, "dtype": args.dtype, "size": args.size,
        "minutes": args.minutes, "bound": spec["bound"], "unit": spec["unit"],
        "matmul_precision": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", "default"),
        "jax": jax.__version__, "platform": devices[0].platform,
        "device_kind": devices[0].device_kind,
    }
    dtype = jnp.dtype(args.dtype)

    t = time.perf_counter()
    fn, inputs = kernels.BUILDERS[args.workload](args.size, dtype, devices)
    jax.block_until_ready(inputs)
    meta["setup_s"] = time.perf_counter() - t
    t = time.perf_counter()
    compiled = fn.lower(*inputs).compile()
    meta["compile_s"] = time.perf_counter() - t
    t = time.perf_counter()
    result = jax.block_until_ready(compiled(*inputs))
    meta["first_run_s"] = time.perf_counter() - t
    # A silent float32 must not pass as float64
    if result.dtype != dtype:
        raise SystemExit(f"result is {result.dtype.name}, expected {dtype.name}")
    del result
    work = spec["work"](args.size, ITEMSIZE[args.dtype])

    start = time.time()
    end = start + 60 * args.minutes
    meta.update(start_utc=utc(start), start_epoch=start)
    (out / f"{label}.json").write_text(json.dumps(meta, indent=2))

    steps = 0
    with open(out / f"{label}.csv", "w", newline="") as f:
        rows = csv.writer(f)
        rows.writerow(["utc", "second", "steps", "median_ms", "min_ms", "max_ms", "rate"])

        def write(second, times):
            median = statistics.median(times)
            rows.writerow([utc(second), second - int(start), len(times), f"{1e3 * median:.4f}",
                           f"{1e3 * min(times):.4f}", f"{1e3 * max(times):.4f}",
                           f"{work / median:.3f}"])
            f.flush()

        second, times = None, []
        while True:
            t = time.perf_counter()
            jax.block_until_ready(compiled(*inputs))
            elapsed = time.perf_counter() - t
            now = time.time()
            if int(now) != second:
                if times:
                    write(second, times)
                second, times = int(now), []
            times.append(elapsed)
            steps += 1
            if now >= end:
                break
        write(second, times)

    meta.update(end_utc=utc(time.time()), end_epoch=time.time(), steps=steps,
                memory_stats=devices[0].memory_stats(), env=env_info.collect())
    (out / f"{label}.json").write_text(json.dumps(meta, indent=2))
    print(f"{label}: {steps} steps in {args.minutes} min")


def number(text):
    """The number in one of nvidia-smi's values, such as '2610 MHz' or '455.12 W'."""
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group()) if match else None


def read_smi(path):
    """
    nvidia-smi's samples as (epoch, fields), in time order.

    sustained.sh runs nvidia-smi under TZ=UTC, so its local timestamps are UTC. Lines that don't
    parse, such as the header or a value nvidia-smi can't read, are left out or left as None.
    """
    samples = []
    if not path.exists():
        return samples
    with open(path, newline="") as f:
        for row in csv.reader(f, skipinitialspace=True):
            if len(row) != len(SMI_FIELDS):
                continue
            fields = dict(zip(SMI_FIELDS, (v.strip() for v in row)))
            try:
                stamp = fields.pop("timestamp")
                fmt = "%Y/%m/%d %H:%M:%S.%f" if "." in stamp else "%Y/%m/%d %H:%M:%S"
                when = dt.datetime.strptime(stamp, fmt).replace(tzinfo=dt.timezone.utc).timestamp()
            except ValueError:
                continue
            reasons = fields.pop("reasons")
            sample = {k: number(v) for k, v in fields.items()}
            sample["reasons"] = int(reasons, 16) if reasons.startswith("0x") else None
            samples.append((when, sample))
    return samples


def card_window(samples, lo, hi):
    """Medians of the card's fields over [lo, hi), and the share of seconds with each reason."""
    window = [s for t, s in samples if lo <= t < hi]
    stats = {"samples": len(window)}
    for key, _, _ in CARD:
        values = [s[key] for s in window if s[key] is not None]
        stats[key] = statistics.median(values) if values else None
    reasons = [s["reasons"] for s in window if s["reasons"] is not None]
    stats["reasons"] = {name: sum(1 for r in reasons if r & bit) / len(reasons)
                        for bit, name in REASONS.items()} if reasons else {}
    if reasons:
        stats["reasons"]["none"] = sum(1 for r in reasons if r == 0) / len(reasons)
    return stats


def step_window(rows, lo, hi):
    """The median of the per-second median step times, and of the rates, over [lo, hi) seconds."""
    window = [r for r in rows if lo <= int(r["second"]) < hi]
    if not window:
        return {"seconds": 0, "median_ms": None, "rate": None}
    return {"seconds": len(window),
            "steps": sum(int(r["steps"]) for r in window),
            "median_ms": statistics.median(float(r["median_ms"]) for r in window),
            "rate": statistics.median(float(r["rate"]) for r in window)}


def fmt(value, digits=1):
    return "–" if value is None else f"{value:,.{digits}f}"


def mins(value):
    """Minutes to two decimals at most: 5, 30, 0.25."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def ratio(a, b):
    return "–" if a is None or b is None or b == 0 else f"{a / b:.3f}"


def summarise(args):
    out = Path(args.dir)
    samples = read_smi(out / "nvidia-smi.csv")
    phases = []
    if (out / "phases.csv").exists():
        with open(out / "phases.csv", newline="") as f:
            phases = list(csv.DictReader(f))
    summary, lines = [], []
    for meta_path in sorted(out.glob("*.json")):
        if meta_path.name == "summary.json":
            continue
        meta = json.loads(meta_path.read_text())
        if "start_epoch" not in meta:
            continue
        label = meta_path.stem
        with open(out / f"{label}.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        start = meta["start_epoch"]
        end = meta.get("end_epoch", start + int(rows[-1]["second"]) + 1 if rows else start)
        total = end - start
        scale = min(1.0, total / 3600)
        first, last = args.first * 60 * scale, args.last * 60 * scale
        windows = {
            "first": {"minutes": first / 60, "steps": step_window(rows, 0, first),
                      "card": card_window(samples, start, start + first)},
            "last": {"minutes": last / 60, "steps": step_window(rows, total - last, total + 1),
                     "card": card_window(samples, end - last, end)},
        }
        # The idle period that ended last before this workload's process started
        idle = None
        before = [p for p in phases if p["phase"] == "idle" and epoch(p["end"]) <= start]
        if before:
            idle_end = epoch(before[-1]["end"])
            idle_start = max(epoch(before[-1]["start"]), idle_end - 60)
            idle = card_window(samples, idle_start, idle_end)
            temps = [s["temp_c"] for t, s in samples
                     if idle_start <= t < idle_end and s["temp_c"] is not None]
            idle.update(minutes=(idle_end - idle_start) / 60,
                        temp_min=min(temps) if temps else None,
                        temp_max=max(temps) if temps else None)
        summary.append({"label": label, "meta": {k: v for k, v in meta.items() if k != "env"},
                        "windows": windows, "idle_before": idle})

        f_, l_ = windows["first"], windows["last"]
        lines += [
            f"## {meta['workload']}, {meta['dtype']}, size {meta['size']:,}",
            "",
            f"{total / 60:.1f} minutes from {meta['start_utc']}, {meta.get('steps', 0):,} steps, "
            f"on {meta['device_kind']}, JAX {meta['jax']}, matrix-product precision "
            f"{meta['matmul_precision']}. Compile {meta['compile_s']:.2f} s, first run "
            f"{meta['first_run_s']:.3f} s.",
            "",
            f"| | First {mins(f_['minutes'])} min | Last {mins(l_['minutes'])} min | Last ÷ first |",
            "|---|---:|---:|---:|",
            f"| Step, ms (median) | {fmt(f_['steps']['median_ms'], 3)} | "
            f"{fmt(l_['steps']['median_ms'], 3)} | "
            f"{ratio(l_['steps']['median_ms'], f_['steps']['median_ms'])} |",
            f"| Rate, {meta['unit']} | {fmt(f_['steps']['rate'])} | {fmt(l_['steps']['rate'])} | "
            f"{ratio(l_['steps']['rate'], f_['steps']['rate'])} |",
        ]
        for key, name, digits in CARD:
            a, b = f_["card"][key], l_["card"][key]
            lines.append(f"| {name} | {fmt(a, digits)} | {fmt(b, digits)} | "
                         f"{ratio(b, a) if key != 'temp_c' else ''} |")
        seen = [n for n in list(REASONS.values()) + ["none"]
                if f_["card"]["reasons"].get(n) or l_["card"]["reasons"].get(n)]
        for name in seen:
            lines.append(f"| Seconds with {'no reason' if name == 'none' else name} | "
                         f"{f_['card']['reasons'].get(name, 0):.0%} | "
                         f"{l_['card']['reasons'].get(name, 0):.0%} | |")
        lines.append(f"| nvidia-smi samples | {f_['card']['samples']} | {l_['card']['samples']} | |")
        lines.append("")
        if idle and idle["samples"]:
            lines += [f"In the last {mins(idle['minutes'])} min of the idle period before it, the card "
                      f"was at {fmt(idle['temp_min'], 0)}–{fmt(idle['temp_max'], 0)} °C, "
                      f"{fmt(idle['power_w'])} W and {fmt(idle['sm_mhz'], 0)} MHz.", ""]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="hold one kernel on the card and time every step")
    r.add_argument("--workload", required=True, choices=["matmul", "bellman_stored",
                                                         "bellman_recompute", "kalman"])
    r.add_argument("--dtype", required=True, choices=["float32", "float64"])
    r.add_argument("--size", type=int, required=True)
    r.add_argument("--minutes", type=float, default=60)
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarise", help="first minutes against last, for each workload in DIR")
    s.add_argument("dir")
    s.add_argument("--first", type=float, default=5, help="minutes at the start, for an hour's run")
    s.add_argument("--last", type=float, default=30, help="minutes at the end, for an hour's run")
    args = parser.parse_args()
    run(args) if args.command == "run" else summarise(args)


if __name__ == "__main__":
    main()
