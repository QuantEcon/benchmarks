"""
Summarise precision benchmark results as markdown tables.

    python precision/summarise.py results/

For each host and label: warm times in both precisions with the float64
penalty, and then the size at which each GPU first beats the same
machine's CPU. Where a cell was run more than once, the latest run counts.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from spec import WORKLOADS


def load(paths):
    rows = []
    for p in map(Path, paths):
        for f in (sorted(p.rglob("*.jsonl")) if p.is_dir() else [p]):
            rows += [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    latest = {}
    for row in sorted(rows, key=lambda r: r["timestamp"]):
        key = (row["host"], row["label"], row["workload"], row["size"], row["device"], row["dtype"])
        latest[key] = row
    return list(latest.values())


def ms(row):
    if row is None:
        return ""
    if row["status"] != "ok":
        return row["status"]
    return f"{row['warm_median_s'] * 1e3:.3g}"


def rate(row):
    if row is None or row["status"] != "ok":
        return ""
    return f"{row['work'] / row['warm_median_s']:.3g}"


def cold(row):
    if row is None or row["status"] != "ok":
        return ""
    return f"{row['compile_s'] + row['first_run_s']:.3g}"


def fidelity(lo, hi):
    """Largest relative difference between the float32 and float64 samples."""
    if not (lo and hi and lo["status"] == hi["status"] == "ok"):
        return ""
    worst = max(abs(a - b) / max(abs(b), 1e-300) for a, b in zip(lo["sample"], hi["sample"]))
    return f"{worst:.1e}"


def penalty(lo, hi):
    if not (lo and hi and lo["status"] == hi["status"] == "ok"):
        return ""
    return f"{hi['warm_median_s'] / lo['warm_median_s']:.1f}x"


def device_order(device):
    kind, _, count = device.partition(":")
    return (kind != "cpu", int(count or 1))


def summarise(rows):
    groups = defaultdict(dict)
    for r in rows:
        groups[(r["host"], r["label"])][(r["workload"], r["size"], r["device"], r["dtype"])] = r

    for (host, label), cells in sorted(groups.items()):
        title = f"{host}" + (f" ({label})" if label else "")
        env = next(iter(cells.values()))["env"]
        print(f"## {title}\n")
        print(f"{env.get('cpu_model')}, {env.get('cpu_count')} threads, "
              f"{env['memory_bytes'] / 2**30:.0f} GiB; GPU: {env.get('gpu') or 'none'}; "
              f"JAX {env['packages'].get('jax', '?')}\n")
        print("| workload | size | device | f32 ms | f64 ms | f64/f32 | f32 rate | f64 rate "
              "| unit | f64 cold s | f32 vs f64 |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        keys = sorted({(w, s, d) for w, s, d, _ in cells},
                      key=lambda k: (list(WORKLOADS).index(k[0]), device_order(k[2]), k[1]))
        for w, s, d in keys:
            lo, hi = cells.get((w, s, d, "float32")), cells.get((w, s, d, "float64"))
            unit = WORKLOADS[w]["unit"]
            print(f"| {w} | {s} | {d} | {ms(lo)} | {ms(hi)} | {penalty(lo, hi)} "
                  f"| {rate(lo)} | {rate(hi)} | {unit} | {cold(hi)} | {fidelity(lo, hi)} |")
        print()

        gpu_devices = sorted({d for _, _, d, _ in cells if d.startswith("gpu")}, key=device_order)
        if not gpu_devices:
            continue
        print("**Where the GPU first beats this machine's CPU** (warm, then cold)\n")
        print("| workload | precision | device | warm | cold |")
        print("|---|---|---|---|---|")
        for w in WORKLOADS:
            for dtype in ("float32", "float64"):
                for d in gpu_devices:
                    sizes = sorted(s for ww, s, dd, t in cells if (ww, dd, t) == (w, d, dtype))
                    if not sizes:
                        continue
                    found = {"warm": None, "cold": None}
                    compared = []
                    for s in sizes:
                        g, c = cells[(w, s, d, dtype)], cells.get((w, s, "cpu", dtype))
                        if not (c and g["status"] == c["status"] == "ok"):
                            continue
                        compared.append(s)
                        if found["warm"] is None and g["warm_median_s"] < c["warm_median_s"]:
                            found["warm"] = s
                        g_cold = g["compile_s"] + g["first_run_s"]
                        if found["cold"] is None and g_cold < c["compile_s"] + c["first_run_s"]:
                            found["cold"] = s
                    never = f"not up to {compared[-1]}" if compared else "no comparison"
                    print(f"| {w} | {dtype} | {d} | {found['warm'] or never} "
                          f"| {found['cold'] or never} |")
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="result files or directories")
    summarise(load(parser.parse_args().paths))


if __name__ == "__main__":
    main()
