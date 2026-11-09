"""
Summarise the problem-type suite's results as markdown tables.

    python problems/summarise.py problems/results/

For each host, problem, sweep and precision, a table gives each backend's
warm and cold times at each size, and the fastest backend by each. Below it
are the cells whose answers don't agree with NumPy's in float64 at the same
size, or which took a different number of iterations.
"""
import argparse
import json
import re
from math import sqrt
from pathlib import Path

from spec import BACKENDS, PROBLEMS


def load(paths):
    """The latest row for each cell, by host and label."""
    latest = {}
    for root in paths:
        for f in sorted(Path(root).rglob("*.jsonl")):
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl", f.name):
                continue  # a run's own records, such as a clock or bandwidth log
            for line in f.read_text().splitlines():
                row = json.loads(line)
                key = (row["host"], row.get("label", ""), row["problem"], row["sweep"],
                       row["size"], row["backend"], row["dtype"])
                latest[key] = row
    return latest


def reference(latest, host, label, problem, sweep, size):
    """NumPy's float64 answer at this size: this host's, or any host's."""
    same = latest.get((host, label, problem, sweep, size, "numpy", "float64"))
    if same and same["status"] == "ok":
        return same
    for key, row in latest.items():
        if (key[2:] == (problem, sweep, size, "numpy", "float64")
                and row["status"] == "ok"):
            return row
    return None


def disagreement(check, row, ref):
    """How far a cell's answer is from the reference, and whether that passes."""
    a, b = row["sample"], ref["sample"]
    if check["kind"] == "close":
        scale = max(abs(x) for x in b) or 1.0
        err = max(abs(x - y) for x, y in zip(a, b)) / scale
        return err, err <= check["rtol"][row["dtype"]]
    n = row["size"]
    z = 0.0
    for p, q in zip(a, b):
        mean = (p + q) / 2
        se = sqrt(max(mean * (1 - mean), 1 / n) * 2 / n)
        z = max(z, abs(p - q) / se)
    return z, z <= check["z"]


def seconds(t):
    return f"{t:.3g}" if t < 1000 else f"{t:,.0f}"


def entry(row):
    if row is None:
        return ""
    if row["status"] != "ok":
        return row["status"]
    return f"{seconds(row['warm_median_s'])} ({seconds(row['cold_s'])})"


def fastest(rows, field):
    ok = {b: r[field] for b, r in rows.items() if r and r["status"] == "ok"}
    return min(ok, key=ok.get) if ok else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    latest = load(args.paths)

    groups = sorted({(k[0], k[1], k[2], k[3], k[6]) for k in latest},
                    key=lambda g: (g[0], g[1], list(PROBLEMS).index(g[2]), g[3],
                                   ["float64", "float32"].index(g[4])))
    for host, label, problem, sweep, dtype in groups:
        spec = PROBLEMS[problem]
        cells = {k: r for k, r in latest.items() if k[:4] == (host, label, problem, sweep)
                 and k[6] == dtype}
        sizes = sorted({k[4] for k in cells})
        backends = [b for b in BACKENDS if any(k[5] == b for k in cells)]
        title = f"{spec['label']} `{problem}`, {sweep} sweep, {dtype}, on {host}"
        print(f"### {title}" + (f" ({label})" if label else "") + "\n")
        print("Median warm time, with cold time (compile and first run) in brackets, "
              "in seconds.\n")
        print("| Size | " + " | ".join(f"`{b}`" for b in backends)
              + " | Fastest warm | Fastest cold |")
        print("|---:|" + "---:|" * len(backends) + "---|---|")
        notes = []
        for size in sizes:
            rows = {b: cells.get((host, label, problem, sweep, size, b, dtype)) for b in backends}
            print(f"| {size:,} | " + " | ".join(entry(rows[b]) for b in backends)
                  + f" | {fastest(rows, 'warm_median_s')} | {fastest(rows, 'cold_s')} |")
            ref = reference(latest, host, label, problem, sweep, size)
            for b, row in rows.items():
                if not row or row["status"] != "ok" or not ref or row is ref:
                    continue
                value, passes = disagreement(spec["check"], row, ref)
                if not passes:
                    measure = "relative difference" if spec["check"]["kind"] == "close" else "z"
                    notes.append(f"- {size:,}, `{b}`: answer differs from NumPy's in float64 "
                                 f"({measure} {value:.2g})")
                if row["iterations"] != ref["iterations"]:
                    notes.append(f"- {size:,}, `{b}`: {row['iterations']:,} iterations, "
                                 f"against NumPy's {ref['iterations']:,} in float64")
        print()
        if notes:
            print("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
