"""
Summarise the problem-type suite's results as markdown tables.

    python problems/summarise.py problems/results/

For each host, problem, sweep and precision, a table gives each backend's
warm and cold times at each size, and the fastest backend by each. Below it
are the cells whose answers don't agree with NumPy's in float64 at the same
size, which took a different number of iterations from NumPy's in the same
precision, or which stopped at their problem's iteration cap. A float32 table
also gives NumPy's float32 iteration counts where they differ from float64's.
"""
import argparse
import ast
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


def reference(latest, host, label, problem, sweep, size, dtype="float64"):
    """
    NumPy's answer at this size and precision: this run's, else this host's
    under another label, else any host's. float32 iteration counts can differ
    a little between hosts, so the nearest run is the fairest comparison.
    """
    same = latest.get((host, label, problem, sweep, size, "numpy", dtype))
    if same and same["status"] == "ok":
        return same
    others = [row for key, row in latest.items()
              if key[2:] == (problem, sweep, size, "numpy", dtype) and row["status"] == "ok"]
    return next((r for r in others if r["host"] == host), others[0] if others else None)


def iteration_cap(problem):
    """
    The problem's MAX_ITER, read from its module's source rather than by
    importing it, so the summary needs no NumPy. None if it has no cap.
    """
    source = (Path(__file__).parent / f"{problem}.py").read_text()
    cap = None
    for node in ast.parse(source).body:
        for target, value in assignments(node):
            if target == "MAX_ITER":
                cap = ast.literal_eval(value)  # the last one, as Python takes it
    if cap is None and "MAX_ITER" in source:
        # Imported, computed or assigned inside a block: don't guess
        raise ValueError(f"{problem}.py names MAX_ITER, but not as a literal at module level")
    return cap


def assignments(node):
    """
    (name, value) pairs of a module-level assignment: `A = 1`, `A: int = 1`,
    `A = B = 1` or `A, B = 1, 2`.
    """
    if isinstance(node, ast.AnnAssign) and node.value is not None:
        targets, value = [node.target], node.value
    elif isinstance(node, ast.Assign):
        targets, value = node.targets, node.value
    else:
        return []
    pairs = []
    for target in targets:
        if isinstance(target, ast.Name):
            pairs.append((target.id, value))
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple):
            pairs += [(t.id, v) for t, v in zip(target.elts, value.elts)
                      if isinstance(t, ast.Name)]
    return pairs


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
        notes, counts, elsewhere = [], [], set()
        # A loop that stops at its cap returns as if it had converged
        cap = iteration_cap(problem)
        for size in sizes:
            rows = {b: cells.get((host, label, problem, sweep, size, b, dtype)) for b in backends}
            print(f"| {size:,} | " + " | ".join(entry(rows[b]) for b in backends)
                  + f" | {fastest(rows, 'warm_median_s')} | {fastest(rows, 'cold_s')} |")
            ref = reference(latest, host, label, problem, sweep, size)
            # Iterations are compared within a precision: float32 can need a
            # different number from float64 by design, as P3's do
            same = ref if dtype == "float64" else reference(latest, host, label, problem,
                                                            sweep, size, dtype)
            elsewhere |= {(r["dtype"], r["host"]) for r in (ref, same) if r and r["host"] != host}
            if same and ref and same is not ref and same["iterations"] != ref["iterations"]:
                counts.append(f"{same['iterations']:,} against {ref['iterations']:,} at {size:,}")
            base = same or ref
            for b, row in rows.items():
                if not row or row["status"] != "ok":
                    continue
                if cap is not None and row["iterations"] >= cap:
                    notes.append(f"- {size:,}, `{b}`: stopped at the cap of {cap:,} iterations")
                if ref and row is not ref:
                    value, passes = disagreement(spec["check"], row, ref)
                    if not passes:
                        measure = ("relative difference" if spec["check"]["kind"] == "close"
                                   else "z")
                        notes.append(f"- {size:,}, `{b}`: answer differs from NumPy's in float64 "
                                     f"({measure} {value:.2g})")
                if base and row is not base and row["iterations"] != base["iterations"]:
                    notes.append(f"- {size:,}, `{b}`: {row['iterations']:,} iterations, against "
                                 f"NumPy's {base['iterations']:,} in {base['dtype']}")
        print()
        if counts:
            notes.insert(0, f"- NumPy's iterations in {dtype}, against its float64 count: "
                            + "; ".join(counts))
        if elsewhere and notes:
            notes.insert(0, "- Where this host has no NumPy row of its own, NumPy's "
                            + "; ".join(f"{d} is {h}'s" for d, h in sorted(elsewhere)))
        if notes:
            print("\n".join(notes) + "\n")


if __name__ == "__main__":
    main()
