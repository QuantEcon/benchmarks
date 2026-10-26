"""
Summarise #4's float32 pass: the float64 lectures run in float64 and in float32, back to back.

    python lectures/whole/summarise_precision.py lectures/results/*/2026-10-25/whole \
        --lectures <directory of the lecture .md files>

Each argument is one host's directory of `<lecture>/<platform>-float64-<run>.json` and
`<platform>-float32-<run>.json` files. For each lecture and device it prints the median
time in each precision and float64's time over float32's.

It also compares what the cells print, as a first check on float32's answers. A
printed number counts where every float64 run prints the same value in the same place,
which leaves out timings and anything else that varies from run to run. A cell is
flagged if a float32 run prints a different count of numbers, or moves a counted one by
more than --rtol. That includes convergence diagnostics, so a flag marks where to look,
not an error. Only each cell's last 3,000 characters of output are kept.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from summarise_whole import device_name, expected

# Not the digits of a name such as float64, which float32 renames
NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|\bnan\b|\binf\b", re.IGNORECASE)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def numbers(text):
    return [float(m) for m in NUMBER.findall(ANSI.sub("", text or ""))]


def failure(run, ok_tags):
    return next((c["index"] for c in run["cells"] if not c["ok"] and c["index"] not in ok_tags), None)


def compare(f64_runs, f32_runs, rtol):
    """Cells whose printed numbers move in float32: [(index, how)], and the number of cells compared.

    A number counts where every float64 run prints the same value at the same place, so a
    timing printed beside a result drops out and the result stays.
    """
    flagged, compared = [], 0
    for i in range(len(f64_runs[0]["cells"])):
        ref = [numbers(r["cells"][i].get("output")) for r in f64_runs]
        if not ref[0] or any(len(x) != len(ref[0]) for x in ref):
            continue
        stable = [k for k in range(len(ref[0])) if all(x[k] == ref[0][k] for x in ref)]
        if not stable:
            continue
        compared += 1
        for r in f32_runs:
            got = numbers(r["cells"][i].get("output"))
            if len(got) != len(ref[0]):
                flagged.append((i, "prints a different count of numbers"))
                break
            worst = max(abs(got[k] - ref[0][k]) / max(abs(ref[0][k]), 1e-300) if got[k] != ref[0][k] else 0.0
                        for k in stable)
            if worst != worst or worst > rtol:       # NaN counts as moved
                flagged.append((i, f"moves by up to {worst:.1e}"))
                break
    return flagged, compared


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hosts", nargs="+", help="one directory per host")
    parser.add_argument("--lectures", help="directory of the lectures' .md files")
    parser.add_argument("--rtol", type=float, default=1e-3, help="relative change that flags a cell")
    args = parser.parse_args()

    runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # lecture -> device -> dtype -> runs
    for host in map(Path, args.hosts):
        for dtype in ("float64", "float32"):
            for f in sorted(host.glob(f"*/*-{dtype}-*.json")):
                run = json.loads(f.read_text())
                runs[f.parent.name][device_name(run)][dtype].append(run)
    devices = sorted({d for by in runs.values() for d in by}, key=lambda d: (d.startswith("Apple"), d))

    print("| lecture | device | float64, s | float32, s | float64 / float32 | float32's answers |")
    print("|---|---|---|---|---|---|")
    totals = defaultdict(lambda: defaultdict(float))
    for lecture in sorted(runs):
        ok_tags = expected(args.lectures, lecture) if args.lectures else set()
        for d in devices:
            by = runs[lecture].get(d)
            if not by or not by["float64"] or not by["float32"]:
                continue
            cell = {}
            for dtype in ("float64", "float32"):
                bad = [failure(r, ok_tags) for r in by[dtype]]
                if any(b is not None for b in bad):
                    first = next(b for b in bad if b is not None)
                    cell[dtype] = f"failed, cell #{first} in {sum(b is not None for b in bad)} of {len(bad)} runs"
                else:
                    cell[dtype] = statistics.median(r["lecture_s"] for r in by[dtype])
            both = all(isinstance(v, float) for v in cell.values())
            if both:
                for dtype, v in cell.items():
                    totals[d][dtype] += v
            ratio = f"{cell['float64'] / cell['float32']:.2f}×" if both else ""
            flagged, compared = compare(by["float64"], by["float32"], args.rtol) if both else ([], 0)
            answers = (f"{len(flagged)} of {compared} cells change: " + "; ".join(f"#{i} {how}" for i, how in flagged)
                       if flagged else f"none of {compared} cells change" if both else "")
            fmt = lambda v: f"{v:.1f}" if isinstance(v, float) else v
            print(f"| {lecture} | {d} | {fmt(cell['float64'])} | {fmt(cell['float32'])} | {ratio} | {answers} |")
    print()
    for d in devices:
        t = totals[d]
        if t["float64"] and t["float32"]:
            print(f"{d}: float64 {t['float64']:.0f} s, float32 {t['float32']:.0f} s, over the lectures that ran in both")


if __name__ == "__main__":
    main()
