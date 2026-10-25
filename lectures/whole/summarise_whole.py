"""
Summarise the whole-lecture runs (#4): each lecture's time by device, and where it goes.

    python lectures/whole/summarise_whole.py lectures/results/*/2026-10-25/whole \
        --lectures <directory of the lecture .md files>

Each argument is one host's directory of `<lecture>/<platform>-as-written-<run>.json`
files. Medians over runs. A cell the lecture tags `raises-exception` counts as an
expected error: the tags are read from the lecture's source, so runs made before
run_lecture.py recorded tags are judged the same way.
"""
import argparse
import importlib.util
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_lecture", HERE.parent / "run_lecture.py")
run_lecture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_lecture)


def device_name(run):
    env = run.get("env") or {}
    if run["platform"] == "gpu":
        return (run.get("device_kind") or "GPU").replace("NVIDIA ", "")
    model = env.get("cpu_model") or "CPU"
    for long, short in (("AMD Ryzen Threadripper 2990WX 32-Core Processor", "Threadripper 2990WX"),
                        ("Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz", "Xeon Platinum 8259CL")):
        model = model.replace(long, short)
    return model


def expected(lectures_dir, lecture):
    """Indexes of the cells run_lecture.py runs that the lecture tags raises-exception."""
    path = Path(lectures_dir) / f"{lecture}.md"
    if not path.exists():
        return set()
    cells = run_lecture.code_cells(path.read_text())
    cells = [(s, t) for s, t in cells if not (s.strip() and all(
        line.strip().startswith("!pip install") for line in s.strip().splitlines()))]
    return {i for i, (_, tags) in enumerate(cells) if "raises-exception" in tags}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hosts", nargs="+", help="one directory per host")
    parser.add_argument("--lectures", help="directory of the lectures' .md files")
    parser.add_argument("--top", type=int, default=2, help="slowest cells to list per device")
    args = parser.parse_args()

    runs = defaultdict(lambda: defaultdict(list))     # lecture -> device -> [run, ...]
    for host in map(Path, args.hosts):
        for f in sorted(host.glob("*/*-as-written-*.json")):
            run = json.loads(f.read_text())
            runs[f.parent.name][device_name(run)].append(run)
    devices = sorted({d for by in runs.values() for d in by}, key=lambda d: (d.startswith("Apple"), d))

    print("| lecture | precision | " + " | ".join(f"{d}, s" for d in devices)
          + " | fastest | slowest cells, by device |")
    print("|---|---|" + "---|" * len(devices) + "---|---|")
    problems, no_code = [], []
    totals, counts = defaultdict(float), defaultdict(int)
    for lecture in sorted(runs):
        by = runs[lecture]
        if not any(r["cells"] for d in by for r in by[d]):
            no_code.append(lecture)
            continue
        ok_tags = expected(args.lectures, lecture) if args.lectures else set()
        precision = sorted({r["default_float"] for d in by for r in by[d]})
        # A device with an unexpected failure in any run gets no time: the run stopped short
        failed = set()
        for d in by:
            bad = [[c["index"] for c in r["cells"] if not c["ok"] and c["index"] not in ok_tags] for r in by[d]]
            if any(bad):
                failed.add(d)
                problems.append(f"{lecture} on {d}: cells {next(b for b in bad if b)} failed, "
                                f"in {sum(map(bool, bad))} of {len(bad)} runs")
        med = {d: statistics.median(r["lecture_s"] for r in by[d]) for d in by if d not in failed}
        for d in med:
            totals[d] += med[d]
            counts[d] += 1
        fastest = min(med, key=med.get)
        cells = []
        for d in devices:
            if d not in by:
                continue
            per_cell = defaultdict(list)
            for r in by[d]:
                for c in r["cells"]:
                    per_cell[(c["index"], c["first_line"])].append(c["seconds"])
            top = sorted(per_cell.items(), key=lambda kv: -statistics.median(kv[1]))[:args.top]
            cells.append(f"{d.split()[0]}: " + ", ".join(
                f"#{i} {statistics.median(s):.1f} s" for (i, _), s in top))
        print(f"| {lecture} | {'/'.join(precision)} | "
              + " | ".join(f"{med[d]:.1f}" if d in med else "failed" if d in failed else "" for d in devices)
              + f" | {fastest} | {'; '.join(cells)} |")
    lectures = max(counts.values())
    print("| **all lectures** | | " + " | ".join(
        f"**{totals[d]:.0f}**" + (f", {counts[d]} lectures" if counts[d] < lectures else "") for d in devices) + " | | |")
    print()
    if no_code:
        print("No code cells:", ", ".join(no_code))
    print("Unexpected cell failures:", "; ".join(problems) if problems else "none")


if __name__ == "__main__":
    main()
