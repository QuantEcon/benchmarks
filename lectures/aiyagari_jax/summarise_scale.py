"""Summarise aiyagari_scale.py results: medians over repetitions, float64/float32 ratios, fidelity.

    python summarise_scale.py scale/*.json
"""
import json
import math
import statistics
import sys
from collections import defaultdict

runs = defaultdict(list)                     # (platform, dtype) -> [run, ...]
for path in sys.argv[1:]:
    r = json.load(open(path))
    runs[(r["platform"], r["dtype"])].append(r)


def med(plat, dtype, n, key):
    vals = [c[key] for r in runs[(plat, dtype)] for c in r["sizes"] if c["a_size"] == n]
    return statistics.median(vals) if vals else math.nan


def per_it(plat, dtype, n):
    vals = [c["hpi_warm_s"] / max(c["hpi_iterations"], 1)
            for r in runs[(plat, dtype)] for c in r["sizes"] if c["a_size"] == n]
    return statistics.median(vals)


sizes = sorted({c["a_size"] for rs in runs.values() for r in rs for c in r["sizes"]})
reps = {k: len(v) for k, v in runs.items()}
print("repetitions:", reps)
for plat in ("gpu", "cpu"):
    print(f"\n### {plat}: warm ms, median of runs; f64/f32 in brackets\n")
    print("| grid | cells | get_greedy (lecture) f64 · f32 | greedy over stored rewards f64 · f32 "
          "| get_value f64 · f32 | one HPI loop f64 · f32 |")
    print("|---|---|---|---|---|---|")
    for n in sizes:
        row = [f"{n:,}", f"{2 * n * n / 1e6:g}M"]
        for key in ("greedy_s", "greedy_stored_s", "value_s"):
            a, b = med(plat, "float64", n, key), med(plat, "float32", n, key)
            row.append(f"{a * 1e3:.2f} · {b * 1e3:.2f} (**{a / b:.1f}×**)")
        a, b = per_it(plat, "float64", n), per_it(plat, "float32", n)
        row.append(f"{a * 1e3:.1f} · {b * 1e3:.1f} ({a / b:.1f}×)")
        print("| " + " | ".join(row) + " |")

print("\n### the card against the CPU beside it (float64, warm, CPU time / GPU time)\n")
print("| grid | get_greedy | stored | get_value | one HPI loop |")
print("|---|---|---|---|---|")
for n in sizes:
    cells = [med("cpu", "float64", n, k) / med("gpu", "float64", n, k)
             for k in ("greedy_s", "greedy_stored_s", "value_s")]
    cells.append(per_it("cpu", "float64", n) / per_it("gpu", "float64", n))
    print(f"| {n:,} | " + " | ".join(f"{x:.1f}×" for x in cells) + " |")

print("\n### fidelity and convergence, every run\n")
for plat in ("gpu", "cpu"):
    ref = {c["a_size"]: c for c in runs[(plat, "float64")][0]["sizes"]}
    for dtype in ("float64", "float32"):
        for i, r in enumerate(runs[(plat, dtype)]):
            notes = []
            for c in r["sizes"]:
                n, f = c["a_size"], ref[c["a_size"]]
                agree = sum(x == y for x, y in zip(c["policy"], f["policy"])) / len(f["policy"])
                bad = not math.isfinite(c["v_sum"])
                notes.append(f"{n}: {c['hpi_iterations']} it{' CAPPED' if c['hpi_capped'] else ''}, "
                             f"policy {agree:.1%} as f64 run 1, v_sum {c['v_sum']:.6g}"
                             f"{' NOT FINITE' if bad else ''}, stored==greedy {c['stored_matches']}")
            print(f"- {plat} {dtype} run {i + 1}: " + "; ".join(notes))
