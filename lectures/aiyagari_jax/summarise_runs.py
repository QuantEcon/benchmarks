"""Summarise run_lecture.py results: totals per configuration, the slow cells, and fidelity.

    python summarise_runs.py runs/*.json
"""
import json
import statistics
import sys
from collections import defaultdict

runs = [json.load(open(p)) | {"path": p} for p in sys.argv[1:]]
groups = defaultdict(list)
for r in runs:
    groups[(r["platform"], r["dtype"])].append(r)

print("| platform | dtype | runs | lecture s, median | min | max | all ok |")
print("|---|---|---|---|---|---|---|")
for (plat, dtype), rs in sorted(groups.items()):
    t = [r["lecture_s"] for r in rs]
    print(f"| {plat} | {dtype} | {len(rs)} | {statistics.median(t):.1f} | {min(t):.1f} | {max(t):.1f} "
          f"| {all(r['all_ok'] for r in rs)} |")

# Per-cell medians for the cells that take time, side by side
keys = sorted(groups)
n_cells = len(runs[0]["cells"])
print("\n| cell | first line | " + " | ".join(f"{p} {d}" for p, d in keys) + " |")
print("|---|---|" + "---|" * len(keys))
for i in range(n_cells):
    meds = [statistics.median(r["cells"][i]["seconds"] for r in groups[k]) for k in keys]
    if max(meds) < 0.5:
        continue
    first = runs[0]["cells"][i]["first_line"].replace("|", "\\|")[:55]
    print(f"| {i} | `{first}` | " + " | ".join(f"{m:.2f}" for m in meds) + " |")

# Fidelity: equilibrium capital and the policy, float32 against float64, per platform
print()
for plat in sorted({p for p, _ in keys}):
    lo, hi = groups.get((plat, "float32")), groups.get((plat, "float64"))
    if not (lo and hi):
        continue
    a, b = lo[0]["record"], hi[0]["record"]
    k32, k64 = a["K_star"], b["K_star"]
    kv32, kv64 = a["K_vals.tolist()"], b["K_vals.tolist()"]
    pol32, pol64 = a["np.asarray(σ_star).ravel().tolist()"], b["np.asarray(σ_star).ravel().tolist()"]
    worst = max(abs(x - y) / abs(y) for x, y in zip(kv32, kv64))
    differ = sum(x != y for x, y in zip(pol32, pol64))
    print(f"{plat}: K_star float32 {k32:.6f} vs float64 {k64:.6f} (rel diff {abs(k32 - k64) / k64:.1e}); "
          f"K over the β sweep, worst rel diff {worst:.1e}; "
          f"policy at default prices differs at {differ} of {len(pol64)} points; "
          f"grid dtype {a['str(household.a_grid.dtype)']} / {b['str(household.a_grid.dtype)']}")
for plat in sorted({p for p, _ in keys}):
    for d in ("float32", "float64"):
        for r in groups.get((plat, d), [])[:1]:
            line = next((c["output"].strip().splitlines()[-1] for c in r["cells"]
                         if "in " in c["output"] and "iterations" in c["output"]), "")
            print(f"{plat} {d}: damped iteration -> {line}")
