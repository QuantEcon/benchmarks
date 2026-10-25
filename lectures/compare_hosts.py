"""
Set the TITAN RTX, its host's Threadripper and qe-mini1's M6 side by side,
for aiyagari_jax and ifp_egm_jax.

    python lectures/compare_hosts.py lectures/results/qe-wharton1-gpu/2026-10-25 \
        lectures/results/qe-mini1/2026-10-25

The first directory supplies the card and its CPU, the second the Mac's CPU.
Medians over each configuration's runs; warm times unless marked.
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

gpu_host, mac_host = map(Path, sys.argv[1:3])
DEVICES = (("TITAN RTX", gpu_host, "gpu"), ("Threadripper", gpu_host, "cpu"), ("M6", mac_host, "cpu"))


def load(root, sub):
    runs = defaultdict(list)
    for p in sorted((root / sub).glob("*.json")):
        r = json.load(open(p))
        runs[(r["platform"], r["dtype"])].append(r)
    return runs


def med(values):
    values = list(values)
    return statistics.median(values) if values else math.nan


def sized(runs, plat, dtype, size_key, n, fn):
    return med(fn(c) for r in runs[(plat, dtype)] for c in r["sizes"] if c[size_key] == n)


def table(head, rows):
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def f(x, digits=2):
    return "" if math.isnan(x) else f"{x:,.{digits}f}"


print("## aiyagari_jax\n\n**The whole lecture as used, seconds:**\n")
rows = []
for name, root, plat in DEVICES:
    runs = load(root, "aiyagari_jax/as-used")
    t = {d: med(r["lecture_s"] for r in runs[(plat, d)]) for d in ("float64", "float32")}
    rows.append([name, f(t["float64"], 1), f(t["float32"], 1)])
table(["device", "float64", "float32"], rows)

scale = {name: load(root, "aiyagari_jax/scale") for name, root, _ in DEVICES}
sizes = sorted({c["a_size"] for runs in scale.values() for rs in runs.values() for r in rs for c in r["sizes"]})
loop = lambda c: c["hpi_warm_s"] / max(c["hpi_iterations"], 1) * 1e3
print("**One policy-iteration loop in float64, ms** (float64/float32 in brackets):\n")
rows = []
for n in sizes:
    row = [f"{n:,}"]
    for name, _, plat in DEVICES:
        a = sized(scale[name], plat, "float64", "a_size", n, loop)
        b = sized(scale[name], plat, "float32", "a_size", n, loop)
        row.append(f"{f(a)} ({a / b:.1f}×)")
    rows.append(row)
table(["asset grid"] + [d[0] for d in DEVICES], rows)

print("**The lecture's `get_greedy` in float64, ms** (float64/float32 in brackets):\n")
rows = []
for n in sizes:
    row = [f"{n:,}"]
    for name, _, plat in DEVICES:
        a = sized(scale[name], plat, "float64", "a_size", n, lambda c: c["greedy_s"] * 1e3)
        b = sized(scale[name], plat, "float32", "a_size", n, lambda c: c["greedy_s"] * 1e3)
        row.append(f"{f(a)} ({a / b:.1f}×)")
    rows.append(row)
table(["asset grid"] + [d[0] for d in DEVICES], rows)

print("## ifp_egm_jax\n\n**The JAX solve at the lecture's grid, warm seconds:**\n")
rows = []
for name, root, plat in DEVICES:
    runs = load(root, "ifp_egm_jax/check")
    t = {d: sized(runs, plat, d, "s_size", 200, lambda c: c["solve_warm_s"]) for d in ("float64", "float32")}
    rows.append([name, f(t["float64"]), f(t["float32"])])
table(["device", "float64", "float32"], rows)

iscale = {name: load(root, "ifp_egm_jax/scale") for name, root, _ in DEVICES}
isizes = sorted({c["s_size"] for runs in iscale.values() for rs in runs.values() for r in rs for c in r["sizes"]})
print("**The EGM operator in float64, ms** (float64/float32 in brackets):\n")
rows = []
for n in isizes:
    row = [f"{n:,}"]
    for name, _, plat in DEVICES:
        a = sized(iscale[name], plat, "float64", "s_size", n, lambda c: c["K_s"] * 1e3)
        b = sized(iscale[name], plat, "float32", "s_size", n, lambda c: c["K_s"] * 1e3)
        row.append(f"{f(a)} ({a / b:.1f}×)")
    rows.append(row)
table(["savings grid"] + [d[0] for d in DEVICES], rows)

print("**The whole lecture as used, float64, seconds:**\n")
rows = []
for name, root, plat in DEVICES:
    runs = load(root, "ifp_egm_jax/as-used")[(plat, "float64")]
    if not runs:
        continue
    rows.append([name, f(med(r["lecture_s"] for r in runs), 1),
                 f(med(r["cells"][10]["seconds"] for r in runs)),
                 f(med(r["record"]["jax_time_without_compile"] for r in runs)),
                 f(med(r["record"]["numba_time_without_compile"] for r in runs), 1)])
table(["device", "lecture", "JAX solve, with compile", "JAX solve, warm", "Numba solve, warm"], rows)
