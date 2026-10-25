"""Summarise the ifp_egm_jax results: the JAX solve, the EGM operator at scale, the lecture as used.

    python summarise_ifp.py <results directory>      # holds check/, scale/ and as-used/
"""
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])


def load(sub):
    runs = defaultdict(list)
    for p in sorted((root / sub).glob("*.json")):
        r = json.load(open(p))
        runs[(r["platform"], r["dtype"])].append(r)
    return runs


def med(runs, plat, dtype, size_key, n, key):
    vals = [c[key] for r in runs[(plat, dtype)] for c in r["sizes"] if c[size_key] == n]
    return statistics.median(vals) if vals else math.nan


check, scale = load("check"), load("scale")
print("### The JAX solve at the lecture's grid (200 x 25), medians of",
      {k: len(v) for k, v in check.items()}, "\n")
print("| device | precision | iterations | converged | solve, cold s | solve, warm s | per iteration ms | K warm ms |")
print("|---|---|---|---|---|---|---|---|")
for plat in ("gpu", "cpu"):
    for d in ("float64", "float32"):
        rs = check[(plat, d)]
        its = sorted({c["iterations"] for r in rs for c in r["sizes"]})
        conv = all(c["converged"] for r in rs for c in r["sizes"])
        print(f"| {plat} | {d} | {', '.join(map(str, its))} | {conv} "
              f"| {med(check, plat, d, 's_size', 200, 'solve_cold_s'):.2f} "
              f"| {med(check, plat, d, 's_size', 200, 'solve_warm_s'):.2f} "
              f"| {med(check, plat, d, 's_size', 200, 'solve_per_iteration_ms'):.2f} "
              f"| {med(check, plat, d, 's_size', 200, 'K_s') * 1e3:.3f} |")
for plat in ("gpu", "cpu"):
    p64 = np.array(check[(plat, "float64")][0]["sizes"][0]["policy"])
    p32 = np.array(check[(plat, "float32")][0]["sizes"][0]["policy"])
    gap = np.abs(p32 - p64)
    print(f"\n{plat}: float32 policy against float64, largest gap {gap.max():.2e}, "
          f"largest relative {np.max(gap / np.maximum(np.abs(p64), 1e-12)):.2e}")
g64 = np.array(check[("gpu", "float64")][0]["sizes"][0]["policy"])
c64 = np.array(check[("cpu", "float64")][0]["sizes"][0]["policy"])
print(f"card against CPU, both float64: largest gap {np.abs(g64 - c64).max():.2e}")

sizes = sorted({c["s_size"] for rs in scale.values() for r in rs for c in r["sizes"]})
print("\n### The EGM operator K at larger savings grids, warm ms (median); f64/f32 in brackets\n")
print("| savings grid | elements | card f64 · f32 | CPU f64 · f32 | CPU / card, f64 |")
print("|---|---|---|---|---|")
for n in sizes:
    g = [med(scale, "gpu", d, "s_size", n, "K_s") for d in ("float64", "float32")]
    c = [med(scale, "cpu", d, "s_size", n, "K_s") for d in ("float64", "float32")]
    print(f"| {n:,} | {n * 625 / 1e6:g}M | {g[0] * 1e3:.2f} · {g[1] * 1e3:.2f} (**{g[0] / g[1]:.1f}×**) "
          f"| {c[0] * 1e3:.2f} · {c[1] * 1e3:.2f} ({c[0] / c[1]:.1f}×) | {c[0] / g[0]:.1f}× |")
bad = [(k, c["s_size"]) for k, rs in scale.items() for r in rs for c in r["sizes"] if not c["finite"]]
print("\nnon-finite policies:", bad or "none")

used = load("as-used") if (root / "as-used").exists() else {}
if used:
    print("\n### The lecture as used, float64 (median of runs)\n")
    print("| device | runs | lecture s | JAX solve, with compile | JAX solve, warm | Numba solve, warm |")
    print("|---|---|---|---|---|---|")
    for (plat, d), rs in sorted(used.items()):
        def cell(i, text):
            vals = []
            for r in rs:
                c = r["cells"][i]
                assert text in c["output"] or text == "", (i, c["first_line"])
                vals.append(c["seconds"])
            return statistics.median(vals)
        print(f"| {plat} | {len(rs)} | {statistics.median(r['lecture_s'] for r in rs):.1f} "
              f"| {cell(10, 'Converged'):.2f} "
              f"| {statistics.median(r['record']['jax_time_without_compile'] for r in rs):.2f} "
              f"| {statistics.median(r['record']['numba_time_without_compile'] for r in rs):.1f} |")
