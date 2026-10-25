"""How far float32 policies sit from float64, in grid steps and in assets.

    python policy_gap.py scale/
"""
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
for plat, prefix in (("gpu", "cuda"), ("cpu", "cpu")):
    ref = {c["a_size"]: c for c in json.load(open(root / f"{prefix}-float64-1.json"))["sizes"]}
    low = {c["a_size"]: c for c in json.load(open(root / f"{prefix}-float32-1.json"))["sizes"]}
    for n, c64 in ref.items():
        p64, p32 = np.array(c64["policy"]), np.array(low[n]["policy"])
        gap = np.abs(p32 - p64)
        step = 20 / (n - 1)                       # the lecture's grid runs from ~0 to 20
        finite = np.isfinite(low[n]["v_sum"])
        print(f"{plat} {n:6d}: differ at {np.mean(gap > 0):6.1%}; by one step {np.mean(gap == 1):6.1%}; "
              f"by more {np.mean(gap > 1):6.1%}; largest {gap.max():5d} steps = {gap.max() * step:.3f} "
              f"of assets; values finite: {finite}")
