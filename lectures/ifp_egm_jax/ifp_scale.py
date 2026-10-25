"""
Time ifp_egm_jax's EGM operator at larger savings grids, in one precision,
in one fresh process.

    python ifp_scale.py ifp_egm_jax.md --dtype float64 --sizes 200,1000 --out s.json

Runs the lecture's definition cells, then for each savings-grid size applies
the lecture's jitted K_egm_jax ten times from the lecture's initial policy
(consume everything), so the policy has the shape iteration gives it, and
times further applications warm. A full solve at large grids takes ~2,200
iterations, which the CPU can't do in reasonable time. The income grid stays
at the lecture's 25 states.
"""
import argparse
import json
import re
import statistics
import time

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lecture")
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--sizes", default="200,1000,5000,20000,50000")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cells = [m.group(1) for m in CELL.finditer(open(args.lecture).read())]
    last = next(i for i, c in enumerate(cells) if "def successive_approx_jax" in c)
    cells = [c for c in cells[:last + 1]
             if "!nvidia-smi" not in c and "!pip install" not in c]
    if args.dtype == "float32":
        cells = [c.replace(X64_ON, X64_OFF) for c in cells]
    ns = {}
    for c in cells:
        exec(compile(c, "<lecture>", "exec"), ns)

    import jax
    import jax.numpy as jnp
    ifp, K = ns["ifp"], ns["K_egm_jax"]

    devices = jax.devices()
    row = {"dtype": args.dtype, "platform": devices[0].platform,
           "device_kind": devices[0].device_kind, "jax": jax.__version__,
           "default_float": jnp.zeros(()).dtype.name, "sizes": []}
    for n in map(int, args.sizes.split(",")):
        model = ifp(s_size=n)
        constants, sizes, arrays = model
        s_grid, y_size = arrays[0], sizes[1]
        σ = jnp.reshape(jnp.repeat(s_grid, y_size), (n, y_size))   # as successive_approx_jax starts
        a = jnp.copy(σ)
        t = time.perf_counter()
        a, σ = jax.block_until_ready(K(a, σ, constants, sizes, arrays))
        cold = time.perf_counter() - t
        for _ in range(9):
            a, σ = K(a, σ, constants, sizes, arrays)
        jax.block_until_ready(σ)
        times = []
        while len(times) < 5 or (sum(times) < 1.0 and len(times) < 50):
            t = time.perf_counter()
            jax.block_until_ready(K(a, σ, constants, sizes, arrays))
            times.append(time.perf_counter() - t)
        cell = {"s_size": n, "y_size": y_size, "elements": n * y_size * y_size,
                "K_cold_s": cold, "K_s": statistics.median(times), "reps": len(times),
                "sigma_sum": float(jnp.sum(σ)), "finite": bool(jnp.all(jnp.isfinite(σ)))}
        row["sizes"].append(cell)
        with open(args.out, "w") as f:
            json.dump(row, f)
        print(f"{args.dtype} {row['platform']} s_size={n}: K warm {cell['K_s'] * 1e3:.3f} ms, "
              f"cold {cold:.2f} s, finite {cell['finite']}", flush=True)


if __name__ == "__main__":
    main()
