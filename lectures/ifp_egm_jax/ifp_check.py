"""
ifp_egm_jax's JAX solve, by precision: does it converge, how fast, and to what?

    python ifp_check.py ifp_egm_jax.md --dtype float32 --sizes 200,1000,5000 --out c.json

Runs the lecture's definition cells (not its Numba cells, which are CPU-only
and the same on every JAX platform), then for each savings-grid size:

- `solve`: the lecture's successive_approx_jax, cold (compile included) and
  warm, with its own tol (1e-5), cap (100,000) and print_skip (1000),
  recording the iterations and whether it converged;
- `K`: the lecture's jitted EGM operator K_egm_jax, warm, applied to the
  converged policy;
- the converged policy, for comparing precisions.

The income grid stays at the lecture's 25 states, so the operator handles
s_size x 25 x 25 interpolations.
"""
import time

T0 = time.perf_counter()

import argparse
import contextlib
import io
import json
import re
import statistics

CELL = re.compile(r"^```\{code-cell\}[^\n]*\n(.*?)^```\s*$", re.S | re.M)
X64_ON = 'jax.config.update("jax_enable_x64", True)'
X64_OFF = 'jax.config.update("jax_enable_x64", False)'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("lecture")
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--sizes", default="200")
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
    ifp, solve, K = ns["ifp"], ns["successive_approx_jax"], ns["K_egm_jax"]

    devices = jax.devices()
    row = {"dtype": args.dtype, "platform": devices[0].platform,
           "device_kind": devices[0].device_kind, "jax": jax.__version__,
           "default_float": jnp.zeros(()).dtype.name, "sizes": []}
    for n in map(int, args.sizes.split(",")):
        model = ifp(s_size=n)
        constants, sizes, arrays = model
        cell = {"s_size": n, "y_size": sizes[1]}
        for phase in ("cold", "warm"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                t = time.perf_counter()
                # The lecture's own print_skip: printing every loop would add a sync per loop
                a_star, σ_star = jax.block_until_ready(solve(model, print_skip=1000))
                cell[f"solve_{phase}_s"] = time.perf_counter() - t
        text = out.getvalue()
        done = re.search(r"Converged in (\d+) iterations", text)
        cell["converged"] = done is not None
        cell["iterations"] = int(done.group(1)) if done else 100_000
        cell["error_every_1000"] = [float(e) for e in re.findall(r"Error at iteration \d+ is ([^ ]+?)\.?$",
                                                                  text, re.M)]
        cell["solve_per_iteration_ms"] = cell["solve_warm_s"] / cell["iterations"] * 1e3
        K(a_star, σ_star, constants, sizes, arrays)
        times = []
        for _ in range(20):
            t = time.perf_counter()
            jax.block_until_ready(K(a_star, σ_star, constants, sizes, arrays))
            times.append(time.perf_counter() - t)
        cell["K_s"] = statistics.median(times)
        cell["policy"] = jnp.ravel(σ_star).tolist()
        cell["assets"] = jnp.ravel(a_star).tolist()
        row["sizes"].append(cell)
        row["process_s"] = time.perf_counter() - T0
        with open(args.out, "w") as f:          # after each size, so a slow one loses nothing
            json.dump(row, f)
        print(f"{args.dtype} {row['platform']} s_size={n}: {cell['iterations']} iterations, "
              f"converged {cell['converged']}; solve cold {cell['solve_cold_s']:.2f} s, "
              f"warm {cell['solve_warm_s']:.2f} s ({cell['solve_per_iteration_ms']:.2f} ms/it), "
              f"K {cell['K_s'] * 1e3:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
