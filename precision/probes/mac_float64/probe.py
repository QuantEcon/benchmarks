"""
Float64 on Apple Silicon's CPU: Accelerate against XLA, for QuantEcon/project-benchmarks-jax#48.

Each subcommand runs one variant in its own process, as run.sh calls it, and prints one JSON line per
cell. A cell's first call (compile included) is timed apart from its warm calls, and its time is the
median of the warm calls, as meta#335 asks.

    python probe.py features
    python probe.py matmul numpy|jax-ynn|jax-xla-dot 8,64,512,1024,4096
    python probe.py cliff default|no-reduce-fusion|add|rowmax 16000,18000
    python probe.py step jax|mlx bellman|pv 2000,8000
    python probe.py summarise <out-dir>

The three matmul variants are NumPy's `a @ b` (Accelerate), `jnp.dot` at JAX's defaults (XLA hands
it to YNNPACK) and `jnp.dot` with YNNPACK's dot fusion off, which leaves it to XLA's own dot. The
cliff is QuantEcon/project-benchmarks-jax#7's reproducer. The step compares MLX on its CPU device with JAX's, in float64.
"""
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

import numpy as np

# The XLA flags each variant runs under; run.sh may add an --xla_dump_to through PROBE_DUMP
FLAGS = {
    "jax-ynn": "",
    "jax-xla-dot": "--xla_cpu_experimental_ynn_fusion_type=-individual_dot",
    "default": "",
    "no-reduce-fusion": "--xla_cpu_experimental_ynn_fusion_type=-reduce",
    "add": "",
    "rowmax": "",
}
BETA = 0.96
STEPS = 10_000  # products in a small-size recurrence, as QuantEcon/project-benchmarks-jax#4 timed them


def jax_cpu(variant):
    """Import JAX on the CPU in float64, under the variant's XLA flags."""
    flags = FLAGS[variant]
    if os.environ.get("PROBE_DUMP"):
        flags += f" --xla_dump_to={os.environ['PROBE_DUMP']} --xla_dump_hlo_as_text"
    os.environ["XLA_FLAGS"] = flags.strip()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    import jax.numpy as jnp
    return jax, jnp


def timed(call, budget=10.0, least=3, most=20):
    """The first call's time, then warm calls' times: at least `least`, then until `budget` is spent."""
    t = time.perf_counter()
    call()
    first = time.perf_counter() - t
    warm, start = [], time.perf_counter()
    while len(warm) < most and (len(warm) < least or time.perf_counter() - start < budget):
        t = time.perf_counter()
        call()
        warm.append(time.perf_counter() - t)
    return first, warm


def rel_diff(out, ref):
    """Largest absolute difference, relative to the reference's largest magnitude."""
    out, ref = np.asarray(out, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


def ynn(f, *args):
    """Whether XLA's compiled module hands any of the work to YNNPACK."""
    return "__ynn_fusion" in f.lower(*args).compile().as_text()


def versions(*names):
    found = {"python": platform.python_version()}
    for name in names:
        try:
            module = __import__(name)
            found[name] = getattr(module, "__version__", "?")
        except ImportError:
            pass
    return found


def emit(**row):
    first, warm = row.pop("first"), row.pop("warm")
    row.update(first_s=first, warm_median_s=statistics.median(warm), warm_min_s=min(warm),
               reps=len(warm), dtype="float64",
               host=os.environ.get("PROBE_HOST", socket.gethostname().split(".")[0]),
               timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               xla_flags=os.environ.get("XLA_FLAGS", ""))
    print(json.dumps(row), flush=True)


def features():
    """Step 1: the chip, its matrix-unit features, macOS, NumPy's BLAS and the frameworks."""
    def run(*cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (r.stdout or r.stderr).strip()
    print("chip:", run("sysctl", "-n", "machdep.cpu.brand_string"))
    print("macOS:", run("sw_vers", "-productVersion"), run("sw_vers", "-buildVersion"))
    print("memory bytes:", run("sysctl", "-n", "hw.memsize"))
    for level in range(3):
        name = run("sysctl", "-n", f"hw.perflevel{level}.name")
        if name and "unknown" not in name:
            print(f"perflevel{level}: {name}, {run('sysctl', '-n', f'hw.perflevel{level}.physicalcpu')} cores")
    print("sysctl:", run("sysctl", "hw.optional.arm.FEAT_SME", "hw.optional.arm.FEAT_SME2",
                         "hw.optional.arm.FEAT_SME_F64F64"))
    every = [line for line in run("sysctl", "hw.optional.arm").splitlines() if "SME" in line]
    print("every SME feature:", "; ".join(every) or "none listed")
    print("versions:", versions("numpy", "jax", "jaxlib"))
    print("NumPy's configuration:")
    import warnings
    warnings.filterwarnings("ignore", message="Install `pyyaml`")
    np.show_config()


def matmul(variant, sizes):
    """Step 2: float64 products, three ways; small sizes also as a recurrence, per product."""
    rng = np.random.default_rng(0)
    if variant != "numpy":
        jax, jnp = jax_cpu(variant)
        f = jax.jit(jnp.dot)
    for n in sizes:
        a, b = rng.uniform(-1, 1, (n, n)), rng.uniform(-1, 1, (n, n))
        ref = a @ b
        if variant == "numpy":
            first, warm = timed(lambda: np.dot(a, b))
            out, lowering = np.dot(a, b), "accelerate"
        else:
            A, B = jnp.asarray(a), jnp.asarray(b)
            first, warm = timed(lambda: f(A, B).block_until_ready())
            out, lowering = f(A, B), "ynn" if ynn(f, A, B) else "xla"
        emit(step="matmul", variant=variant, size=n, first=first, warm=warm,
             gflops=2 * n ** 3 / statistics.median(warm) / 1e9, max_rel_diff=rel_diff(out, ref),
             lowering=lowering, versions=versions("numpy", "jax", "jaxlib"))
        if n > 64:
            continue
        # One product a step, STEPS steps, as QuantEcon/project-benchmarks-jax#4 timed it: per
        # product, the kind of cost a Kalman
        # filter pays. NumPy's includes Python's dispatch of three array operations
        M = np.eye(n) * 0.5 + 0.01

        def recurrence_np():
            X = np.eye(n)
            for _ in range(STEPS):
                X = M @ X * 0.9 + M
            return X
        ref = recurrence_np()
        extra = {}
        if variant == "numpy":
            first, warm = timed(recurrence_np)
            out = recurrence_np()
            X, buf = np.eye(n), np.empty((n, n))

            def bare():
                for _ in range(STEPS):
                    np.dot(M, X, out=buf)
            _, bare_warm = timed(bare)
            extra["bare_dot_us"] = statistics.median(bare_warm) / STEPS * 1e6
        else:
            Mj = jnp.asarray(M)
            run = jax.jit(lambda X0: jax.lax.scan(lambda X, _: (Mj @ X * 0.9 + Mj, None), X0, None,
                                                  length=STEPS)[0])
            X0 = jnp.eye(n)
            first, warm = timed(lambda: run(X0).block_until_ready())
            out = run(X0)
            extra["lowering"] = "ynn" if ynn(run, X0) else "xla"
        emit(step="recurrence", variant=variant, size=n, first=first, warm=warm,
             per_product_us=statistics.median(warm) / STEPS * 1e6, max_rel_diff=rel_diff(out, ref),
             versions=versions("numpy", "jax", "jaxlib"), **extra)


def cliff(variant, sizes):
    """Step 3: QuantEcon/project-benchmarks-jax#7's fused row reduction, and its two halves."""
    jax, jnp = jax_cpu(variant)
    kernels = {
        "default": lambda R, v: jnp.max(R + BETA * v[None, :], axis=1),
        "no-reduce-fusion": lambda R, v: jnp.max(R + BETA * v[None, :], axis=1),
        "add": lambda R, v: R + BETA * v[None, :],
        "rowmax": lambda R, v: jnp.max(R, axis=1),
    }
    f = jax.jit(kernels[variant])
    for m in sizes:
        R, v = jnp.full((m, m), -1.0, jnp.float64), jnp.zeros(m, jnp.float64)
        first, warm = timed(lambda: f(R, v).block_until_ready(), least=3, most=5)
        emit(step="cliff", variant=variant, size=m, first=first, warm=warm, gib=R.nbytes / 2 ** 30,
             gbps=R.nbytes / statistics.median(warm) / 1e9, lowering="ynn" if ynn(f, R, v) else "xla",
             versions=versions("numpy", "jax", "jaxlib"))
        del R, v


def step_inputs(kernel, n):
    """The same float64 inputs for both frameworks, with NumPy's answer."""
    rng = np.random.default_rng(1)
    if kernel == "bellman":
        # kernels.py's stored Bellman step: log utility on one wealth grid, -inf where infeasible
        w = np.linspace(0.1, 10.0, n)
        c = 1.03 * w[:, None] + 1.0 - w[None, :]
        R = np.where(c > 0, np.log(np.where(c > 0, c, 1.0)), -np.inf)
        v = np.log(1.0 + w)
        return (R, v), np.max(R + BETA * v[None, :], axis=1)
    P = rng.uniform(0, 1, (n, n))
    P /= P.sum(axis=1, keepdims=True)
    V = rng.uniform(-1, 1, (n, n))
    return (P, V), P @ V


def step(framework, kernel, sizes):
    """Step 4: one value-function kernel in MLX on its CPU device, or in JAX, in float64."""
    if framework == "jax":
        jax, jnp = jax_cpu("default")
        fn = (lambda R, v: jnp.max(R + BETA * v[None, :], axis=1)) if kernel == "bellman" else jnp.dot
        f = jax.jit(fn)
    else:
        import mlx.core as mx
        mx.set_default_device(mx.cpu)
        fn = (lambda R, v: mx.max(R + BETA * v[None, :], axis=1)) if kernel == "bellman" else mx.matmul
        f = mx.compile(fn)
    for n in sizes:
        inputs, ref = step_inputs(kernel, n)
        try:
            if framework == "jax":
                args = [jnp.asarray(x) for x in inputs]
                call = lambda: f(*args).block_until_ready()
                result = lambda: f(*args)
            else:
                args = [mx.array(x, dtype=mx.float64) for x in inputs]
                call = lambda: mx.eval(f(*args))
                result = lambda: np.array(f(*args))
            first, warm = timed(call)
            out = result()
        except Exception as e:  # an operation MLX lacks in float64, say
            print(json.dumps(dict(step=f"step-{kernel}", variant=framework, size=n, status="error",
                                  error=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")), flush=True)
            continue
        work = dict(gbps=inputs[0].nbytes / statistics.median(warm) / 1e9) if kernel == "bellman" \
            else dict(gflops=2 * n ** 3 / statistics.median(warm) / 1e9)
        diff = rel_diff(out, ref)
        emit(step=f"step-{kernel}", variant=framework, size=n, first=first, warm=warm, status="ok",
             max_rel_diff=diff, agrees=diff <= 1e-12,
             versions=versions("numpy", "jax", "jaxlib", "mlx"), **work)


def summarise(folder):
    """Markdown tables of a run's rows."""
    def rows(name):
        path = os.path.join(folder, name)
        return [json.loads(line) for line in open(path)] if os.path.exists(path) else []
    out = []
    mm = rows("matmul.jsonl")
    if mm:
        out += ["## Matrix products, float64", "",
                "| n | NumPy (Accelerate) | JAX, YNNPACK | JAX, XLA's own dot | Largest relative difference from NumPy |",
                "|---:|---:|---:|---:|---|"]
        for n in sorted({r["size"] for r in mm if r["step"] == "matmul"}):
            cell = {r["variant"]: r for r in mm if r["step"] == "matmul" and r["size"] == n}
            g = [f"{cell[v]['gflops']:.1f}" if v in cell else "" for v in ("numpy", "jax-ynn", "jax-xla-dot")]
            d = ", ".join(f"{v} {cell[v]['max_rel_diff']:.1e}" for v in ("jax-ynn", "jax-xla-dot") if v in cell)
            out.append(f"| {n:,} | {g[0]} GFLOPS | {g[1]} GFLOPS | {g[2]} GFLOPS | {d} |")
        rec = [r for r in mm if r["step"] == "recurrence"]
        if rec:
            out += ["", "Small products in a recurrence, as QuantEcon/project-benchmarks-jax#4 timed them, µs per product (NumPy's includes Python's dispatch; its bare `np.dot` call in brackets):", "",
                    "| n | NumPy | JAX, YNNPACK | JAX, XLA's own dot |", "|---:|---:|---:|---:|"]
            for n in sorted({r["size"] for r in rec}):
                cell = {r["variant"]: r for r in rec if r["size"] == n}
                np_ = cell.get("numpy")
                np_s = f"{np_['per_product_us']:.3f} ({np_['bare_dot_us']:.3f})" if np_ else ""
                js = [f"{cell[v]['per_product_us']:.3f}" if v in cell else "" for v in ("jax-ynn", "jax-xla-dot")]
                out.append(f"| {n} | {np_s} | {js[0]} | {js[1]} |")
    cl = rows("cliff.jsonl")
    if cl:
        out += ["", "## The cliff (QuantEcon/project-benchmarks-jax#7), float64, GB/s of the m x m array (warm median)", "",
                "| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |",
                "|---:|---:|---:|---:|---:|---:|"]
        for m in sorted({r["size"] for r in cl}):
            cell = {r["variant"]: r for r in cl if r["size"] == m}
            g = [f"{cell[v]['gbps']:.1f} ({cell[v]['lowering']})" if v in cell else ""
                 for v in ("default", "no-reduce-fusion", "add", "rowmax")]
            gib = next(iter(cell.values()))["gib"]
            out.append(f"| {m:,} | {gib:.2f} | " + " | ".join(g) + " |")
    st = rows("step.jsonl")
    if st:
        out += ["", "## One value-function kernel, MLX on its CPU device against JAX, float64 (warm median)", "",
                "| Kernel | n | JAX | MLX | MLX's difference from NumPy |", "|---|---:|---:|---:|---|"]
        for k in ("bellman", "pv"):
            for n in sorted({r["size"] for r in st if r["step"] == f"step-{k}"}):
                cell = {r["variant"]: r for r in st if r["step"] == f"step-{k}" and r["size"] == n}
                t = [f"{cell[v]['warm_median_s'] * 1e3:.2f} ms" if cell.get(v, {}).get("status") == "ok"
                     else cell.get(v, {}).get("error", "") for v in ("jax", "mlx")]
                mlx_d = f"{cell['mlx']['max_rel_diff']:.1e}" if cell.get("mlx", {}).get("status") == "ok" else ""
                out.append(f"| {'stored Bellman step' if k == 'bellman' else 'P @ V'} | {n:,} | {t[0]} | {t[1]} | {mlx_d} |")
    print("\n".join(out))


if __name__ == "__main__":
    command, rest = sys.argv[1], sys.argv[2:]
    sizes = lambda s: [int(x) for x in s.split(",")]
    if command == "features":
        features()
    elif command == "matmul":
        matmul(rest[0], sizes(rest[1]))
    elif command == "cliff":
        cliff(rest[0], sizes(rest[1]))
    elif command == "step":
        step(rest[0], rest[1], sizes(rest[2]))
    elif command == "summarise":
        summarise(rest[0])
    else:
        sys.exit(f"unknown command {command}")
