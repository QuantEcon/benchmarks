"""float64 / matmul-precision probe for a TPU runtime (Colab, Kaggle or a TPU VM).

Run in a FRESH process:  python tpu_f64_probe.py  (or paste into one notebook cell).
Every check prints one line: OK with the error against CPU float64, or FAIL with
the first line of the exception.  "bits" = -log2(max relative error): IEEE
float64 gives ~52, float32 ~23-24, a pair-of-float32 (ef57) format ~45-49.
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
import time, math, importlib.metadata as md
import numpy as np
import jax, jax.numpy as jnp
from jax import lax
jax.config.update("jax_enable_x64", True)

ACC = jax.devices()[0]            # the TPU when one is attached
CPU = jax.devices("cpu")[0]       # the TPU VM's host CPU: the reference
print("jax", jax.__version__, "| jaxlib", md.version("jaxlib"), "| device:",
      ACC.platform, ACC.device_kind, "| n devices:", jax.device_count())
for pkg in ("libtpu", "libtpu-nightly"):
    try: print(pkg, md.version(pkg))
    except Exception: pass
print("jax_default_matmul_precision =", jax.config.jax_default_matmul_precision)
rng = np.random.default_rng(0)

def bits(e):
    e = float(np.max(e)) if np.size(e) else 0.0
    return float("inf") if e == 0 else -math.log2(e)

def err(got, ref):
    """Max elementwise relative error, and normwise (max-abs) relative error."""
    g = np.asarray(got).astype(np.complex128).ravel()
    r = np.asarray(ref).astype(np.complex128).ravel()
    fin = np.isfinite(r)
    if not np.array_equal(np.isfinite(g), fin):
        return float("nan"), float("nan")          # inf/nan pattern differs
    g, r = g[fin], r[fin]
    if r.size == 0: return 0.0, 0.0
    big = np.abs(r) > 1e-300
    ew = np.max(np.abs(g[big] - r[big]) / np.abs(r[big])) if big.any() else 0.0
    scale = np.max(np.abs(r))
    nw = np.max(np.abs(g - r)) / scale if scale > 0 else np.max(np.abs(g))
    return float(ew), float(nw)

def run(name, f, *args, ref=None):
    """jit f on the accelerator and on CPU (or ref on CPU); compare."""
    try:
        out = jax.block_until_ready(jax.jit(f)(*[jax.device_put(a, ACC) for a in args]))
        want = jax.jit(ref or f)(*[jax.device_put(a, CPU) for a in args])
        outs = out if isinstance(out, (tuple, list)) else [out]
        wants = want if isinstance(want, (tuple, list)) else [want]
        ews, nws = zip(*[err(o, w) for o, w in zip(outs, wants)])
        dts = ",".join(str(o.dtype) for o in outs)
        print(f"{name:34s} OK   {dts:10s} elem={max(ews):.1e} ({bits(max(ews)):4.1f} bits)"
              f"  norm={max(nws):.1e} ({bits(max(nws)):4.1f} bits)")
    except Exception as ex:
        msg = str(ex).strip().splitlines()[0] if str(ex).strip() else ""
        print(f"{name:34s} FAIL {type(ex).__name__}: {msg[:160]}")

print("\n== 1. storage and range (host f64 -> device -> host)")
x = rng.standard_normal(100_000)                     # full 53-bit mantissas
back = np.asarray(jax.device_put(x, ACC))
print("dtype on device:", jax.device_put(x, ACC).dtype,
      "| round-trip max rel err %.1e (%.1f bits)" % (err(back, x)[0], bits(err(back, x)[0])))
probe = np.array([1e-30, 1e-37, 1.2e-38, 1e-39, 1e-45, 1e-300, 5e-324,
                  3.4e38, 3.5e38, 1e300, np.finfo(np.float64).max])
for p, b in zip(probe, np.asarray(jax.device_put(probe, ACC))):
    print(f"   {p!r:>26} -> {b!r}")
run("on-device 1e20*1e20 (IEEE: 1e40)", lambda a: a * a, np.float64(1e20))
run("on-device 1e-20*1e-20 (IEEE:1e-40)", lambda a: a * a, np.float64(1e-20))
run("(1 + 1e-10) - 1 (f32 gives 0)", lambda a, b: (a + b) - a, np.float64(1.0), np.float64(1e-10))
try:   # after the TPU compiler's X64 rewrite, f64 should be gone from the optimised HLO
    hlo = jax.jit(lambda a, b: a * b + a).lower(jax.device_put(u0 := x[:8], ACC),
                                                 jax.device_put(u0, ACC)).compile().as_text()
    print("compiled HLO for a*b+a: 'f64[' x%d, 'f32[' x%d" % (hlo.count("f64["), hlo.count("f32[")))
except Exception as ex:
    print("compiled HLO: FAIL", type(ex).__name__, str(ex).splitlines()[0][:120])

print("\n== 2. elementwise and transcendentals (inputs in [0.5, 2) unless noted)")
u, v = rng.uniform(0.5, 2.0, 10_000), rng.uniform(0.5, 2.0, 10_000)
w = rng.uniform(-30, 30, 10_000)
for nm, f in [("add", jnp.add), ("sub", jnp.subtract), ("mul", jnp.multiply),
              ("div", jnp.divide), ("power", jnp.power), ("arctan2", jnp.arctan2)]:
    run(nm, f, u, v)
for nm, f in [("sqrt", jnp.sqrt), ("exp", jnp.exp), ("log", jnp.log), ("log1p", jnp.log1p),
              ("expm1", jnp.expm1), ("sin", jnp.sin), ("cos", jnp.cos), ("tan", jnp.tan),
              ("tanh", jnp.tanh), ("erf", jax.scipy.special.erf),
              ("gammaln", jax.scipy.special.gammaln), ("logistic", jax.nn.sigmoid)]:
    run(nm, f, u)
run("exp on [-30, 30]", jnp.exp, w)
run("sin on [-30, 30]", jnp.sin, w)
run("lax.fma", lambda a, b, c: lax.fma(a, b, c) if hasattr(lax, "fma") else a * b + c, u, v, u)
run("nextafter", jnp.nextafter, u, v)
run("view as int64 (bitcast)", lambda a: a.view(jnp.int64), u)
run("float64 -> float32 -> float64", lambda a: a.astype(jnp.float32).astype(jnp.float64), u)

print("\n== 3. reductions, scans, control flow, indexing")
big = rng.standard_normal(1_000_000)
run("sum 1e6", jnp.sum, big)
run("cumsum 1e6", jnp.cumsum, big)
run("max, argmax", lambda a: (jnp.max(a), jnp.argmax(a)), big)
run("sort", jnp.sort, u)
run("argsort", jnp.argsort, u)
run("searchsorted", lambda a, b: jnp.searchsorted(jnp.sort(a), b), u, v)
idx = rng.integers(0, 10_000, 50_000)
run("gather x[idx]", lambda a, i: a[i], u, idx)
run("scatter-add x.at[idx].add", lambda a, i: a.at[i].add(1e-3), u, idx)
run("while_loop (Newton sqrt, 60 it)", lambda a: lax.fori_loop(
    0, 60, lambda k, y: 0.5 * (y + a / y), jnp.ones_like(a)), u)
run("scan AR(1) 10k steps", lambda e: lax.scan(lambda c, z: (0.95 * c + z, c), 0.0, e)[1], big[:10_000])
key = jax.random.key(0)
run("random.uniform f64", lambda: jax.random.uniform(key, (100_000,), jnp.float64))
run("random.normal f64", lambda: jax.random.normal(key, (100_000,), jnp.float64))

print("\n== 4. dot / matmul in float64 (does `precision` change anything?)")
A, B = rng.standard_normal((512, 512)), rng.standard_normal((512, 512))
for p in (None, "default", "highest"):
    run(f"matmul f64 precision={p}", lambda a, b, p=p: jnp.matmul(a, b, precision=p), A, B)
run("matmul f64 preset F64_F64_F64", lambda a, b: jnp.matmul(a, b, precision="F64_F64_F64"), A, B)
run("einsum f64", lambda a, b: jnp.einsum("ij,jk->ik", a, b), A, B)

print("\n== 5. jnp.linalg / lax.linalg in float64 (n=256)")
n = 256
M = rng.standard_normal((n, n)); S = M @ M.T / n + np.eye(n); b = rng.standard_normal(n)
L_ = np.linalg.cholesky(S)
run("cholesky", jnp.linalg.cholesky, S)
run("solve (LU)", jnp.linalg.solve, M, b)
run("inv", jnp.linalg.inv, M)
run("det, slogdet", lambda a: (jnp.linalg.det(a), jnp.linalg.slogdet(a)[1]), M)
run("lu_factor", lambda a: jax.scipy.linalg.lu_factor(a)[0], M)
run("triangular_solve", lambda l, r: jax.scipy.linalg.solve_triangular(l, r, lower=True), L_, b)
run("qr (Q@R reconstruct)", lambda a: (lambda q, r: q @ r)(*jnp.linalg.qr(a)), M)
run("eigh eigenvalues (default)", jnp.linalg.eigvalsh, S)
run("eigh residual |SV-VW|/|S| (norm=abs)", lambda a: (lambda w, V: jnp.linalg.norm(
    a @ V - V * w) / jnp.linalg.norm(a))(*jnp.linalg.eigh(a)), S, ref=lambda a: jnp.zeros(()))
run("eigh JACOBI eigenvalues", lambda a: lax.linalg.eigh(
    a, implementation=lax.linalg.EighImplementation.JACOBI)[1], S, ref=jnp.linalg.eigvalsh)
run("svd singular values", lambda a: jnp.linalg.svd(a, compute_uv=False), M)
run("lstsq", lambda a, r: jnp.linalg.lstsq(a, r)[0], M[:, :128], b)

print("\n== 6. FFT, complex128, int64")
run("fft f64 (-> complex128)", jnp.fft.fft, u)
run("rfft f64", jnp.fft.rfft, u)
z = u + 1j * v
run("complex128 mul/abs/exp", lambda q: (q * q, jnp.abs(q), jnp.exp(q)), z)
run("complex128 matmul", lambda q: q[:64, None] @ q[None, :64], z)
I = np.array([2**40 + 1, 2**53 + 1, -(2**62), 12345678901234], dtype=np.int64)
run("int64 arith (*3, //7, %7)", lambda a: (a * 3, a // 7, a % 7), I)
run("int64 sum, cumsum", lambda a: (jnp.sum(a), jnp.cumsum(a)), I)
run("int64 matmul", lambda a: a[:, None] @ a[None, :] // (2**40), I)

print("\n== 7. float32 matmul precision levels (n=1024; error vs CPU float64)")
A32, B32 = (rng.standard_normal((1024, 1024)).astype(np.float32) for _ in range(2))
ref64 = A32.astype(np.float64) @ B32.astype(np.float64)
for p in (None, "bfloat16", "default", "high", "highest", "BF16_BF16_F32", "BF16_BF16_F32_X3",
          "BF16_BF16_F32_X6", "BF16_BF16_F32_X9", "F32_F32_F32", "TF32_TF32_F32"):
    try:
        out = np.asarray(jax.jit(lambda a, b, p=p: jnp.matmul(a, b, precision=p))(
            jax.device_put(A32, ACC), jax.device_put(B32, ACC)))
        e = np.max(np.abs(out - ref64)) / np.max(np.abs(ref64))
        print(f"   precision={str(p):18s} normwise rel err {e:.1e} ({bits(e):4.1f} bits)")
    except Exception as ex:
        print(f"   precision={str(p):18s} FAIL {type(ex).__name__}: {str(ex).splitlines()[0][:120]}")
for pet in (jnp.float32, jnp.float64):
    try:
        out = np.asarray(jax.jit(lambda a, b, pet=pet: lax.dot(a, b, preferred_element_type=pet))(
            jax.device_put(A32, ACC), jax.device_put(B32, ACC)))
        e = np.max(np.abs(out - ref64)) / np.max(np.abs(ref64))
        print(f"   f32 inputs, preferred_element_type={pet.__name__}: dtype {out.dtype}, rel err {e:.1e}")
    except Exception as ex:
        print(f"   preferred_element_type={pet.__name__} FAIL {type(ex).__name__}: {str(ex).splitlines()[0][:120]}")
with jax.default_matmul_precision("highest"):
    out = np.asarray(jax.jit(jnp.matmul)(jax.device_put(A32, ACC), jax.device_put(B32, ACC)))
    print("   jax.default_matmul_precision('highest') context: rel err %.1e"
          % (np.max(np.abs(out - ref64)) / np.max(np.abs(ref64))))
x32 = rng.standard_normal(4096).astype(np.float32)
f_acc = np.asarray(jnp.fft.ifft(jnp.fft.fft(jax.device_put(x32, ACC))).real)
print("   fft->ifft f32 round trip max abs err %.1e" % np.max(np.abs(f_acc - x32)))

print("\n== 8. timing (warm, median of 5; seconds)")
def t(f, *args, dev=ACC):
    try:
        g = jax.jit(f); a = [jax.device_put(x, dev) for x in args]
        jax.block_until_ready(g(*a)); ts = []
        for _ in range(5):
            t0 = time.perf_counter(); jax.block_until_ready(g(*a)); ts.append(time.perf_counter() - t0)
        return f"{np.median(ts):.2e}"
    except Exception as ex:
        return f"FAIL {type(ex).__name__}"
e64 = rng.uniform(0.5, 2, 2**24); e32 = e64.astype(np.float32)
print("   exp+log, 2^24 elems     f32:", t(lambda a: jnp.log(jnp.exp(a) + 1), e32),
      " f64:", t(lambda a: jnp.log(jnp.exp(a) + 1), e64))
for n in (1024, 4096):
    a64 = rng.standard_normal((n, n)); a32 = a64.astype(np.float32)
    print(f"   matmul n={n}  f32 default:", t(lambda a: jnp.matmul(a, a), a32),
          " f32 high:", t(lambda a: jnp.matmul(a, a, precision="high"), a32),
          " f32 highest:", t(lambda a: jnp.matmul(a, a, precision="highest"), a32),
          " f64:", t(lambda a: jnp.matmul(a, a), a64))
rec = lambda r: lax.fori_loop(0, 100_000, lambda k, c: r * c * (1.0 - c), 0.3 * jnp.ones_like(r))
for dev, nm in ((ACC, "accelerator"), (CPU, "host CPU")):
    print(f"   serial recursion 1e5 steps on {nm:11s} f32:",
          t(rec, np.float32(3.7), dev=dev), " f64:", t(rec, np.float64(3.7), dev=dev))
print("\nLAST (compile hung ~20 min on TPU in 2022, jax#10145):")
run("random.randint (f64 maxval)", lambda m: jax.random.randint(key, m.shape, 0, m), np.full((128, 6), 7.0))
print("\nhost CPU:", os.cpu_count(), "logical CPUs;", end=" ")
try: print([l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].strip())
except Exception: print("(no /proc/cpuinfo)")
