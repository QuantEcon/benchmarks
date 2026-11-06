# scatter_loop_repro.py: wrong scatter-add results on AMD Instinct MI355X (gfx950) with
# --xla_gpu_enable_scatter_determinism_expander=true or --xla_gpu_deterministic_ops=true.
#   XLA_FLAGS=--xla_gpu_enable_scatter_determinism_expander=true python3 scatter_loop_repro.py   # fails
#   XLA_FLAGS=--xla_gpu_deterministic_ops=true python3 scatter_loop_repro.py                     # fails
#   python3 scatter_loop_repro.py                                                                 # passes
#
# A lax.scan of T steps; each step scatter-adds 3,145,728 updates (3 fields x 256 x 256 cells x a 4x4
# stencil around a displaced point, i.e. the transpose of a semi-Lagrangian advection gather) into a
# zero f64[3,256,256] array. The displaced points are precomputed integer tables (one per step), so numpy
# uses exactly the same indices; the reference uses np.bincount. Only the summation order differs, so the
# results must agree to rounding (~1e-15 relative); with either flag they must also be bitwise identical.
import argparse, os, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_enable_x64', True)
ap = argparse.ArgumentParser()
ap.add_argument('--N', type=int, default=256)
ap.add_argument('--F', type=int, default=3)
ap.add_argument('--T', type=int, default=200, help='scatter-adds per call (scan length)')
ap.add_argument('--calls', type=int, default=100)
ap.add_argument('--disp', choices=['smooth', 'hash'], default='smooth')
a = ap.parse_args()
N, F, T = a.N, a.F, a.T
print(f'jax {jax.__version__}  {jax.devices()[0].device_kind}  XLA_FLAGS={os.environ.get("XLA_FLAGS", "")!r}')

gy, gx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
if a.disp == 'smooth':    # departure point of a smooth, time-dependent flow, up to 3 cells away
    Y0 = np.stack([np.floor(gy - 3 * np.sin(gx / 17 + 0.3 * t) * np.cos(gy / 23)) for t in range(T)])
    X0 = np.stack([np.floor(gx - 3 * np.cos(gy / 13 - 0.2 * t) * np.sin(gx / 29)) for t in range(T)])
else:                     # pseudo-random integer displacement in -3..3 cells
    Y0 = np.stack([gy - ((gy * 7 + gx * 13 + t * 5) % 7 - 3) for t in range(T)])
    X0 = np.stack([gx - ((gy * 11 + gx * 3 + t * 3) % 7 - 3) for t in range(T)])
Y0, X0 = Y0.astype(np.int32), X0.astype(np.int32)

def indices(y0, x0, xp):
    off = xp.arange(-1, 3, dtype=xp.int32)
    iy = xp.clip(y0[..., None] + off, 0, N - 1)          # (N, N, 4)
    ix = xp.clip(x0[..., None] + off, 0, N - 1)
    return iy[:, :, :, None], ix[:, :, None, :]           # broadcast to (N, N, 4, 4)

def step(f, yx):
    iy, ix = indices(*yx, jnp)
    u = jnp.broadcast_to(f[:, :, :, None, None] / 16, (F, N, N, 4, 4))
    return jnp.zeros_like(f).at[:, iy, ix].add(u), None

run = jax.jit(lambda f, y0, x0: lax.scan(step, f, (y0, x0))[0])

def reference(f):
    for t in range(T):
        iy, ix = (np.broadcast_to(x, (N, N, 4, 4)) for x in indices(Y0[t], X0[t], np))
        key = (np.arange(F)[:, None, None, None, None] * N * N + iy * N + ix).ravel()
        f = np.bincount(key, np.broadcast_to(f[:, :, :, None, None] / 16, (F, N, N, 4, 4)).ravel(),
                        minlength=F * N * N).reshape(F, N, N)
    return f

rng = np.random.default_rng(0)
f0 = [rng.uniform(0.5, 1.5, (F, N, N)) for _ in range(2)]     # two inputs, used alternately
ref = [reference(x) for x in f0]
f0_d = [jnp.asarray(x) for x in f0]
Y0_d, X0_d = jnp.asarray(Y0), jnp.asarray(X0)
errs, outs, t0 = [], [set(), set()], time.time()
for c in range(a.calls):
    g = np.asarray(run(f0_d[c % 2], Y0_d, X0_d))
    errs.append(np.max(np.abs(g - ref[c % 2])) / np.max(np.abs(ref[c % 2])))
    outs[c % 2].add(g.tobytes())
    if errs[-1] > 1e-10:
        print(f'  call {c}: max relative error vs numpy {errs[-1]:.1e}')
wrong = sum(e > 1e-10 for e in errs)
print(f'{T} scatter-adds of {F * N * N * 16} updates per call, {(time.time() - t0) / a.calls * 1e3:.0f} ms per call; '
      f'wrong in {wrong} of {a.calls} calls (max rel. error {max(errs):.1e}); '
      f'{len(outs[0]) + len(outs[1])} distinct results for 2 inputs')
flags = os.environ.get('XLA_FLAGS', '')
det = 'scatter_determinism_expander=true' in flags or 'deterministic_ops=true' in flags
# without either flag, the scatter-add uses atomics and results differ at the rounding level, as expected
print('RESULT', 'WRONG' if wrong else ('NONDETERMINISTIC' if det and len(outs[0]) + len(outs[1]) > 2 else 'OK'))
