# sort_iota_repro.py: wrong results from a 3-operand stable lax.sort on AMD Instinct MI355X (gfx950),
# with default XLA flags.   python3 sort_iota_repro.py
#
# lax.sort((s32 keys, f64 values, s32 iota), num_keys=1) with the iota made inside the jit (so XLA fuses
# it into the sort and uses it for the stable tie-break). Every output row must be an input row; the iota
# column tells which. On MI355X, about one call in five returns an iota column in which 32, 64 or 96
# entries are copies of other entries (the keys and values are still a correct sorted permutation).
import argparse, os, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

jax.config.update('jax_enable_x64', True)
ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=3 * 2**20, help='number of rows')
ap.add_argument('--repeats', type=int, default=100)
a = ap.parse_args()
n = a.n
print(f'jax {jax.__version__}  {jax.devices()[0].device_kind}  XLA_FLAGS={os.environ.get("XLA_FLAGS", "")!r}')

rng = np.random.default_rng(0)
keys = rng.integers(0, 3 * 2**16, n).astype(np.int32)
vals = rng.standard_normal(n)
sorted_keys = np.sort(keys)

@jax.jit
def sort3(k, v):
    return lax.sort((k, v, lax.iota(jnp.int32, n)), num_keys=1)

k_d, v_d = jnp.asarray(keys), jnp.asarray(vals)
bad, t0 = [], time.time()
for r in range(a.repeats):
    sk, sv, si = (np.asarray(x) for x in sort3(k_d, v_d))
    src = np.clip(si, 0, n - 1)
    nbad = int(np.sum((sk != sorted_keys) | (keys[src] != sk) | (vals[src] != sv) | (si != src)))
    dup = int(np.sum(np.bincount(src, minlength=n) > 1))
    if nbad or dup:
        bad.append(r)
        if len(bad) <= 3:
            print(f'  sort call {r}: {nbad} output rows are not the input row their iota points to; '
                  f'{dup} iota values appear more than once')
print(f'lax.sort((s32, f64, iota)) of {n} rows: {(time.time() - t0) / a.repeats * 1e3:.0f} ms per call '
      f'incl. checks; wrong in {len(bad)} of {a.repeats} calls')
print('RESULT', 'WRONG' if bad else 'OK')
