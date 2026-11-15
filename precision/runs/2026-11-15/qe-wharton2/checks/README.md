# Small float32 products on qe-wharton2's M5 Pro, on macOS 26.7.2, 2026-11-15

For QuantEcon/project-benchmarks-jax#4: [the M6's checks of 2026-11-09](../../../2026-11-09/qe-mini1/checks/README.md), run on the M5 Pro from 19:57:31 to 19:59:04 UTC. The script is `session/m5_ynn4.sh`, part 1 of the M6's `m6_window.sh` unchanged but for the host and the paths. #48's probe, its part 2, already ran here on 2026-11-06 ([`../../../2026-11-06/qe-wharton2/`](../../../2026-11-06/qe-wharton2/), #49).

| File | What ran |
|---|---|
| `ynn_dot_check.txt` | `../../../2026-10-25/checks/ynn_dot_check.py`, as on `main` (sha256 `d94bc0fe…`), in float32 and float64, then three times in float32 with `XLA_FLAGS=--xla_cpu_experimental_ynn_fusion_type=` |
| `repro4.txt` | #4's reproducer, `../../../2026-11-09/qe-mini1/checks/session/repro4.py` (`7eacfd22…`), in the same three settings |
| `matrix.txt`, `matrix-default/`, `matrix-ynn-cleared/` | `precision/run_matrix.py`'s `kalman` and `matmul` at 1,000 and 10,000, in both precisions, with and without the fusion, from a worktree at `5fbf033`, the M6's runs commit |
| `state.txt`, `run.log` | The machine every 5 s, and the run's log |

**The machine:** an M5 Pro, 18 cores (6 Super and 12 Performance), 64 GB, macOS 26.7.2 (25G332), on AC power with Low Power Mode off; Python 3.12.15, JAX 0.12.0.1, NumPy 2.5.4 and SciPy 1.18.1, from `~/bench/benchmarks/.venv`. The earlier M5 Pro rows ran on macOS 26.7.1 (25G241); the machine updated on 2026-11-14.

**The `lowering` column** of `ynn_dot_check.txt` also lists `ynn4` and `ynn_dot_check`, since the script's pattern matches the run folder's name and its own in the compiled module. Only `ynn_fusion` and `Custom` say how the product lowered.

**Other load:** it's a guest machine, so nothing was paused. Its owner was logged in at the console, and macOS's software-update service was staging an update. In the 5-second samples that service took:
- **2.8 to 3.3 cores at 19:57:31 and 19:57:40,** during the default `ynn_dot_check`;
- **0.6 to 3.2 cores from 19:57:45 to 19:58:31,** during the fusion-cleared `ynn_dot_check`, the reproducer and the default matrix;
- **under 0.1 of a core from 19:58:36,** during the fusion-cleared matrix.

The other processes listed took 1.2 to 2.4 cores between them until 19:57:55, and under one after. `state.txt` shows managed software as `<managed agent>`, redacted for publication; as in the machine's earlier runs, the run's own account isn't. The load average of 27 at the end is the 10,000 matmul's own 17 cores.

**So the single-threaded figures carry some noise.** Within a script they repeat: the three fusion-cleared `ynn_dot_check` runs agree within 4% at every size from 8 to 64, and within 9% at 128. Between the two scripts, one 10 × 10 product in float64 took 0.258 µs in `ynn_dot_check`, under the update's 3.3 cores, and 0.202 µs in the reproducer. The default matmul at 10,000 shared the machine with the update, so `matrix.txt`'s float32 and float64 rates at 10,000 aren't compared here.

## What they show

| | M6, 2026-11-09 (macOS 27.2) | M5 Pro, 2026-11-15 (macOS 26.7.2) |
|---|---:|---:|
| One 10 × 10 product in float32, µs | 1.078 | 0.828 |
| The same in float64 | 0.166 | 0.258 |
| The same in float32, the fusion cleared | 0.162 | 0.197 to 0.202 |
| #4's reproducer: float32, float64, float32 cleared, µs | 1.098, 0.174, 0.169 | 0.729, 0.202, 0.202 |
| `kalman` at T = 1,000, float32 by default, ms | 5.04 | 4.08 |
| The same, the fusion cleared | 1.40 | 1.96 |
| The same in float64 | 1.36 | 1.69 |
| `matmul` at 1,000, float32 by default, GFLOPS | 1,938 | 2,141 |
| The same, the fusion cleared | 849 | 1,089 |

**So #4's cost is on the M5 Pro too, but smaller.** A small float32 product under YNNPACK's fusion costs about 0.73 to 0.83 µs, against about 0.20 to 0.26 µs in float64 or with the fusion cleared: 3.2 to 3.6 times, against the M6's 6.3 to 6.5.

**The Kalman filter shows the same.** At T = 1,000 it's 2.4 times slower in float32 than in float64 by default, against 3.7 times on the M6. With the fusion cleared it's 1.16 times, against 1.03.

**Clearing the fusion costs about half** of float32's matmul rate at 1,000 on both chips.
