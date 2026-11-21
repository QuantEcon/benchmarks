# Phase 2's four problem types on qe-wharton1-gpu, 2026-11-21

For [#5](https://github.com/QuantEcon/benchmarks/issues/5)'s Phase 2: P5, P2, P6 and P8 with its three companions, every size in both precisions, on all five stacks: the Threadripper 2990WX for the four CPU stacks, and the TITAN RTX at `0b:00.0` for `jax-gpu`. The same set ran the same day on [`qe-mini1`'s M6](../../qe-mini1/2026-11-21/README.md) and [`qe-wharton2`'s M5 Pro](../../qe-wharton2/2026-11-21/README.md). The rows are in [`../2026-11-21.jsonl`](../2026-11-21.jsonl), and `summary.md` tables them. The same run also made the two checks that [`problems/README.md`](../../../README.md) asks of P8's first Linux and GPU run.

| | |
|---|---|
| Host | `qe-wharton1-gpu`: AMD Ryzen Threadripper 2990WX, 32 cores and 64 threads on four NUMA nodes, two with memory, Linux 7.0.0-43. `jax-gpu` used the TITAN RTX at `0b:00.0` (the suite's `--gpus 0`, in PCI order). The card at `43:00.0` drives the desktop and was idle throughout |
| Software | The runs checkout at `11c1c59`, #82's merge, with its venv: NumPy 2.5.4 and SciPy 1.18.1, each with its own OpenBLAS (0.3.34 and 0.3.31.dev; Numba's BLAS calls go through SciPy's), Numba 0.68.1, JAX 0.12.0.1 with `jax-cuda13-plugin` 0.12.0.1, NVIDIA driver 595.99.02, Python 3.12.3. `setup.txt` lists the CUDA libraries |
| When | 2026-11-21, 05:15:11 to 09:45:48 UTC, and a second session from 10:11:30 to 10:42:14, each under project-econ-on-ai-hardware's machine lock, with both cards idle at the start |
| Script | [`session/wharton1_p2_run.sh`](session/wharton1_p2_run.sh), then [`session/wharton1_p2_rerun.sh`](session/wharton1_p2_rerun.sh), each started with `nohup setsid` |

## What ran

| Part | What | Rows | When (UTC) |
|---|---|---|---|
| suite | `problems/run_suite.py --problems stationary,hpi,time_iter,mle,mle_hand,mle_onepass,mle_blocked --backends numpy,numba,numba-parallel,jax-cpu,jax-gpu --gpus 0`, with each problem's own stacks, sweeps and sizes from `spec.py` and the default limit of 900 s a cell | `20261121T051512Z`, unlabelled, 525 cells | 05:15:11 to 09:27:44 |
| OpenBLAS check | `mle_blocked`'s `params` sweep again on `numba` and `numba-parallel`, with `OPENBLAS_NUM_THREADS=1` | `20261121T092744Z`, labelled `openblas-1-thread`, 24 cells | 09:27:44 to 09:45:41 |
| hoisting check | [`session/hoist_check.py`](session/hoist_check.py) on the card, at 10 and 300 parameters: `hoist.txt` | none | 09:45:41 to 09:45:48 |
| rerun | the suite's one timed-out cell again, with `--timeout 1800` | `20261121T101130Z`, labelled `timeout-1800`, 1 cell | 10:11:30 to 10:41:30 |
| autotuner check | `mle` and `mle_hand` at 10,000,000 observations in float64 on the card, with `XLA_FLAGS=--xla_gpu_autotune_level=0` | `20261121T104131Z`, labelled `autotune0`, 2 cells | 10:41:30 to 10:42:14 |

The suite's problems, by when each one's first cell started:

| Problem | Stacks | Cells | When (UTC) |
|---|---|---:|---|
| P5 `stationary` | all five | 110 | 05:15:11 to 06:16:16 |
| P2 `hpi` | all five, float64 only | 55 | 06:16:16 to 06:25:54 |
| P6 `time_iter` | all five | 100 | 06:25:54 to 08:09:44 |
| P8 `mle` | all five | 120 | 08:09:44 to 08:41:35 |
| P8 `mle_hand` | `jax-cpu`, `jax-gpu` | 48 | 08:41:35 to 08:48:44 |
| P8 `mle_onepass` | `numba`, `numba-parallel` | 44 | 08:48:44 to 09:01:49 |
| P8 `mle_blocked` | `numba`, `numba-parallel` | 48 | 09:01:49 to 09:27:44 |

`run.log` is the suite's output, then the OpenBLAS check's, the rerun's and the autotuner check's. `state.txt` records the first session, every 10 minutes, and `state-rerun.txt` the second, every 5.

## Checks

- **524 of the 525 cells ran,** status `ok`. P6's `numpy` at a grid of 50,000 in float32 passed the 900 s limit, and its row has status `timeout`. Its rerun with 1,800 s, labelled `timeout-1800`, passed that limit too, so its cold and warm solves took about 900 s each or more. The same cell in float64 took 551 s for the process, with one warm run of 275 s. So float32 is over 3.3× slower here, where it is 1.47× slower at 20,000 on this host and 1.76× at 50,000 on the M6. These runs don't say what else slows it at this size, and the cell wasn't run again (Matt, 2026-11-21). The longest of the card's 101 cells took 140 s.
- **Every answer but two agrees with NumPy's in float64** at the same size, within `spec.py`'s tolerances, P8's companions' with `mle`'s NumPy. The two are both in P5's `dense` sweep in float32, at a tolerance of 1e-3:
    - `jax-cpu` at 1,000 assets, 2.0e-3 off;
    - `jax-gpu` at 5,000, 1.4e-2 off.

  On this host the float32 `dense` answers are mostly further off than on the Macs, NumPy's own included: 6e-6 to 1.4e-2, with NumPy's at 7.8e-5 to 8.9e-4, against 5e-6 to 8e-5 for every stack on both Macs.
- **The iteration counts that differ, all in float32,** are listed under their tables in `summary.md`:
    - P5's `sparse` sweep takes the Macs' counts, 157 to 192 against float64's 266 to 299, and `jax-gpu` 195 at 10,000,000 against NumPy's 192.
    - P8's NumPy takes float64's counts here, 17 at 3,000,000 observations and 16 at 10,000,000, where on the Macs it takes one more. `jax-cpu` takes 18 and 19 there, in `mle` and `mle_hand`.

## P8's two checks

- **`mle_blocked`'s BLAS calls inside `prange` do oversubscribe the cores.** With OpenBLAS held to one thread, `numba-parallel`'s median warm time is lower at every size, by up to 11× in float64, at 300 parameters (1.03 s against 11.8 s), and up to 19× in float32, at 30 (0.00745 s against 0.139 s). At 1,000 it takes 20.3 s against 43.5 s, and 5.58 s against 24.3 s. The smallest gain is 1.09×, at 10 parameters in float32, where the two cells' warm runs overlap. Serial Numba is slower held to one thread at 300 and 1,000 parameters, since by default its BLAS calls use OpenBLAS's threads: 162 s against 50.7 s at 1,000 in float64. Below 300 it is within 8%, except at 30 in float32, where one thread made it 2.55× faster (0.0919 s against 0.234 s).
- **On the card, XLA doesn't take autodiff's extra product out of the Newton loop.** In `hoist.txt`, which compiles both solves for the card in float64, the autodiff solve's [k, n] product sits inside the while loop at both 10 and 300 parameters, so it is computed at every step. The solve written by hand has no such product. That XLA's CPU backend computes it once, outside the loop, comes from #81's development on the M6, as [`problems/README.md`](../../../README.md)'s P8 note records, and not from this check. The rows agree with both. On the card in float64, `mle` takes 1.73× to 1.95× as long as `mle_hand` from 100 parameters up, 11.6 s against 5.96 s at 1,000, and 1.36× to 1.63× in float32. On `jax-cpu` it takes 0.89× to 1.17× as long there, on all three CPUs.

## The card's float64 compiles

On P8's `obs` sweep in float64, the card's compile time grows with the observations: from 0.40 s for `mle` and 0.35 s for `mle_hand` at 10,000 to 11.8 s and 22.0 s at 10,000,000, where each first run takes under a second and every float32 compile under 1 s. With XLA's autotuner off (`autotune0`), the compiles at 10,000,000 take 0.37 s and 0.31 s, as at 10,000, so the autotuner, which times candidate kernels on buffers of the real size, accounts for them, as the 2026-11-01 notes found on other cards. The kernels it finds are faster:

| At 10,000,000, float64 | Compile, autotuned | Compile, off | Warm, autotuned | Warm, off | Cold, autotuned | Cold, off |
|---|---:|---:|---:|---:|---:|---:|
| `mle` | 11.8 s | 0.37 s | 0.901 s | 3.32 s | 13.4 s | 4.53 s |
| `mle_hand` | 22.0 s | 0.31 s | 0.869 s | 2.64 s | 23.6 s | 3.70 s |

So one solve in a fresh process is faster with the autotuner off, and the autotuner pays for itself from the 5th solve in one process for `mle` and from the 13th for `mle_hand`.

## The machine during the run

- **CPU:** `state.txt` samples the machine every 10 minutes. Besides the run and the sampler's own `ps` and `head`, the only processes above 1% of a core in any sample were `btop`, a system monitor open in a terminal, at about 3%, and a kernel worker for the network card at 1.4%.
- **The cards:** the ten-minute samples caught `0b:00.0` idle each time, at 300 to 405 MHz, since `jax-gpu`'s cells are short beside the CPU stacks'. The record after the run, taken just after the hoisting check, has it at 1,935 MHz and 66%, and the second session's after the autotuner check at 1,920 MHz and 71%. `43:00.0` stayed at 300 MHz and 0% throughout.
