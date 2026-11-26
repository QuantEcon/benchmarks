# An MI350X under ROCm 10.1, 2026-11-05

This was one rented session, started from a laptop for QuantEcon/project-benchmarks-jax#32, on one AMD Instinct MI350X (`gfx950`).
- **What ran:** Phase 2's precision set and the problem-type suite, the first run of either on an AMD card, then two ROCm/jax reproducers.
- **Afterwards:** the droplet was destroyed once its results were copied off.

| | |
|---|---|
| Host | `digitalocean-gpu-mi350x1-288gb-spot`: DigitalOcean's `gpu-mi350x1-288gb-spot` droplet in ric1, on spot, with 24 vCPUs of an AMD EPYC 9575F and 252 GiB. DigitalOcean listed no one-GPU MI300X that day, so this card went first |
| Image | "ROCm 10.1 on Ubuntu 24.04" (`amddeveloperclou-rocm101`, 2026-10-07): Ubuntu 24.04.5, kernel 6.8.0-146-generic |
| Card | AMD Instinct MI350X VF, a virtual function, `gfx950`, 294,592 MB, with amdgpu 6.19.14.31400000, ROCm 10.1.0 and AMD SMI 27.1.0. JAX's allocator gets 215.77 GiB by default, and 273.3 GiB at a memory fraction of 0.95 (`machine.txt`) |
| Software | JAX 0.12.0 with `jax-rocm10-plugin` and `jax-rocm10-pjrt` 0.12.0, from `constraints-rocm.txt`, on Python 3.12.3 (`environment.txt`) |
| Harness | `5cd9d3d`, with `ROCM_MAJOR=10` (see below) |
| On the machine (UTC) | 2026-11-05 22:55 to 2026-11-06 00:33, creation to destruction |

**Where the results are:**
- **The precision set:** `results/digitalocean-gpu-mi350x1-288gb-spot/2026-11-05.jsonl`, and `lectures/results/digitalocean-gpu-mi350x1-288gb-spot/2026-11-05/`, which holds `machine.txt`, `environment.txt`, `precision-summary.md` and the lectures' `whole/`.
- **The problem types:** `problems/results/digitalocean-gpu-mi350x1-288gb-spot/`, with [their own notes](../../../problems/results/digitalocean-gpu-mi350x1-288gb-spot/2026-11-05/README.md).
- **This folder's `digitalocean-gpu-mi350x1-288gb-spot/`:** the session's scripts and logs (`session/`), the reproducers and their output (`repro/`), and the first attempt's logs (`attempt1/`).

## The image's ROCm had no `gfx950` libraries

- **The first attempt failed,** from 23:03 to 23:07. Both `--quick` checks ran (`attempt1/`).
    - Every float64 matrix product failed with "All backends failed to find supported configs: HipblasLt error 3".
    - The Kalman filter failed in both precisions: in float64 because hipBLASLt had no configurations ("Failed to get configs"), and in float32 with jaxlib's "invalid kernel file".
    - Float32 matrix products, the Bellman steps, P3 in float32, P4, `schelling_shocks` and both scale scripts ran.
- **The cause:** the image's ROCm 10.1 carries device libraries for `gfx942`, the MI300X, alone. A diagnostic, `session/diag_rocm101.py`, showed it (output in `attempt1/diag_rocm101.txt`):
    - its packages are `amdrocm-*10.1-gfx942`;
    - hipBLASLt couldn't find `TensileLibrary_lazy_gfx950.dat` or `Kernels.so-000-gfx950.hsaco`;
    - rocBLAS reported "Illegal seek for GPU arch : gfx950", then aborted.
    The image also holds a pre-release ROCm 7.14 (`7.14.0~pre3`) with `gfx950` libraries, which wasn't used.
- **The fix:** install the nine `gfx950` counterparts of the installed `gfx942` packages.
    - **The packages:** `amdrocm-{core,core-dev,blas,solver,sparse,fft,rand,dnn,rccl}10.1-gfx950` at 10.1.0-3, the same build. They came from the image's own apt source, `stable.repo.amd.com` (`session/apt-gfx950.log`, `session/apt-history-gfx950.txt`).
    - **Unsigned:** the image marks that source `[trusted=yes]`, so apt checked no signatures.
    - **Nothing else changed:** `session/dpkg-amdrocm-before.txt` and `session/dpkg-amdrocm-after.txt` differ by exactly those nine packages.
    - **The result:** with them in, the diagnostic passed in both precisions (`session/diag_rocm101-after-gfx950.txt`). So the driver is the image's, and the ROCm 10.1 libraries are the image's with AMD's `gfx950` packages added.
- **The JAX plugin wasn't the problem:** it carries `gfx950` code itself. `llvm-objdump --offloading` on `jax_rocm10_plugin/_linalg.so` lists `gfx950` among the targets of a compressed bundle; `strings` finds none, because the bundle is compressed. The "invalid kernel file" was most likely a stale error from the failed library load, and it went once the libraries were in.

## Other things about the image

- **The ROCm version file is elsewhere:** `/opt/rocm/.info/version` doesn't exist; the file is `/opt/rocm/core-10.1/.info/version`.
    - So both `run_by_hand.sh` scripts read "ROCm none", and they ran with `ROCM_MAJOR=10`, their own override.
    - `env_info.py` falls back to `amd-smi`'s "ROCm version: 10.1.0", so the rows record 10.1.0.
- **A public notebook server:** the image runs a JupyterLab container as root on public port 8888. It was stopped before the runs.
- **Its firewall limits SSH,** refusing an address that opens six connections in 30 s, so the session kept to one multiplexed connection.

## The runs, on the fixed stack

| Step | Start (UTC) | Wall | `run_id` | Rows |
|---|---|---|---|---|
| Precision, `--quick` | 23:11:28 | 79 s | 20261105T231130Z, and 20261105T231214Z for `highest` | Left out of the `.jsonl` |
| Problem types, `--quick` | 23:12:47 | 34 s | 20261105T231248Z | Left out |
| Precision, full | 23:17:48 | 990 s | 20261105T231751Z, and 20261105T232807Z for `highest` | 92 and 12, all `ok` |
| Problem types, full | 23:34:18 | 262 s | 20261105T233419Z | 50, all `ok` |
| Problem types, `highest` pass | 23:38:40 | 97 s | 20261105T233840Z | 22, all `ok` |
| Reproducers | 23:40:17 | 115 s | | `repro/` |

- **Every step exited 0.** Each ran through `session/step.sh`, and `session/full.sh` chained the full runs.
- **The `highest` pass** is `session/highest.sh`, as on the RTX PRO 6000 (`problems/results/aws-g7e.2xlarge/2026-11-05/`).
- **The `--quick` rows are left out,** because the full runs repeat their cells. Their output is in `session/`.

## What they show

- **Every precision cell ran, and agrees with the droplet's CPU.** In float64, the card's results differ from the CPU's by at most:
    - 5.8e-15 on the matrix product;
    - 7.9e-16 on the Bellman steps;
    - 7.3e-8 on the Kalman filter.
  All three are within Phase 2's H100 (`precision-summary.md`).
- **The float64 matrix product reached 69.8 TFLOPS at n = 8,192,** and float32 reached 138 TFLOPS. Under `highest`, float32 ran at the same speed.
- **At JAX's default precision, a float32 matrix product is full float32 on this card.**
    - P3's float32 iterations, 2,204 to 2,212 against NumPy float64's 2,190 to 2,192, are the same under `highest` at every size, and every answer agrees.
    - On the RTX PRO 6000, TF32 cut them to 809 to 1,381.
- **The Kalman filter is slow on the card:** 3.6 to 4.1 thousand steps a second, against 286 to 342 thousand on the droplet's CPU. P7 takes 0.24 s to 0.33 s warm for a filter of 1,000 steps, at every state size and for batches up to 1,000.
- **The five lectures** ran three times each, with every cell ok (`whole/lectures-set.log`). `opt_invest`'s first run took 30.4 s, and its others 11.8 s.
- **The reproducers** come from ROCm/jax#895 and ROCm/jax#896, which report both bugs on an MI355X, the other `gfx950` part, and not on an MI300A.
    - **How they ran:** through `repro/run_repro.sh`, which fetched each issue's script from its body at run time.
    - **The saved scripts:** those fetched scripts are saved beside it as `rocm-jax-895.py` and `rocm-jax-896.py`. Each matches its issue's body, last edited on 2026-10-21, before the run (sha256 `b8abbe75d297…` and `7f76e190e096…`).
    - **To re-run them,** use the saved copies rather than fetching again, since an issue's body can change.

  The results:
    - **ROCm/jax#895's three-operand sort** was wrong in 19 of 100 calls. Through `jnp.argsort(x, stable=True)`, the two-operand sort, it was wrong in 0 of 100 (`repro/argsort.py`).
    - **ROCm/jax#896's scatter-add** was right in all 100 calls at the default. It was wrong in 3 of 100 calls with `--xla_gpu_enable_scatter_determinism_expander=true`, by up to 0.20, and in 3 of 100 with `--xla_gpu_deterministic_ops=true`, by up to 0.19.
