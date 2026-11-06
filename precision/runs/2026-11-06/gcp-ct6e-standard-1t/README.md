# A TPU v6e on Google Cloud, 2026-11-06

This was one rented session, started from a laptop for QuantEcon/project-benchmarks-jax#38, on one TPU v6e chip. It's a diagnostic, not a harness run. The harness has no TPU device at `5cd9d3d`, so nothing here is in `results/` form.
- **What ran:** QuantEcon/project-benchmarks-jax#38's float64 probe, the precision set's cells through `precision/run_cell.py`, the five lectures and the two scale scripts.
- **Afterwards:** the instance was deleted, once its results were copied off.

| | |
|---|---|
| Host | `gcp-ct6e-standard-1t`: Compute Engine's `ct6e-standard-1t`, one TPU v6e chip ("TPU v6 lite", 32 GB), on spot in europe-west4-a, with 44 vCPUs of an AMD EPYC 9B14 and 172 GiB (`probe/machine.txt`). Five other zones had no capacity that day, and a sixth returned an internal error |
| Image | `ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e-v20261104a`: Ubuntu 22.04.5, kernel 6.8.0-1070-gcp |
| Software | JAX 0.12.0.1 with `jax[tpu]` and libtpu 0.0.50.1, on Python 3.12.16, under `precision/constraints.txt` (`probe/environment.txt`). The lectures added the TITAN RTX run's packages under the same constraints (`lectures/environment.txt`) |
| Harness | `5cd9d3d`'s `precision/run_cell.py` and `lectures/run_lecture.py`, unchanged, driven by this folder's own loops |
| On the machine (UTC) | 03:15 to 04:47, creation to deletion. Spot reclaimed it 56 s after creation, it was started again by 03:19, and it wasn't reclaimed again |

**Where the results are,** all in this folder:
- **`probe/`:** the probe, `tpu_f64_probe.py`, as attached to QuantEcon/project-benchmarks-jax#38, with its sha256 in `probe-sha256.txt`. It's the same file the MI300X ran, in [`../digitalocean-gpu-mi300x1-192gb/tpu-probe/`](../digitalocean-gpu-mi300x1-192gb/tpu-probe/). Its output is `probe.txt`, beside the pip freeze, `devices.txt` and `machine.txt`.
- **`cells/`:** `tpu-cells.jsonl`, 52 rows, and `tpu_cells.sh`, the loop that ran them.
- **`lectures/`:** `tpu_lectures.sh`, the packages, `whole/`, the scale scripts' JSON and `aiyagari-summary.md`.
- **`session/`:** the instance's set-up script, each step's log and `.done` file, and `laptop/`, the scripts that created, synced, watched and deleted the instance.

**The cells' rows aren't the harness's rows.** `precision/run_matrix.py` accepts only `cpu` and `gpu`, so `cells/tpu_cells.sh` stands in for it.
- **What runs the same:** one fresh process per cell, with `JAX_PLATFORMS=tpu` and `JAX_ENABLE_X64` set by precision, as `run_matrix.py` sets them.
- **What the rows have:** `run_cell.py`'s own fields, plus the loop's `host`, `workload`, `size`, `dtype`, `device` ("tpu"), `label`, `run_id`, `timestamp` and `cell_env`.
- **What they lack:** `run_matrix.py`'s `bound`, `unit`, `work`, `env`, `prefix`, `process_s` and `cell_flags`.
- **No CPU rows:** none ran on this host.

## The runs

Each step ran in tmux, one process at a time, since one process holds the chip.

| Step | Start (UTC) | Wall | What it left |
|---|---|---|---|
| Set-up (`session/setup_and_probe.sh`) | 03:19:36 | 8 s | `session/setup.log`; exit 0 |
| The probe | 03:19:44 | 385 s | `probe/probe.txt`; exit 0 |
| The precision cells (`cells/tpu_cells.sh`) | 03:26:11 | 712 s | `run_id` 20261106T032611Z: 52 rows, 50 `ok` |
| The lectures, then the scale scripts (`lectures/tpu_lectures.sh`) | 03:38:10 | Until the deletion | `lectures/whole/lectures-set.log` |

- **The cells' order:** the quick sizes first, so that a reclaim would still leave the core cells, then the other sizes, then every float32 matrix product again with `JAX_DEFAULT_MATMUL_PRECISION=highest`, labelled `highest`.
- **The lectures' order:** one run of each lecture, then the scale scripts, then the second and third runs, each lecture under `timeout 900`.
- **The deletion:** a watchdog on the laptop copied the results off and deleted the instance at 90 minutes, ending at 04:47:40. The results were also copied off every 90 s throughout.

## What they show

- **How to read the probe:** it prints `OK` for any check that runs without raising, so an `OK` with `nan` errors is a mismatch, not a pass. It means the result differs from the CPU's in which values are finite.
    - **Two lines show it:** `1e20*1e20`, which overflows to inf on the TPU where the CPU gives 1e40, and `det, slogdet` at n = 256.
    - **Its int64 checks compare through complex128,** so they can't show a difference in the low bits of values above 2^53. These notes draw nothing from them but the int64 matrix product's failure.
- **float64 is emulated, as a pair of float32s** (`probe/probe.txt`, section 1).
    - **Precision:** a round trip keeps 49 bits, against IEEE float64's 52.
    - **Range:** float32's. 1e-300 becomes 0, and 3.5e38 becomes inf, while subnormals down to 1.4e-45 survive.
    - **Outside linear algebra:** most operations keep 45 to 49 bits normwise. Of the elementwise functions, the worst are cos and tan at 34.5 bits elementwise, subtraction at 34.9, log at 36.1 and `gammaln` at 27.7.
    - **Matrix products:** float64 ones keep 48.4 bits normwise, and `precision` changes nothing.
    - **Linear algebra keeps less, without raising an error** (section 5, at n = 256). The same probe on the MI300X, whose float64 is IEEE ([`../digitalocean-gpu-mi300x1-192gb/tpu-probe/probe-mi300x.txt`](../digitalocean-gpu-mi300x1-192gb/tpu-probe/probe-mi300x.txt)), shows what the test matrices themselves allow.
        - **About float32's accuracy:** normwise, `lstsq` keeps 25.1 bits and `eigh`'s residual 24.4, against 47.9 and 49.5 on the MI300X.
        - **Less lost:** `eigh`'s eigenvalues keep 33.8 bits, against 50.7, and `inv`, `svd` and `cholesky` lose 5.6 to 6.6 bits.
    - **What fails in float64:** `nextafter`, FFTs of float64 or complex128, int64 matrix products, and the `F64_F64_F64` dot algorithm.
- **float32 matrix products default to one bfloat16 pass:** 8.6 bits normwise at n = 1,024. `high` makes it three passes, at 16.2 bits, and `highest` six, at 22.6.
- **In float64, the precision cells agree with a CPU** as closely as Phase 2's H100 did with its own.
    - **The reference:** the cells' inputs are seeded the same on every host, so the H200 droplet's CPU rows serve as one (`results/digitalocean-gpu-h200x1-141gb/2026-11-06.jsonl`). In float64 they agree with the MI300X droplet's CPU rows to within 6.1e-12.
    - **The differences:** at most 2.2e-15 on the matrix product, 6.0e-15 on the Bellman steps and 1.4e-8 on the Kalman filter.
    - **Phase 2's H100, against its own CPU:** 6e-15, 6e-15 and 9e-7.
    - **float32 matrix products** differ from the same CPU's float32 by up to 5.0e-4 at the default, and 7.7e-7 under `highest`.
- **The rates,** at the largest size each cell ran:

    | Precision cell | float64 | float32 |
    |---|---:|---:|
    | Matrix product, n = 8,192, TFLOPS | 5.27 | 449 at the default, 127 under `highest` |
    | Stored Bellman step, GB/s | 139 at n = 20,000 | 1,147 at n = 60,000 |
    | Recomputing Bellman step, n = 60,000, Gcells/s | 2.09 | 739 |
    | Kalman filter, 100,000 steps, ksteps/s | 3.09 | 114 |

- **The two failed cells** are the float64 stored step at 40,000 and 60,000.
    - **The error:** compiling ran out of the chip's 32 GB, with `CompileTimeHbmOom`.
    - **How much they needed:** 62.68G and 140.89G of temporaries, as XLA reports them, for arrays of 12.8 GB and 28.8 GB.
    - **float32 ran at both sizes.**
- **Compiles:** float64 cells compile in 0.4 s to 3.8 s. float32 matrix products under `highest` take 8.4 s to 10.8 s from n = 2,048.
- **The lectures:** four ran as written with every cell ok.

    | Lecture | Runs, seconds |
    |---|---|
    | `overborrowing` (float32) | 126.0, 122.1, 122.1 |
    | `wealth_dynamics_jax` (float32) | 82.0, 81.8, 82.0 |
    | `schelling_shocks` (float32) | 37.8, 37.7, 37.7 |
    | `hopenhayn` (float64) | 145.1, 145.0; the third wasn't reached |

- **`opt_invest` (float64) didn't finish.**
    - The timeout ended its first two runs at 900 s, and the deletion its third.
    - `run_lecture.py` writes its record only at the end, so which cell it was in isn't known.
    - `whole/opt_invest/tpu-1.err` and `tpu-2.err` hold only the SIGTERM's stack trace, from a wait for a result on the device.
    - QuantEcon/project-benchmarks-jax#42 follows it up.
- **The scale scripts** ran once each.
    - **`aiyagari_jax`** converged in both precisions at every grid from 200 to 10,000, in 11 to 15 iterations. In float32 at 5,000 its value sum stayed finite. On the H100 and the H200 that case gives NaN.
    - **`ifp_egm_jax`** ran in both precisions, with finite answers.
    - **`aiyagari-summary.md`** is nearly empty, because `summarise_scale.py` doesn't tabulate a `tpu` platform. The JSON has every result.
