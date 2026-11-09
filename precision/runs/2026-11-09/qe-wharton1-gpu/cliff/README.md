# The float64 cliff on qe-wharton1-gpu's Threadripper, 2026-11-09

For QuantEcon/project-benchmarks-jax#7: whether the stored Bellman step's cliff past about 2 GiB comes from YNNPACK's reduce on x86 too, as on [the M6](../../../2026-11-06/qe-mini1/mac-float64/README.md) and [the M5 Pro](../../../2026-11-06/qe-wharton2/mac-float64/README.md). The step is `precision/probes/mac_float64/probe.py`'s `cliff`, from `main` at `ba58764` (the same file as at `3806db6`, sha256 `391eb052…`), in its four variants at 16,000, 18,000 and 20,000, in float64, on the CPU only.

| | |
|---|---|
| Host | `qe-wharton1-gpu`: a Threadripper 2990WX, 64 CPUs, boost off (`acpi-cpufreq`, 3.2 GHz at most), 125 GiB, kernel 7.0.0-42 (`machine.txt`), at the default NUMA placement |
| Software | JAX 0.12.0.1 and NumPy 2.5.4, from the runs clone's venv |
| When | 2026-11-09, 01:26:22 to 01:27:18 UTC, under project-econ-on-ai-hardware's machine lock, with both cards idle (`state.txt`) |
| Script | `session/cliff7.sh`, which ran the four variants with `PROBE_DUMP` set, as `run.sh` does, and released the lock |

**One edit:** `PROBE_HOST` wasn't set, so the rows recorded the machine's hostname. Their `host` is set to `qe-wharton1-gpu`.

## What it shows

GB/s of the m × m array, warm median, with the code each ran as (`lowering`, from the optimised modules in `xla-dump/`):

| m | GiB | Default | YNNPACK's reduce fusion off | The add alone | The row max alone |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 1.91 | 32.6 (ynn) | 3.6 (xla) | 4.2 (xla) | 27.0 (ynn) |
| 18,000 | 2.41 | 2.4 (ynn) | 3.6 (xla) | 4.2 (xla) | 20.4 (ynn) |
| 20,000 | 2.98 | 2.9 (ynn) | 3.5 (xla) | 4.2 (xla) | 23.5 (ynn) |

- **On x86 too, the drop is in YNNPACK's fused reduce:** past 2 GiB the default falls 13.6-fold, from 32.6 to 2.4 GB/s.
- **With YNNPACK's reduce off** (`--xla_cpu_experimental_ynn_fusion_type=-reduce`), XLA's own code holds 3.5–3.6 GB/s at every size. So there is no cliff, but past it that is only 1.2 to 1.5 times the default's rate, and below it nine times slower. On the Macs the same flag held 46 to 80 GB/s.
- **The row max alone,** in YNNPACK, falls by about a quarter rather than collapsing, and XLA's own `add` holds 4.2 GB/s.
