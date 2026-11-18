# The TITAN RTX's sustained load in its own case, 2026-11-18

For QuantEcon/project-benchmarks-jax#65: [`sustained.sh`](../../../sustained.sh) on qe-wharton1-gpu's TITAN RTX at `0b:00.0`, in that workstation's own case, from QuantEcon/benchmarks at `f085c6d`. Its output is in `sustained/`. It ran from 02:11 to 04:32 UTC, after the [two-job test](../../../../problems/results/qe-wharton1-gpu/2026-11-18/README.md), in the same window under econ-on-ai-hardware's machine lock: 10 minutes idle, an hour of `bellman_stored` at 40,000 in float64, 10 minutes idle, and an hour of `matmul` at 8,192 in float32. Turing has no TF32, so the product ran in full float32. The rented GeForce RTX 5090's run, with a datacentre's cooling, is in [`../runpod-rtx5090/`](../runpod-rtx5090/README.md).

**The card:** an NVIDIA TITAN RTX, 24,576 MiB, VBIOS 90.02.23.00.01, on driver 595.99.02. Its power limit is 280 W, the default; it can be set from 100 to 320 W, but `--cap` needs root and wasn't used. The other TITAN RTX, at `43:00.0`, joined to it by an NVLink bridge, drove the desktop at 300 MHz throughout. In `card-q.txt` the card's serial number, UUID and PDI are redacted. Its process list shows Xorg, the display server, holding 4 MiB. No GitHub runner runs on this machine. Nothing records the case's airflow or the room, and the room's temperature wasn't measured.

| An hour each | Last 30 min ÷ first 5, step time | SM clock, first 5 → last 30 | Power | Temperature | Clock-event reasons, last 30 |
|---|---:|---|---|---|---|
| `bellman_stored`, float64, 40,000 | 1.018 (46.9 → 47.7 ms; 273 → 268 GB/s) | 1,860 → 1,830 MHz | 250 → 258 W | 73 → 83 °C | `SwPowerCap` 88%, none 12% |
| `matmul`, float32, 8,192 | 1.035 (80.9 → 83.7 ms; 13.6 → 13.1 TFLOPS) | 1,560 → 1,500 MHz | 277 → 274 W | 76 → 84 °C | `SwPowerCap` throughout |

Before each hour the card was idle at 37–38 °C, 30 W and 300 MHz, against 34 °C at the start of the window, before the two-job test.

**So in its case the TITAN loses a little speed over an hour, held by power.** Neither hour set a thermal reason. As the card warmed by 8–10 °C at its 280 W limit, its clock fell 2–4%, and its step time rose about as much. The rented RTX 5090 in a datacentre lost under 1% over the same hours, but it's a different card, so this doesn't separate the case from the card.
