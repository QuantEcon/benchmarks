# A `.noindex` folder for XLA's dump on qe-mini1's M6, 2026-11-10

On 2026-11-06 and 2026-11-09, macOS's media-analysis daemon, `mediaanalysisd`, woke within about 3 s of the cliff step of [QuantEcon/project-benchmarks-jax#48's probe](../../../../probes/mac_float64/README.md), which writes an XLA dump ([2026-11-09's notes](../../../2026-11-09/qe-mini1/mac-float64/README.md)). This check asks whether a folder whose name ends in `.noindex`, which Spotlight leaves alone, stops that.

It ran on `qe-mini1`'s M6 on macOS 27.2.1 (26B131), from 04:06:30 to 04:14:36 UTC, with the account logged in at the screen. The window was agreed with the Claude sessions on the Mac, which stayed running and held builds, tests and other file-heavy work. The script is [`session/noindex_check.sh`](session/noindex_check.sh).

## What ran

The probe's cliff step, `probe.py cliff <variant> 16000,18000` for its four variants, as `run.sh` runs them. The probe is byte-identical to `precision/probes/mac_float64/probe.py` at `ee5ddfa`, sha256 `391eb052…`, and it ran with the benchmarks venv. A pass of the four variants takes about 4 s.

It ran three ways, in three rounds, alternating, with a 45 s rest before each pass:

| Way | Where the dump went | Files a pass |
|---|---|---:|
| `plain` | an ordinary folder under `~/bench`, as `run.sh` did | 346 |
| `noindex` | a folder named `dump.noindex` | 346 |
| `nodump` | no dump | 0 |

`run.sh` deletes all but the optimised modules after the step, which is why the earlier runs' folders hold 32 files. The dumps aren't kept here.

| File | What it holds |
|---|---|
| `samples.txt` | Every second, the CPU of the media and Spotlight processes above zero, as `ps` gives it, with each pass's start and end |
| `cliff-plain.jsonl`, `cliff-noindex.jsonl`, `cliff-nodump.jsonl` | The probe's rows, three rounds of four variants at two sizes each |
| `state.txt`, `run.log` | The machine before and after, and the window's log |

## What it shows

**A `.noindex` folder stops the dump waking `mediaanalysisd`.** CPU-seconds over each pass and the 45 s after it, summed from `samples.txt`, one figure for each round:

| Way | `mediaanalysisd` | Spotlight: `mdworker_shared`, `mds_stores`, `mds` |
|---|---|---|
| `plain` | 38.5, 37.4, 39.1 | 3.9, 3.2, 5.0 |
| `noindex` | 2.6, 1.6, 2.1 | 0.2, 0.1, 0.0 |
| `nodump` | 1.9, 1.4, 1.0 | 0.4, 0.2, 0.3 |

- **With the dump in an ordinary folder,** `mediaanalysisd` ran at up to 123% of a core for about 33 s after each pass, and Spotlight indexed the files.
- **With the dump in a `.noindex` folder,** the daemons did what they did with no dump at all.
- **In both, a blip of 2 s to 4 s, at up to 114%,** came at each pass's start. One came at the script's own start too (1.2 CPU-seconds), before any probe ran, when it wrote its first files. So small writes outside a `.noindex` folder, or a process starting, set it off. A `.noindex` folder doesn't stop that.
- **The figures are approximate.** `ps` gives a decaying average of each process's CPU, not its use in that second. That spreads a burst over the seconds after it, but it doesn't change which way is which: the gap is 15× to 39×, round by round.

**The dump slows the step itself a little.** In the `plain` way, `add` ran at 57.7 GB/s at 16,000 (median of three rounds), against 63.1 to 63.4 in the other two. Its slowest pass at 18,000 ran at 25.2 GB/s. `default` at 16,000 had one pass at 117.4 GB/s, against 131 to 133 otherwise. The other cells' medians agree within 3% across the ways. One `noindex` pass of `add` at 18,000 also ran slow, at 39.3 GB/s, so not every dip is the dump's. The probe's own timings run only 5 warm calls, so these are signs, not measurements.

## What changed

`precision/probes/mac_float64/run.sh` now writes the dump to `xla-dump.noindex/`. The probe's earlier runs' folders, which name `xla-dump/`, are unchanged.
