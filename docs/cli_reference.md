# CLI Reference

All entry points are defined in `pyproject.toml` under `[project.scripts]` and wrap `runners.py` orchestration via `cli.py`.

## Common flags

All three primary commands (`casm-autocorr`, `casm-waterfall`, `casm-fringe-stop`) accept:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | path | — | Directory containing one observation's binary files |
| `--obs` | str | — | Observation start timestamp `YYYY-MM-DD-HH:MM:SS`; omit to use time-range mode |
| `--data-root` | path | `/mnt` | Root to scan for `visibilities_*` dirs when `--obs` is omitted |
| `--format` | str or path | — | Format name (`layout_64ant`, `layout_32ant`) or path to JSON |
| `--layout` | path | — | `AntennaMapping`-compatible CSV from `casm-layout apply` (or the legacy `casm-build-layout`) |
| `--output-dir` | path | `./output` | Directory for saved figures and NPZ files |
| `--time-start` | str | — | Window start `"YYYY-MM-DD HH:MM:SS"` |
| `--time-end` | str | — | Window end `"YYYY-MM-DD HH:MM:SS"` |
| `--time-tz` | str | `UTC` | IANA timezone for `--time-start`/`--time-end` |
| `--freq-order` | str | `descending` | Frequency axis order: `descending` (native) or `ascending` |
| `--nfiles` | int | — | Limit the read to this many files |
| `--skip-nfiles` | int | 0 | Skip this many files before reading |
| `--show` | flag | off | Render plots inline (Jupyter); skip disk save |
| `--include-inactive` | flag | off | Include non-functional ADCs in plots |

When `--obs` is omitted, `--time-start`, `--time-end`, and (optionally) `--data-root` trigger auto-discovery mode. `casm_io.correlator.read_visibilities` scans for every observation overlapping the range, stitches them, and warns on gaps.

## casm-autocorr

Per-SNAP autocorrelation power spectra. One panel per ADC, grouped by SNAP.

```bash
casm-autocorr \
  --format layout_64ant \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --time-start '2026-05-06 06:00:00' \
  --time-end   '2026-05-06 10:00:00' \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt \
  --output-dir ./output
```

Extra flags:

| Flag | Default | Description |
|---|---|---|
| `--scale` | `dB` | Y-axis scale: `dB` or `linear` |
| `--ncols` | `4` | Columns in the panel grid |

## casm-waterfall

Upper-triangle waterfall matrix. Diagonal shows autocorrelation power; upper triangle shows cross-correlation phase.

```bash
casm-waterfall \
  --format layout_64ant \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --time-start '2026-05-06 06:00:00' \
  --time-end   '2026-05-06 10:00:00' \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt \
  --output-dir ./output
```

Extra flags:

| Flag | Default | Description |
|---|---|---|
| `--split-max` | `16` | Maximum antennas per figure (splits into multiple figures) |

## casm-fringe-stop

Fringe-stop toward a named source, optionally fit and correct delays, produce diagnostic plots.

```bash
casm-fringe-stop \
  --format layout_64ant \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --time-start '2026-05-06 06:00:00' \
  --time-end   '2026-05-06 10:00:00' \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt \
  --ref-ant 10 \
  --source sun \
  --sign -1 \
  --delay-model linear \
  --antenna-delays \
  --save-npz \
  --output-dir ./output
```

Extra flags:

| Flag | Default | Description |
|---|---|---|
| `--ref-ant` | required | Antenna ID to use as the phase reference |
| `--source` | required | Source name: `sun`, `cas-a`, `cyg-a`, `tau-a`, `b0329-54` |
| `--sign` | `-1` | Fringe-stop sign convention; **always -1 for CASM data** |
| `--min-alt` | `10.0` | Minimum source altitude in degrees for the transit time mask |
| `--delay-model` | — | One or more model names: `linear` (recommended), `per_freq_phasor` |
| `--antenna-delays` | off | Decompose baseline delays into per-antenna contributions |
| `--save-npz` | off | Save fringe-stopped visibilities to NPZ |
| `--rfi-mask` | — | Path to an RFI mask file |

To use a single observation instead of a time range:

```bash
casm-fringe-stop \
  --obs 2026-05-06-06:00:00 \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --format layout_64ant \
  ...
```

## casm-viz-data-span

Survey a directory and print observation start times, durations, and file counts.

```bash
casm-viz-data-span \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --format layout_64ant
```

An observation ID is its start timestamp. An observation may run many hours; check this before assuming a window is not on disk.

## casm-fit-positions

Fit antenna ENU positions by minimising circular variance of fringe-stop coherence vs position offset. Used to correct position entries in the layout CSV.

```bash
casm-fit-positions \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --format layout_64ant \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --ref-ant 10 \
  --source sun \
  --sign -1 \
  --metric circvar \
  --axis x \
  --x-range '-4,4' \
  --x-step 0.05 \
  --time-start '2026-03-21 10:00:00' \
  --time-end   '2026-03-21 15:00:00' \
  --time-tz    US/Pacific \
  --output-dir ./output \
  --output-layout corrected_layout.csv
```

Extra flags:

| Flag | Default | Description |
|---|---|---|
| `--metric` | `circvar` | Optimisation metric: `circvar` (circular variance) |
| `--axis` | `x` | Position axis to scan: `x`, `y`, `z` |
| `--x-range` | `-4,4` | Scan range in metres as `lo,hi` |
| `--x-step` | `0.05` | Scan step in metres |
| `--cross-plank` | off | Fit antennas across the plank (E-W) separately |
| `--output-layout` | — | Write corrected layout CSV to this path |

## casm-validate-bf-weights

Round-trip a deployed int8 weights file: beamform at each deployed pointing using the same in-memory visibilities and cal that built the file, predict which sources cross which beams, report per-beam pass/fail.

```bash
casm-validate-bf-weights /tmp/weights_int8.h5 \
  --cal-h5 /tmp/cal_demo.h5 \
  --time-start "2026-05-08 11:30" \
  --time-end   "2026-05-08 14:30" \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt \
  --format     layout_64ant \
  --layout     ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --sources    sun cas-a cyg-a tau-a \
  --freq-band  405 433 \
  --fwhm-factor 1.0 \
  --pass-ratio  5.0 \
  --output     /tmp/bf_validation.png
```

Extra flags:

| Flag | Default | Description |
|---|---|---|
| `--cal-h5` | required | HDF5 calibration file from `bf_weights_generator` |
| `--sources` | `sun cas-a cyg-a tau-a` | Space-separated source names to test |
| `--freq-band` | `405 433` | Two values: `lo hi` MHz for the power freq-mean |
| `--fwhm-factor` | `1.0` | In-beam radius in units of half-FWHM (recommended: 1.0) |
| `--pass-ratio` | `5.0` | Peak/noise-floor threshold to call a beam passing (recommended: 5.0) |
| `--output` | — | Save the validation figure to this path |

## casm-layout

The primary interface to the antenna-layout pipeline. Wraps the two
lower-level stages below (`casm-sync-wiring`, `casm-build-layout`) behind
three verbs. Unlike the legacy commands, every invocation pulls the latest
CAsMan database snapshot from GitHub releases first (checksum-skip if
already current; falls back to the local copy with a loud warning if
offline). Geographic coordinates come from the surveyed positions CSV;
CAsMan decides which antenna occupies which grid cell and how it is wired.
The diff is reported per `(snap, adc)`.

```bash
casm-layout status   # pull CAsMan, print a one-line diff summary (read-only)
casm-layout diff     # pull CAsMan, print the full position + wiring-row diff (read-only)
casm-layout apply    # show the diff, confirm, then rewrite casm_wiring.csv + the dated layout CSV
```

`status` prints a one-line summary, e.g.:

```
8 antenna positions differ (4 removed, 4 enabled); 3 wiring-metadata changes
run 'casm-layout diff' for details
```

`diff` additionally lists, per `(snap, adc)`: `ADDED` / `REMOVED` / `MOVED` /
`ENABLED` / `DISABLED` / `CHANGED` position-level sections, then a
wiring-row detail section (added/removed/changed CAsMan rows).

`apply` shows the diff, asks `Apply these changes? [y/N]` (skip with
`-y`/`--yes`), rewrites `casm_wiring.csv` (`.bak` backup; aborts if CAsMan
returns fewer than 5 chassis-1 P1 antennas unless `--force` is passed),
rebuilds the dated `casm_antenna_layout_YYYY-MM-DD.csv`, and atomically
repoints the `current` symlink at it.

| Flag | Description |
|---|---|
| `--offline` | Skip the GitHub pull; use the local CAsMan copy (may be stale — no freshness check). |
| `--positions` | Positions CSV (default: `antenna_layout_april_ovro.csv`) |
| `--overrides` | Wiring overrides CSV (default: `casm_wiring_overrides.csv`) |
| `--snap-map` | Trusted `(chassis, slot) -> (feng_id, snap_ip)` map (default: `casm_snap_map.csv`) |
| `--wiring` | Wiring CSV (default: `casm_wiring.csv`) |
| `--layout-dir` | Directory holding the layout CSVs + `current` symlink (default: `/home/casm/software/dev/antenna_layouts`) |
| `-y`, `--yes` | (`apply` only) Skip the interactive confirmation prompt. |
| `--force` | (`apply` only) Bypass the <5-antenna sanity guard. |

Conflict policy: **CAsMan wins**. Hand-edits to `casm_wiring.csv` do not
survive `apply`. Encode them as override rows in
`casm_wiring_overrides.csv` (`add`, `disable`, or `replace`, keyed by
`(chassis, slot, adc)`). Default overrides path:
`/home/casm/software/dev/antenna_layouts/casm_wiring_overrides.csv`.

Python API: `layout.casman_pull.pull_casman`,
`layout.sync.build_wiring_candidate`, `layout.build.build_layout_dataframe`,
`layout.diff.{resolve_current_layout, diff_layouts, summarize_diff}`. Also
runnable via the in-process dispatcher: `run("casm-layout diff")`.

## casm-sync-wiring (legacy)

Lower-level stage behind `casm-layout`: pull CAsMan and regenerate
`casm_wiring.csv` directly. Prints a tip pointing at `casm-layout` on every
run. Kept for scripted/advanced use.

```bash
casm-sync-wiring           # dry-run: show diff vs current wiring CSV
casm-sync-wiring --apply   # write new CSV (backs up .bak first)
casm-sync-wiring --pull    # force fresh GitHub pull of CAsMan DB before sync
casm-sync-wiring --force   # bypass the minimum-antenna sanity check
```

Abort condition: CAsMan returns fewer than 5 chassis-1 P1 antennas. Pass `--force` to proceed anyway.

Conflict policy: **CAsMan wins**. Hand-edits to `casm_wiring.csv` do not survive `--apply`. Encode them as override rows in `casm_wiring_overrides.csv` (`add`, `disable`, or `replace`, keyed by `(chassis, slot, adc)`). Default overrides path: `/home/casm/software/dev/antenna_layouts/casm_wiring_overrides.csv`.

## casm-build-layout (legacy)

Lower-level stage behind `casm-layout`: build the `AntennaMapping`-compatible
consumer CSV from `casm_wiring.csv` plus hand-measured ENU positions
directly. Prints a tip pointing at `casm-layout` on every run. Kept for
scripted/advanced use.

```bash
casm-build-layout                  # rebuild using current wiring CSV
casm-build-layout --check-casman   # rebuild + show diff against CAsMan DB
```

The output CSV is the file you pass as `--layout` to the three primary CLIs.
