# casm_vis_analysis

Fringe-stopping, delay correction, off-source visibility tools, beamformer validation, and diagnostic plotting for CASM correlator visibilities.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_vis_analysis
pip install -e .
```

Requires `casm_io` (v0.2.0+). The layout pipeline additionally requires network access to CAsMan.

## Primary example

```python
from casm_io.correlator import read_visibilities, load_format, AntennaMapping
from casm_vis_analysis.fringe_stop import fringe_stop
from casm_vis_analysis.beam_power import beam_power_vs_time

fmt = load_format("layout_64ant")
# No path -> the canonical $CASM_LAYOUT_DIR/current layout.
ant = AntennaMapping.load().with_inactive([3])

data = read_visibilities(
    time_start="2026-05-16 11:30:00",
    time_end="2026-05-16 14:30:00",
    time_tz="America/Los_Angeles",
    data_root="/mnt",
    fmt=fmt,
)

# sign=-1 is the CASM convention — never flip it
fs = fringe_stop(data, ant, ref_ant=10, source="sun", sign=-1)

# fs["vis_for_calibration"] is what casm_calibrator.svd_calibrate consumes
# fs["time_mask"] selects samples when the Sun is above min_alt_deg
```

## Python API

| Function | Purpose |
|---|---|
| `fringe_stop(data, ant, *, ref_ant, source, sign=-1)` | Fringe-stop toward a source, returns `FringeStoppedData` dict |
| `beam_power_vs_time(data, ant, sources, *, cal_weights)` | Coherent cross-baseline beam power vs time |
| `build_static_visibility(date, *, fmt, data_root)` | Find quiet window, read it, return static-vis estimate |
| `find_quiet_windows(time_unix, *, altitude_caps)` | Locate intervals where all named sources are below their caps |
| `subtract_static_visibility(data, static_vis)` | Subtract static floor; returns a new data dict |
| `save_static_visibility(path, static_vis, *, freq_mhz)` | Persist static estimate to NPZ |
| `load_static_visibility(path)` | Reload saved static estimate |
| `validate_source(int8_h5, data, ant, *, source, cal_weights)` | Per-source beam-transit pass/fail |
| `validate_source_at_time(int8_h5, cal_weights, *, source, time_start, time_end)` | Same, but reads fresh visibilities internally |
| `validate_beam_weights(int8_h5, data, ant, *, cal_weights)` | Multi-source orchestrator |
| `plot_source_validation(result)` | Zenithal projection + per-beam power timeseries for `validate_source` output |
| `print_source_validation_summary(result)` | Text summary: beam direction, transit times, PASS/FAIL per beam |
| `RFIMask(bad_ranges_mhz)` | Frequency mask from contaminated MHz ranges; use `from_static()` for the shipped config |
| `apply_rfi_mask(data, static)` | Attach `freq_mask` to a data dict in place |
| `plot_phase_vs_freq(panels, freq_mhz)` | Phase vs frequency diagnostic; returns `list[Figure]` |

Runner functions (match the CLIs exactly):

```python
from casm_vis_analysis import run_autocorr, run_waterfall, run_fringe_stop

fs = run_fringe_stop(
    format="layout_64ant",
    layout="/home/casm/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv",
    time_start="2026-05-06 06:00:00",
    time_end="2026-05-06 10:00:00",
    time_tz="America/Los_Angeles",
    data_root="/mnt",
    ref_ant=10,
    source="sun",
    sign=-1,              # CASM convention
    delay_model=["linear"],
    show=True,            # render inline in Jupyter
)
# Returns: vis, vis_stopped, freq_mhz, time_unix, time_mask, tau_s,
#          target_aids, target_labels, delay_fits, figures
```

## CLI tools

Primary tools:

| Command | Purpose |
|---|---|
| `casm-autocorr` | Per-SNAP autocorrelation power spectra |
| `casm-waterfall` | Upper-triangle waterfall matrix |
| `casm-fringe-stop` | Fringe-stop + optional delay correction + diagnostics |

Supporting tools:

| Command | Purpose |
|---|---|
| `casm-viz-data-span` | Survey a data directory and list observation time ranges |
| `casm-fit-positions` | Solar fringe-stop antenna position fits |
| `casm-validate-bf-weights` | Validate a deployed SNAP int8 weights file |
| `casm-layout` | Antenna-layout pipeline front door: `status` / `diff` / `apply` (see below) |
| `casm-sync-wiring` | *Legacy* — pull CAsMan wiring and regenerate `casm_wiring.csv` |
| `casm-build-layout` | *Legacy* — build the `AntennaMapping`-compatible consumer CSV |

All three primary CLIs share: `--data-dir`, `--obs`, `--format`, `--layout`, `--output-dir`, `--freq-order`, `--time-start`, `--time-end`, `--time-tz`, `--nfiles`, `--skip-nfiles`, `--show`, `--data-root`.

`casm-fringe-stop` additionally accepts: `--ref-ant`, `--source`, `--sign`, `--min-alt`, `--save-npz`, `--rfi-mask`, `--delay-model`, `--antenna-delays`.

```bash
casm-fringe-stop \
  --format layout_64ant \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --time-start '2026-05-06 06:00:00' \
  --time-end   '2026-05-06 10:00:00' \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt \
  --ref-ant 10 --source sun --sign -1 \
  --delay-model linear --antenna-delays \
  --save-npz \
  --output-dir ./output
```

See [docs/cli_reference.md](docs/cli_reference.md) for every flag and all commands.

## Data selection

Two modes, pick one:

| Mode | How |
|---|---|
| Single observation | `obs="YYYY-MM-DD-HH:MM:SS"` with optional `data_dir`; trim further with `time_start`/`time_end`/`nfiles` |
| Time range (auto-discovery) | `obs=None` + `time_start`/`time_end`; `data_root` scanned for `visibilities_*` dirs |

Check what is on disk with `casm-viz-data-span` before choosing a window. An observation ID is its start timestamp and may run many hours.

## Detailed documentation

- [docs/sources_and_transits.md](docs/sources_and_transits.md) — source catalog, ENU direction vectors, transit window detection
- [docs/fringe_stop.md](docs/fringe_stop.md) — sign convention, geometric delay, `FringeStoppedData`, `coherence_metric`, `auto_detect_sign`, `plot_phase_vs_freq`
- [docs/rfi.md](docs/rfi.md) — `RFIMask`, `apply_rfi_mask`, mask propagation through the pipeline
- [docs/delay.md](docs/delay.md) — delay fitting models, antenna decomposition, when to use each
- [docs/svd_calibration_application.md](docs/svd_calibration_application.md) — applying calibration weights via `beam_power_vs_time`
- [docs/beam_validation.md](docs/beam_validation.md) — `validate_source`, `validate_source_at_time`, `plot_source_validation`, `print_source_validation_summary`, `load_beams_from_int8` HDF5 schema
- [docs/offsource.md](docs/offsource.md) — quiet-window detection, static-vis builder/subtractor
- [docs/cli_reference.md](docs/cli_reference.md) — all CLI entry points and flags
- [docs/walkthroughs.md](docs/walkthroughs.md) — end-to-end tutorial notebooks

## Antenna layout pipeline

CAsMan (the assembly database, pulled from GitHub releases) plus the surveyed
positions CSV (`antenna_layout_april_ovro.csv`) combine into the consumer
layout CSV that `AntennaMapping.load` reads. CAsMan decides which antenna
occupies which grid cell (row/col) and how each feed is wired; the surveyed
CSV supplies the geographic coordinates. A feed is keyed by `(snap, adc)`,
and that's the level the diff is reported at.

`casm-layout` is the primary interface — three verbs, and every invocation
pulls the latest CAsMan snapshot from GitHub first (checksum-skip if already
current; `--offline` skips the network and falls back to the local copy with
a loud warning):

```bash
casm-layout status   # pull CAsMan, print a one-line diff summary (read-only)
casm-layout diff     # pull CAsMan, print the full position + wiring-row diff (read-only)
casm-layout apply    # show the diff, confirm, then regenerate casm_wiring.csv + the dated layout CSV
```

Example:

```
$ casm-layout status
CAsMan release: database-snapshot-20260720-191117 (already up to date)
snap map: 4 entries (casm_snap_map.csv)
overrides: 3 rows (casm_wiring_overrides.csv)
CAsMan candidate: 17 P1 rows in mapped slots
current layout: casm_antenna_layout_2026-07-01.csv

8 antenna positions differ (4 removed, 4 enabled); 3 wiring-metadata changes
run 'casm-layout diff' for details
```

`casm-layout diff` prints that same preamble, then a full listing —
position-level `ADDED` / `REMOVED` / `MOVED` / `ENABLED` / `DISABLED` /
`CHANGED` sections per `(snap, adc)`, followed by a wiring-row detail
section. `casm-layout apply` shows the diff, asks
`Apply these changes? [y/N]` (skip with `-y`/`--yes`), then rewrites
`casm_wiring.csv` (`.bak` backup; aborts under 5 chassis-1 antennas unless
`--force`), rebuilds the dated `casm_antenna_layout_YYYY-MM-DD.csv`, and
atomically repoints the `current` symlink — one command end-to-end.

Shared flags: `--offline`, `--positions`, `--overrides`, `--snap-map`,
`--wiring`, `--layout-dir`.

Conflict policy is unchanged: **CAsMan wins**. Hand-edits to
`casm_wiring.csv` do not survive `apply` — encode them as override rows in
`casm_wiring_overrides.csv` (`add` / `disable` / `replace`, keyed by
`(chassis, slot, adc)`).

See [docs/cli_reference.md](docs/cli_reference.md) for the full flag reference.

## Testing

```bash
pytest tests/ -v
```

35 tests, all passing, using synthetic fixtures. No real data required.
