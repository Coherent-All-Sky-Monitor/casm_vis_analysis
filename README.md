# casm_vis_analysis

Fringe-stopping, delay correction, off-source visibility tools, beamformer validation, and diagnostic plotting for CASM correlator visibilities.

## Install

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_vis_analysis
pip install -e .
```

## Surfaces

The package exposes three CLIs plus a richer Python API. The CLIs are thin wrappers over `runners.py` — the same orchestration code drives both surfaces.

CLI entry points (`pyproject.toml [project.scripts]`):

- `casm-autocorr` → per-SNAP autocorrelation power spectra
- `casm-waterfall` → upper-triangle waterfall matrix
- `casm-fringe-stop` → fringe-stop + optional delay correction + diagnostics

Supporting CLIs:

- `casm-viz-data-span` — survey a directory and list observations
- `casm-fit-positions` — solar fringe-stopping antenna position fits
- `casm-validate-bf-weights` — validate a deployed SNAP int8 weights file
- `casm-sync-wiring`, `casm-build-layout` — antenna layout pipeline

The Python API also covers off-source baselines, beam power vs time, and per-source validation — see [Other Python APIs](#other-python-apis) below.

## Walkthrough notebooks

Two end-to-end tutorial notebooks live in `notebooks/` with their outputs committed as the verified reference:

- `notebooks/casm_calibration_beamforming_walkthrough.ipynb` — SVD calibration + beamformer validation
- `notebooks/casm_pulsar_search_walkthrough.ipynb` — beamformed visibilities → filterbank → prepfold

Use them as the canonical starting point before reaching for the lower-level APIs.

## Antenna layout pipeline

CASM antenna positions and SNAP-to-ADC wiring live in CAsMan. This package pulls from CAsMan, applies a small overrides file, and produces an `AntennaMapping`-compatible CSV that every other CLI consumes via `--layout`.


### Pull from CAsMan and rebuild

CLI:

```bash
casm-sync-wiring                 # dry-run; show diff vs current casm_wiring.csv
casm-sync-wiring --apply         # regenerate casm_wiring.csv (with .bak)
casm-sync-wiring --pull          # force fresh GitHub pull of the CAsMan DB
casm-build-layout --check-casman # rebuild the consumer CSV + diff against CAsMan
```

Python:

```python
from casm_vis_analysis import run_sync_wiring, run_build_layout

run_sync_wiring(dry_run=True)            # see diff
run_sync_wiring(dry_run=False)           # apply (writes .bak first)
run_build_layout(check_casman=True)      # build consumer CSV
# returns {'output_csv': Path, 'n_total': 48, 'n_active': 23, ...}
```


## Data selection: obs-mode vs time-range mode

Every reader accepts the dataset either way:

| Pass... | Behavior |
|---|---|
| `obs="YYYY-MM-DD-HH:MM:SS"` (with optional `data_dir`) | Open that single observation; trim with `time_start`/`time_end`/`nfiles`. |
| `obs=None` + `time_start`/`time_end` (with `data_dir` or `data_root`) | **Auto-discovery** via `casm_io.correlator.read_visibilities`: scans for every observation overlapping the range, stitches them, warns on gaps. |

CLI: omit `--obs` and pass `--time-start`/`--time-end` (and optionally `--data-root`).

## Autocorr / Waterfall / Fringe-stop

All examples use the time-range API for autodiscovery. `LAYOUT` below is whatever `casm-build-layout` last produced.

### Python API

```python
from casm_vis_analysis import run_autocorr, run_waterfall, run_fringe_stop

COMMON = dict(
    format     = '/home/casm/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json',
    layout     = '/home/casm/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv',
    time_start = '2026-05-06 06:00:00',
    time_end   = '2026-05-06 10:00:00',
    time_tz    = 'America/Los_Angeles',
    data_root  = '/mnt',     
    show       = True,                       # render inline (Jupyter); skip disk save
)

# Autocorrelation power spectra (one panel per SNAP)
auto = run_autocorr(**COMMON, scale='dB', ncols=3)

# Upper-triangle waterfall matrix
wf = run_waterfall(**COMMON, split_max=16)

# Fringe-stop on a source with optional linear delay fit
fs = run_fringe_stop(
    **COMMON,
    ref_ant=10,            # antenna ID at ENU origin (N21 E1 in the May-2026 layout)
    source='sun',          # also: 'cas-a', 'cyg-a', 'B0329+54'
    sign=-1,
    delay_model=['linear'],
    antenna_delays=True,   # decompose baseline delays per-antenna
)
# fs is a dict: vis, vis_stopped, freq_mhz, time_unix, time_mask,
# tau_s, target_aids, target_labels, delay_fits, figures
```

Helpful kwargs across all three runners:
- `include_inactive=True` — plot all 12 ADCs of every connected SNAP, not just `functional==1`. Default `False`.
- `nfiles=N`, `skip_nfiles=N` — bound the read by file count instead of time.
- `freq_order='ascending'` — flip from native descending.

### CLI

```bash
casm-autocorr \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --time-start '2026-05-06 06:00:00' \
  --time-end   '2026-05-06 10:00:00' \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt  \
  --output-dir ./output

casm-waterfall \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --time-start '2026-05-06 06:00:00' \
  --time-end   '2026-05-06 10:00:00' \
  --time-tz    America/Los_Angeles \
  --data-root  /mnt \
  --output-dir ./output

casm-fringe-stop \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
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

To use a single observation instead of a time range, swap `--time-start`/`--time-end`/`--data-root` for `--obs YYYY-MM-DD-HH:MM:SS --data-dir /mnt/.../visibilities_64ant/`.

## Other CLIs

### casm-viz-data-span — survey a directory

```bash
casm-viz-data-span \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json
```

### casm-fit-positions — solar fringe-stop position fits

```bash
casm-fit-positions \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv \
  --ref-ant 10 --cross-plank --source sun --sign -1 \
  --metric circvar --axis x \
  --x-range='-4,4' --x-step 0.05 \
  --time-start '2026-03-21 10:00:00' --time-end '2026-03-21 15:00:00' --time-tz US/Pacific \
  --output-dir ./output \
  --output-layout corrected_layout.csv
```

### casm-validate-bf-weights — round-trip a deployed int8 weights file

```bash
casm-validate-bf-weights /tmp/my_weights_int8.h5 \
  --cal-h5 /tmp/cal_demo.h5 \
  --time-start "2026-05-08 11:30" --time-end "2026-05-08 14:30" \
  --time-tz America/Los_Angeles \
  --sources sun cas-a cyg-a tau-a \
  --freq-band 405 433 \
  --output /tmp/bf_validation.png
```

Beamforms at every deployed pointing using the same in-memory visibilities + cal that built the int8 file, predicts which sources cross which beams during the window, and reports per-beam pass/fail.

## Other Python APIs

The compose-friendly Python API exposes more than the three primary CLIs:

```python
from casm_vis_analysis import (
    # off-source / static visibility
    build_static_visibility,           # (vis, freq_mhz, time_unix) -> static template
    find_quiet_windows,                # locate Sun/source-free LST intervals
    average_visibility,
    subtract_static_visibility,
    save_static_visibility, load_static_visibility,
    DEFAULT_ALTITUDE_CAPS_OVRO,
    # beamformer power vs time (cross-only, coherent)
    beam_power_vs_time, plot_beam_power,
    # deployed-int8 validation
    validate_source,                   # one-source pass/fail vs predicted transit
    validate_source_at_time,           # snapshot at a chosen time
    validate_beam_weights,             # multi-source orchestrator (CLI mirror)
    find_source_beam_transits,
    load_beams_from_int8,
    plot_source_validation, plot_beam_validation,
    print_source_validation_summary,
    # RFI
    RFIMask, apply_rfi_mask,
)
```

Typical workflow: load visibilities → `apply_rfi_mask` → `build_static_visibility` (sets the off-source floor) → `beam_power_vs_time` toward known sources → if you have a deployed int8 file, run `validate_source` for hard pass/fail. The walkthrough notebooks chain these end-to-end.

## Testing

```bash
pytest tests/ -v
```
