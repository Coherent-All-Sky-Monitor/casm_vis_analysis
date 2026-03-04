# casm_vis_analysis

Fringe-stopping, delay correction, and diagnostic plotting for CASM correlator visibilities.

## Installation

```bash
pip install -e ".[dev]"
```

`casm_io` is a required dependency — install it separately from its own repo before using this package.

## Prerequisites

Before running any commands you need:

- **Visibility `.dat` files** on disk (e.g. `/data/casm/visibilities_64ant`)
- **Antenna layout CSV** compatible with `casm_io.correlator.AntennaMapping`
- **Format name** (`layout_32ant` or `layout_64ant`) or path to a format JSON
- **Observation ID** — UTC base timestamp (e.g. `2026-02-14-16:12:49`)

## Quick Start — CLI Commands

### `casm-autocorr`

Plot autocorrelation power spectra grouped by SNAP board.

```bash
casm-autocorr \
  --data-dir /data/casm/visibilities_64ant \
  --obs 2026-02-14-16:12:49 \
  --format layout_64ant \
  --layout antenna_layout.csv \
  --output-dir ./output \
  --freq-order descending \
  --ncols 4
```

Produces one PNG per SNAP board in `output/<obs>/autocorr/`.

### `casm-waterfall`

Plot upper-triangle waterfall matrix (diagonal = power, upper = phase).

```bash
casm-waterfall \
  --data-dir /data/casm/visibilities_64ant \
  --obs 2026-02-14-16:12:49 \
  --format layout_64ant \
  --layout antenna_layout.csv \
  --output-dir ./output \
  --split-max 16
```

Splits large arrays into figures of at most `--split-max` antennas. Output goes to `output/<obs>/waterfall/`.

### `casm-fringe-stop`

Fringe-stop visibilities, optionally fit and remove delays, and produce diagnostic plots.

```bash
casm-fringe-stop \
  --data-dir /data/casm/visibilities_64ant \
  --obs 2026-02-14-16:12:49 \
  --format layout_64ant \
  --layout antenna_layout.csv \
  --ref-ant 5 \
  --source sun \
  --sign -1 \
  --min-alt 10 \
  --output-dir ./output \
  --delay-model linear per_freq_phasor \
  --save-npz
```

Additional flags:
- `--rfi-mask mask.npz` — NPZ file with boolean array (key `mask`) to flag channels
- `--time-start` / `--time-end` — restrict time range during data loading

Output goes to `output/<obs>/fringe_stop/` (diagnostic waterfalls, phase-vs-freq plot, and optional NPZ).

## Python API

### Loading data and extracting autocorrelations

```python
from casm_io.correlator import load_format, VisibilityReader, AntennaMapping
from casm_io.correlator.baselines import triu_flat_index

fmt = load_format("layout_64ant")
ant = AntennaMapping.load("antenna_layout.csv")
reader = VisibilityReader("/data/casm/visibilities_64ant", "2026-02-14-16:12:49", fmt)
data = reader.read(freq_order="descending", verbose=True)

vis = data["vis"]            # (T, F, n_bl) complex64
freq_mhz = data["freq_mhz"]
time_unix = data["time_unix"]

# Extract autocorrelations
for aid in ant.active_antennas():
    pidx = ant.packet_index(aid)
    auto_idx = triu_flat_index(fmt.nsig, pidx, pidx)
    auto = vis[:, :, auto_idx]  # (T, F) complex — imag ≈ 0
```

### Fringe-stop + delay correction pipeline

```python
import numpy as np
from casm_vis_analysis.sources import source_enu, find_transit_window
from casm_vis_analysis.fringe_stop import compute_baselines_enu, geometric_delay, fringe_stop
from casm_vis_analysis.delay import (
    fit_delay, apply_delay, build_delay_design_matrix, solve_antenna_delays,
)

# ... load data with VisibilityReader (ref= and targets= args) ...

# Fringe-stop
s_enu = source_enu("sun", time_unix)               # (T, 3)
bl_enu = compute_baselines_enu(positions, ref_idx, target_idxs)  # (n_bl, 3)
tau_s = geometric_delay(s_enu, bl_enu)              # (T, n_bl)
fs = fringe_stop(vis, freq_mhz, tau_s, sign=-1)
vis_fs = fs["vis_for_calibration"]  # API contract for casm_calibration

# Transit window + delay fit
i_start, i_end = find_transit_window("sun", time_unix, min_alt_deg=10.0)
time_mask = np.zeros(len(time_unix), dtype=bool)
time_mask[i_start:i_end + 1] = True

params = fit_delay(vis_fs, freq_mhz, time_mask=time_mask, model="linear")
vis_corrected = apply_delay(vis_fs, freq_mhz, params, model="linear")

# Per-antenna delay decomposition
A = build_delay_design_matrix(n_ant, baseline_pairs)
ant_delays = solve_antenna_delays(params["delay_ns"], A, ref_ant_idx=ref_idx)
```

## Plotting API

Call plotting functions directly for custom figures:

```python
from casm_vis_analysis.plotting.autocorr import plot_autocorr
from casm_vis_analysis.plotting.waterfall import plot_waterfall
from casm_vis_analysis.plotting.fringe_diag import plot_fringe_diagnostic
from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq

# Autocorrelation spectra
plot_autocorr(vis, freq_mhz, antenna_labels, output_path="autocorr.png", ncols=4)

# Waterfall matrix
plot_waterfall(vis, freq_mhz, time_unix, nsig, antenna_labels=labels,
               split_max=16, output_dir="./waterfalls")

# Fringe-stop diagnostic (waterfall panels per baseline, grouped by SNAP pair)
plot_fringe_diagnostic(panels, time_unix, freq_mhz, target_labels,
                       target_snaps, ref_snap, output_dir="./diag")

# Phase vs frequency (time-averaged)
plot_phase_vs_freq(panels, freq_mhz, baseline_labels=labels,
                   output_path="phase_vs_freq.png")
```

All plot functions return a matplotlib `Figure` and optionally save to disk when an output path is provided.

## Output Structure

`make_output_dir` creates the following tree:

```
output/<obs>/
├── autocorr/
├── waterfall/
└── fringe_stop/
```

## Key Concepts

- **Frequency order** — descending (native) by default; use `--freq-order ascending` to flip.
- **Fringe-stop sign** — `-1` removes geometric phase (default), `+1` adds it.
- **Delay models** — `linear` (slope fit across frequency) and `per_freq_phasor` (per-channel phase correction). Apply multiple models sequentially via `--delay-model linear per_freq_phasor`.
- **RFI mask** — always user-provided via `--rfi-mask` (NPZ with key `mask`, boolean array). No built-in defaults.
- **API contract** — `vis_for_calibration` (fringe-stopped, NOT delay-corrected) is what `casm_calibration` consumes.

## Running Tests

```bash
pytest tests/ -v
```
