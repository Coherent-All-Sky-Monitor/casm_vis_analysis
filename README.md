# casm_vis_analysis

Fringe-stopping, delay correction, and diagnostic plotting for CASM correlator visibilities.

## Install

```bash
pip install -e .
```

You need `casm_io` installed first — get it from its own repo.

For running tests, install with dev extras: `pip install -e ".[dev]"`

## What you need

- Visibility `.dat` files on disk (e.g. `/data/casm/visibilities_64ant`)
- Antenna layout CSV (for `casm_io` `AntennaMapping`)
- Format name (`layout_32ant` or `layout_64ant`) or path to a format JSON
- Observation ID — UTC base timestamp, e.g. `2026-02-14-16:12:49`

## CLI

### `casm-autocorr`

Autocorrelation power spectra grouped by SNAP board.

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

One PNG per SNAP board in `output/<obs>/autocorr/`.

### `casm-waterfall`

Upper-triangle waterfall matrix (diagonal = power, upper = phase).

```bash
casm-waterfall \
  --data-dir /data/casm/visibilities_64ant \
  --obs 2026-02-14-16:12:49 \
  --format layout_64ant \
  --layout antenna_layout.csv \
  --output-dir ./output \
  --split-max 16
```

Splits into figures of at most `--split-max` antennas. Output in `output/<obs>/waterfall/`.

### `casm-fringe-stop`

Fringe-stop visibilities, optionally fit and remove delays, produce diagnostic plots.

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

Other useful flags:
- `--rfi-mask mask.npz` — boolean channel mask (NPZ key `mask`)
- `--time-start` / `--time-end` — restrict time range

Output in `output/<obs>/fringe_stop/` (diagnostic waterfalls, phase-vs-freq plot, optional NPZ).

## Python API

### Load data and extract autocorrelations

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

### Fringe-stop + delay correction

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
vis_fs = fs["vis_for_calibration"]  # this is what casm_calibration consumes

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

## Plotting

You can call plotting functions directly:

```python
from casm_vis_analysis.plotting.autocorr import plot_autocorr
from casm_vis_analysis.plotting.waterfall import plot_waterfall
from casm_vis_analysis.plotting.fringe_diag import plot_fringe_diagnostic
from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq

plot_autocorr(vis, freq_mhz, antenna_labels, output_path="autocorr.png", ncols=4)

plot_waterfall(vis, freq_mhz, time_unix, nsig, antenna_labels=labels,
               split_max=16, output_dir="./waterfalls")

plot_fringe_diagnostic(panels, time_unix, freq_mhz, target_labels,
                       target_snaps, ref_snap, output_dir="./diag")

plot_phase_vs_freq(panels, freq_mhz, baseline_labels=labels,
                   output_path="phase_vs_freq.png")
```

All return a matplotlib `Figure`. Pass an output path to save to disk.

## Output layout

`make_output_dir` creates:

```
output/<obs>/
├── autocorr/
├── waterfall/
└── fringe_stop/
```

## Good to know

- **Freq order**: descending (native) by default, `--freq-order ascending` to flip
- **Fringe-stop sign**: `-1` removes geometric phase (default), `+1` adds it
- **Delay models**: `linear` (slope fit) and `per_freq_phasor` (per-channel phase). Stack them with `--delay-model linear per_freq_phasor`
- **RFI mask**: always user-provided via `--rfi-mask`, no built-in defaults
- **`vis_for_calibration`**: fringe-stopped but NOT delay-corrected — that's what `casm_calibration` expects

## Tests

```bash
pytest tests/ -v
```
