# Fringe-Stopping

`casm_vis_analysis.fringe_stop` computes geometric delays from source direction and baseline ENU vectors, then applies phase corrections to stop fringes.

## Sign convention

The phase applied to each visibility is:

```
phase = sign * 2 * pi * freq_hz * tau_s
vis_stopped = vis * exp(1j * phase)
```

**Use `sign=-1` always.** This is the CASM convention. It cancels the natural baseline phase `exp(+2*pi*i * f * (b . s_hat) / c)` so that the cross-baseline sum adds coherently for the source direction. `sign=+1` would look for the source's anti-phase image. There is no reason to change this for CASM data.

## High-level API: `fringe_stop`

This is the compose-friendly wrapper. It accepts the dict returned by `read_visibilities`, performs all intermediate steps internally, and returns a fully populated result dict.

```python
from casm_vis_analysis.fringe_stop import fringe_stop

fs = fringe_stop(
    data,          # VisibilityResult or dict with vis, freq_mhz, time_unix
    ant,           # AntennaMapping (with_inactive overrides respected)
    ref_ant=10,    # antenna ID at the ENU origin
    source="sun",  # "sun", "cas_a", "cyg_a", "tau_a", "b0329_54"
    sign=-1,       # CASM convention; never flip
    min_alt_deg=10.0,  # source below this is excluded from time_mask
    rfi_mask=None,     # RFIMask, bool array, or None
)
```

### Output dict keys

| Key | Shape | Description |
|---|---|---|
| `vis` | (T, F, n_bl) | Raw visibilities, sliced to ref-target baselines |
| `vis_stopped` | (T, F, n_bl) | Fringe-stopped visibilities |
| `vis_for_calibration` | (T, F, n_bl) | Alias of `vis_stopped` — this is what `casm_calibrator.svd_calibrate` takes as input |
| `geometric_phase` | (T, F, n_bl) | Applied phase in radians |
| `tau_s` | (T, n_bl) | Geometric delay in seconds |
| `freq_mhz` | (F,) | Frequency axis |
| `time_unix` | (T,) | Unix timestamps |
| `time_mask` | (T,) bool | True where source is above `min_alt_deg` |
| `freq_mask` | (F,) bool | True = good channel (from `rfi_mask` or `data["freq_mask"]`) |
| `source` | str | Source name passed in |
| `ref_ant` | int | Reference antenna ID |
| `sign` | int | Sign used |
| `target_aids` | list[int] | Antenna IDs of the target antennas |
| `target_labels` | list[str] | Human-readable labels (AntID, SNAP, ADC) |

### Baseline slicing

`fringe_stop` handles two input shapes transparently:

- **Full upper-triangle** `(T, F, n*(n+1)/2)`: sliced internally to ref-target baselines via `triu_flat_index`. Conjugation is applied when the packet index ordering requires it.
- **Pre-filtered** `(T, F, n_targets)`: used as-is when the last axis exactly matches `len(active_antennas) - 1`.

If you pre-filtered with a different reference or target set, slice externally and call `fringe_stop_array` directly.

### RFI mask

Precedence for `freq_mask` in the output:

1. Explicit `rfi_mask=` kwarg (an `RFIMask` object, a bool array, or `None` to ignore)
2. `data["freq_mask"]` populated by `apply_rfi_mask()`
3. All-good (all True)

Flagged channels are **not** NaN-filled in `vis_stopped`. Downstream stages read `fs["freq_mask"]` and skip or zero flagged channels themselves. This avoids breaking `np.unwrap` and `np.polyfit` in diagnostic plotters. To see NaN gaps in a diagnostic plot, do it at plot time:

```python
vis_for_diag = fs["vis_stopped"].copy()
vis_for_diag[:, ~fs["freq_mask"], :] = np.nan + 1j * np.nan
```

## FringeStoppedData TypedDict

`fringe_stop()` returns a `FringeStoppedData` TypedDict. Dict-style access (`fs["key"]`) and attribute access both work, so the return is compatible with downstream functions that accept either style.

All keys are optional in the TypedDict declaration, but `fringe_stop()` always populates the full set listed in the Output dict keys table above.

## Low-level primitives

For cases where you need direct control over the geometry computation.

### `compute_baselines_enu(positions_enu, ref_idx, target_idxs)`

```
positions_enu : ndarray, shape (n_ant, 3) — ENU positions in metres
ref_idx       : int — row index of the reference antenna in positions_enu
target_idxs   : array-like of int — row indices of target antennas
Returns       : ndarray, shape (n_targets, 3) — target - ref in ENU metres
```

### `geometric_delay(source_enu, baseline_enu)`

```
source_enu   : ndarray, shape (T, 3) — unit direction vectors from source_enu()
baseline_enu : ndarray, shape (3,) or (n_bl, 3) — from compute_baselines_enu()
Returns      : ndarray, shape (T,) or (T, n_bl) — delay tau = (b . s) / c in seconds
```

### `fringe_stop_array(vis, freq_mhz, tau_s, sign=-1)`

Array-level fringe-stop without the `AntennaMapping` dependency.

```
vis      : ndarray, shape (T, F, n_bl)
freq_mhz : ndarray, shape (F,)
tau_s    : ndarray, shape (T,) or (T, n_bl)
Returns  : dict with vis_raw, vis_stopped, vis_for_calibration,
           geometric_phase, tau_s, sign, freq_mhz
```

`runners.py` calls `fringe_stop_array` explicitly. The high-level `fringe_stop()` wraps it.

### `fringe_stop_single_baseline(vis, freq_hz, tau_s, sign=-1)`

Single-baseline variant. Accepts `freq_hz` in Hz (not MHz). Returns a corrected `(T, F)` array directly rather than a dict. Used by `casm-bf-imaging` and `auto_detect_sign`.

```
vis      : ndarray, shape (T, F)
freq_hz  : ndarray, shape (F,) — note: Hz, not MHz
tau_s    : ndarray, shape (T,)
Returns  : ndarray, shape (T, F) — fringe-stopped
```

### `coherence_metric(vis, freq_mask=None)`

Scalar coherence: `|mean_f(exp(i * phase(vis)))|`. High when phases are aligned across frequency. Returns `(T,)` or `(T, n_bl)`.

```python
coh = coherence_metric(fs["vis_stopped"], freq_mask=fs["freq_mask"])
# coh shape: (T, n_bl) — one value per time and baseline
```

A value near 1.0 means the fringe-stop worked well at that time sample. Values near 0 indicate the source is below the horizon or the sign is wrong.

### `auto_detect_sign(vis, freq_mhz, tau_s, freq_mask=None)`

Try both signs and return the one that produces higher mean coherence. Use this only when the correct sign is genuinely uncertain; for standard CASM data, pass `sign=-1` directly. The auto-detect adds runtime (two full fringe-stops on the data) and can misfire on low-coherence baselines.

```
vis      : ndarray, shape (T, F)
freq_mhz : ndarray, shape (F,)
tau_s    : ndarray, shape (T,)
Returns  : int — +1 or -1
```

## Diagnostic: `plot_phase_vs_freq`

```python
from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq

figs = plot_phase_vs_freq(
    panels=[
        ("raw",    data["vis"][:, :, bl_idxs]),
        ("fs",     fs["vis_stopped"]),
        ("delay",  vis_delay_corrected),   # optional third panel
    ],
    freq_mhz=fs["freq_mhz"],
    baseline_labels=fs["target_labels"],
    unwrap=True,
    time_mask=fs["time_mask"],
    freq_mask=fs["freq_mask"],
    output_path=None,   # None = don't save
    split_max=8,        # recommended: 8 (caps figure height for inline rendering)
)
# figs is always a list of Figure, even when there is only one figure
```

`panels` is a list of `(label, vis_array)` tuples. Each `vis_array` can be `(T, F)` for a single baseline or `(T, F, n_bl)` for all baselines. Complex values are time-averaged (using `time_mask` when given) and then phase is extracted.

When `n_baselines > split_max`, the function produces multiple figures, one per chunk. The return is always `list[Figure]`. Jupyter inline rendering silently truncates very tall figures, so the default `split_max=8` keeps each figure at a manageable height (~18 inches).

When `output_path` is given and the input splits into multiple figures, `_partN` is appended before the suffix (e.g. `diag_part0.png`, `diag_part1.png`).

`freq_mask` (True = good) removes RFI-flagged channels before phase unwrapping. This prevents accumulated wrap errors at gaps from corrupting downstream channels. The removed channels appear as gaps in the plot rather than erroneous phase jumps.

## Calibration boundary

`vis_for_calibration` (alias of `vis_stopped`) is the **only** key consumed by SVD beamforming in `casm_calibration`. Delay-corrected arrays are diagnostic only. Do not pass delay-corrected visibilities into the calibrator.

## Example: manual geometry

```python
import numpy as np
from casm_vis_analysis.fringe_stop import (
    compute_baselines_enu,
    geometric_delay,
    fringe_stop_array,
)
from casm_vis_analysis.sources import source_enu

positions = ant.get_positions()          # (n_ant, 3)
ref_row = active_sorted.index(ref_ant)
target_rows = [active_sorted.index(a) for a in target_aids]

bl_enu = compute_baselines_enu(positions, ref_row, target_rows)  # (n_bl, 3)
s_enu = source_enu("sun", data["time_unix"])                     # (T, 3)
tau = geometric_delay(s_enu, bl_enu)                             # (T, n_bl)

fs = fringe_stop_array(data["vis"][:, :, target_bl_idxs], data["freq_mhz"], tau, sign=-1)
```
