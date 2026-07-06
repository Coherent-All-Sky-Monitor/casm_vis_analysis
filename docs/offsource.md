# Off-Source Visibility Subtraction

`casm_vis_analysis.offsource` estimates a static (direction-independent) per-baseline visibility from a quiet sky window and subtracts it from the observation. This removes correlated cross-talk, common-mode RFI, ground-pickup pedestals, and similar contamination that does not fringe with hour angle.

## Why this matters

Fringe-stopped visibilities contain a direction-independent floor from:
- Electronics cross-talk between adjacent SNAP channels
- Ground-pickup reflected into baselines
- RFI that leaks past the mask but has a broadband correlated component

This floor does not fringe with source transit, so it does not cancel by time-averaging on the source. Subtracting a static estimate built from a genuinely quiet window removes it before beamforming.

## End-to-end orchestrator: `build_static_visibility`

For the common case where you want a static estimate for a given calendar date:

```python
from casm_io.correlator import load_format
from casm_vis_analysis.offsource import build_static_visibility

fmt = load_format("layout_64ant")

result = build_static_visibility(
    "2026-05-16",
    fmt=fmt,
    data_root="/mnt",
    time_tz="America/Los_Angeles",
    altitude_caps=None,    # defaults to DEFAULT_ALTITUDE_CAPS_OVRO
    min_duration_s=15*60,  # reject windows shorter than 15 min
    max_duration_s=60*60,  # trim windows longer than 60 min
    rfi_mask=None,         # RFIMask if you want flagged channels NaN'd
    verbose=True,
)

result["static_vis"]     # (F, n_bl) complex — the estimate
result["freq_mhz"]       # (F,)
result["window_unix"]    # (t_start, t_end) float
result["altitudes"]      # {source: (min, mean, max)} over window
```

This function scans a 1-minute cadence grid across the full date, finds the first qualifying quiet window, reads just that window from disk, optionally applies the RFI mask, and returns the time-averaged visibility.

## Default altitude caps

```python
from casm_vis_analysis.offsource import DEFAULT_ALTITUDE_CAPS_OVRO
# {'sun': 0.0, 'tau-a': 0.0, 'cyg-a': 20.0, 'cas-a': 15.0}
```

Cas A is circumpolar at OVRO (minimum altitude ~6°) and never fully sets, so requiring it below the horizon would never qualify. The defaults require Sun and Tau A below the horizon while allowing Cas A and Cyg A at low altitudes where the primary beam attenuates them significantly.

## Manual two-step workflow

When the quiet window is at night but your science target transits during the day (the usual pulsar observation scenario), read the quiet window once and cache it:

```python
from casm_io.correlator import read_visibilities
from casm_vis_analysis.offsource import (
    find_quiet_windows,
    average_visibility,
    save_static_visibility,
    load_static_visibility,
    subtract_static_visibility,
    plot_offsource_diagnostic,
)
from casm_vis_analysis.rfi import RFIMask, apply_rfi_mask

# --- One-off: build and cache the static estimate ---
quiet = read_visibilities(
    time_start="2026-05-09 22:29", time_end="2026-05-09 23:20",
    time_tz="America/Los_Angeles", data_root="/mnt", fmt=fmt,
)
apply_rfi_mask(quiet, RFIMask.from_static())

windows = find_quiet_windows(
    quiet["time_unix"],
    altitude_caps={"sun": 0, "tau-a": 0, "cyg-a": 20, "cas-a": 15},
    min_duration_s=15*60,
)
# windows[0] is the first qualifying interval

static_vis = average_visibility(quiet, time_mask=windows[0]["mask"])
# (F, n_bl) complex128

# Verify visually before persisting
fig = plot_offsource_diagnostic(quiet, static_vis, windows[0]["mask"])

save_static_visibility(
    "/path/static_2026-05-09.npz",
    static_vis,
    freq_mhz=quiet["freq_mhz"],
    window_unix=(windows[0]["t_start"], windows[0]["t_end"]),
    altitudes=windows[0]["altitudes"],
    notes="layout_may2026 + static_rfi_mask",
)
del quiet    # free the quiet-window read

# --- Later, on the science data ---
data = read_visibilities(
    time_start="2026-05-09 11:00", time_end="2026-05-09 15:00", ...
)
cached = load_static_visibility("/path/static_2026-05-09.npz")
data_clean = subtract_static_visibility(data, cached["static_vis"])
```

## Function reference

### `find_quiet_windows(time_unix, *, altitude_caps, min_duration_s=0)`

```
Returns : list of dict, one per qualifying window
  i_start, i_end    : slice indices into time_unix
  t_start, t_end    : float Unix seconds (inclusive)
  duration_s        : float
  mask              : bool (T,), True inside window
  altitudes         : {source: (min, mean, max)} over window
```

### `average_visibility(data, *, time_mask=None, time_range_unix=None)`

Exactly one of `time_mask` or `time_range_unix` must be given.

```
Returns : ndarray (F, n_bl) complex128
```

Flagged channels (from `data["freq_mask"]`) are set to NaN in the output by default. NaN channels are treated as "no estimate available" in `subtract_static_visibility` and are passed through unchanged.

### `subtract_static_visibility(data, static_vis)`

Returns a new dict (shallow copy of `data`) with `vis` replaced by `vis - static_vis`. Does not mutate `data`. Channels where `static_vis` is NaN are subtracted as zero (no-op).

### `save_static_visibility(path, static_vis, *, freq_mhz, window_unix=None, altitudes=None, notes="")`

Writes a compressed NPZ. The frequency axis is stored alongside the array so consumers can verify alignment. The `notes` field is free-form text (use it to record layout version and RFI mask version).

### `load_static_visibility(path)`

```
Returns : dict with static_vis, freq_mhz, notes (str),
          and optionally window_unix and altitudes
```

Pass `out["static_vis"]` directly into `subtract_static_visibility`.

## Notes

- The static estimate is computed on the full upper-triangle visibility, including autocorrelations. `subtract_static_visibility` subtracts it from the same-shaped array, so the subtracted data still contains autocorrelations — they will be reduced to near zero if the quiet-window autocorr matches the science data autocorr. This is usually fine; autocorrelations are excluded from beamforming anyway.
- Verify the quiet window with `plot_offsource_diagnostic` before caching. A contaminated window produces a static estimate that amplifies certain baselines rather than suppressing them.
- If the science data and quiet window were taken at different SNAP configurations or with a different set of active antennas, the `(F, n_bl)` shapes will not match and `subtract_static_visibility` will raise `ValueError`.
