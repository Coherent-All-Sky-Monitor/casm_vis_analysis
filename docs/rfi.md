# RFI Masking

`casm_vis_analysis.rfi` provides the `RFIMask` class and `apply_rfi_mask` function. The mask is always user-provided; there are no built-in default masks applied automatically.

## Convention

Throughout the codebase, `freq_mask` stored on a data dict uses **True = flagged (bad)**. This is the opposite of what plotting functions and `fit_delay` expect as input (they want True = good). `apply_rfi_mask` follows the True-flagged convention. `fringe_stop()` inverts this when populating `fs["freq_mask"]` so that `fs["freq_mask"]` is True = good, matching the downstream caller convention.

When in doubt, check which convention a specific parameter expects. For `fringe_stop(rfi_mask=...)`, pass either an `RFIMask` object or a bool array where True = good (or None to skip). For `apply_rfi_mask`, the stored result has True = bad.

## `RFIMask`

```python
from casm_vis_analysis.rfi import RFIMask

# From explicit frequency ranges (lo, hi in MHz, inclusive)
mask = RFIMask(
    bad_ranges_mhz=[(375, 390), (450, 452.5), (460, 468)],
    label="my_obs_2026-05-16",
)

# From the static config shipped in the package (versioned JSON files)
mask = RFIMask.from_static()            # latest version
mask = RFIMask.from_static(version=2)  # specific version

# From a JSON file
mask = RFIMask.from_json("/path/to/rfi_config.json")

# Evaluate the mask on a frequency axis
bad_channels = mask.flag_bins(freq_mhz)   # (F,) bool, True = contaminated
good_channels = mask(freq_mhz)            # (F,) bool, True = clean
```

`bad_ranges_mhz` entries can be either `(lo, hi)` tuples or `{"lo": lo, "hi": hi}` dicts. Reversed ranges (hi < lo) raise `ValueError` at construction time.

## `apply_rfi_mask`

Attaches flags to a visibility data dict in place. The visibility values are not modified; only the mask fields are written.

```python
from casm_vis_analysis.rfi import RFIMask, apply_rfi_mask

mask = RFIMask.from_static()
apply_rfi_mask(data, static=mask)

# With a dynamic (per-observation) mask in addition
dynamic = RFIMask(bad_ranges_mhz=[(410, 415)])
apply_rfi_mask(data, static=mask, dynamic=dynamic)
```

`data` must expose a `freq_mhz` key or attribute. After the call:

| Key added to `data` | Content |
|---|---|
| `data["freq_mask"]` | `(F,)` bool, True = flagged (OR of static and dynamic) |
| `data["freq_mask_static"]` | Static component, or None if not given |
| `data["freq_mask_dynamic"]` | Dynamic component, or None if not given |

Passing only `static=mask` leaves `freq_mask_dynamic = None` in the dict.

## Propagation through the pipeline

1. Call `apply_rfi_mask(data, ...)` on the raw `VisibilityResult` before passing it to `fringe_stop`.
2. `fringe_stop` picks up `data["freq_mask"]` and stores it (inverted to True = good) as `fs["freq_mask"]`.
3. `fit_delay`, `plot_phase_vs_freq`, `svd_calibrate`, and `beam_power_vs_time` all read `fs["freq_mask"]` (True = good) to skip contaminated channels.

Channels are never NaN-filled in `vis_stopped` by the mask. Fill at plot time if needed:

```python
vis_diag = fs["vis_stopped"].copy()
vis_diag[:, ~fs["freq_mask"], :] = np.nan + 1j * np.nan
```

## Static RFI config JSON schema

```json
{
  "site": "OVRO",
  "version": 2,
  "bands_mhz": [
    {"lo": 375.0, "hi": 390.0},
    {"lo": 450.0, "hi": 452.5}
  ]
}
```

JSON files are kept under `src/casm_vis_analysis/configs/rfi_static_v{N}.json`. `from_static()` with no argument picks the highest version number on disk.
