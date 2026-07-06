# Beam Validation

`casm_vis_analysis.beam_validation` validates a deployed SNAP int8 weights file against in-memory visibility data. The central question is: does the cal+geometry combination produce real beams on the sky, not just a coherent sum toward the calibrator direction?

## When to run this

After generating an int8 weights file (via the `bf_weights_generator` pipeline), run validation before starting a pulsar fold. A passing result means the deployed pointings produce a measurable signal when a known source transits through them.

## Public API

```python
from casm_vis_analysis.beam_validation import (
    load_beams_from_int8,
    find_source_beam_transits,
    validate_source,           # one-source pass/fail
    validate_beam_weights,     # multi-source orchestrator
    plot_beam_validation,
    BeamHit,
)
```

## `validate_source`

The recommended starting point when you want a simple pass/fail for one source.

```python
result = validate_source(
    "/path/to/weights_int8.h5",  # deployed SNAP int8 weights file
    data,                         # VisibilityResult (vis, freq_mhz, time_unix)
    ant,                          # AntennaMapping
    source="sun",                 # one source name
    cal_weights=cal,              # CalibrationWeights used to build the int8 file
    freq_band_mhz=(405, 433),     # band for the power freq-mean
    n_control_beams=1,            # number of far-from-track control beams
    fwhm_factor=1.0,              # "in-beam" radius in units of half-FWHM (recommended)
    pass_ratio=5.0,               # peak/noise_floor threshold to call a beam passing (recommended)
)
```

### `fwhm_factor` and `pass_ratio`

| Parameter | Recommended value | Effect of increasing |
|---|---|---|
| `fwhm_factor` | **1.0** | Larger catches more grazing transits but may flag beams a source barely skims as "hit" |
| `pass_ratio` | **5.0** | Higher threshold rejects marginal beams; lower passes beams with weaker signal |

The FWHM used for the in-beam test is the diffraction-limited E-W and N-S FWHM computed from the active antenna positions stored in the int8 file. If your active set differs from what was embedded in the file, check `result["beams"]["fwhm_ew_deg"]`.

### Output dict

```python
result["source"]              # str
result["beams"]               # from load_beams_from_int8
result["hits"]                # list of BeamHit objects
result["hit_beam_idxs"]       # beam indices the source crosses
result["control_beam_idxs"]   # far-from-track beams for null reference
result["power"]               # dict[beam_idx -> ndarray(T,)]
result["per_beam_metrics"]    # dict[beam_idx -> dict with pass/fail]
result["time_unix"]           # (T,)
result["freq_band_used_mhz"]  # (lo, hi) after RFI mask
result["n_chan_used"]         # int
```

### Per-beam metrics (hit beams)

```python
m = result["per_beam_metrics"][beam_idx]
m["expected_hit"]          # True
m["peak_in_window"]        # max power during the in-beam interval
m["median_out_window"]     # median power outside the in-beam interval
m["ratio"]                 # peak_excursion / noise_floor_excursion
m["pass"]                  # bool: ratio >= pass_ratio
m["sources"]               # list of source names crossing this beam
```

## `validate_beam_weights`

Multi-source orchestrator. Runs geometric prediction for all sources simultaneously and selects the most prominent beam-transit combinations for beamforming.

```python
result = validate_beam_weights(
    "/path/to/weights_int8.h5",
    data,
    ant,
    cal_weights=cal,
    sources=("sun", "cas-a", "cyg-a", "tau-a"),  # default
    freq_band_mhz=(405.0, 433.0),
    max_hit_panels=12,     # cap on plotted source-hit beams
    n_control_beams=2,
    fwhm_factor=1.0,       # recommended
    pass_ratio=5.0,        # recommended
)
```

Hit beams are ranked by total in-beam dwell time. Control beams are chosen to be maximally distant from every source track over the entire data window (not just at peak altitude), to avoid beams that accidentally sit under a source sidelobe.

## `load_beams_from_int8`

Reads beam pointings and active-array geometry from the HDF5.

```python
beams = load_beams_from_int8("/path/to/weights_int8.h5")
beams["alt_deg"]                  # (n_beams,)
beams["az_deg"]                   # (n_beams,)
beams["names"]                    # list of n_beams beam name strings
beams["fwhm_ew_deg"]              # float
beams["fwhm_ns_deg"]              # float
beams["active_positions_enu"]     # (n_active, 3)
```

Raises `ValueError` if the file does not carry `format_type="int8_snap_weights"` or if its schema version is not `"1.0"` or `"2.0"`.

## `find_source_beam_transits`

Geometric-only prediction: which (beam, source) pairs cross within `fwhm_factor` half-FWHMs during the time window. Returns a list of `BeamHit` dataclass objects.

```python
hits = find_source_beam_transits(
    beams,          # from load_beams_from_int8
    ["sun", "cas-a"],
    data["time_unix"],
    fwhm_factor=1.0,
)

for h in hits:
    print(h.beam_idx, h.source, h.duration_min, h.min_dist_deg)
```

## `plot_beam_validation`

```python
fig = plot_beam_validation(result, output_path=None)
```

One subplot per selected beam. Per-source colored shaded regions show predicted in-beam intervals; vertical dashed line marks predicted transit peak; pass/fail badge in the corner. Returns a Matplotlib Figure.

## Workflow example

```python
from casm_io.correlator import read_visibilities, load_format, AntennaMapping
from casm_vis_analysis.beam_validation import validate_source, plot_beam_validation

fmt = load_format("layout_64ant")
ant = AntennaMapping.load("/path/to/layout.csv").with_inactive([3])

data = read_visibilities(
    time_start="2026-05-16 11:30:00",
    time_end="2026-05-16 14:30:00",
    time_tz="America/Los_Angeles",
    data_root="/mnt",
    fmt=fmt,
)

result = validate_source(
    "/tmp/weights_int8.h5",
    data, ant,
    source="sun",
    cal_weights=cal,
    fwhm_factor=1.0,   # recommended
    pass_ratio=5.0,    # recommended
)

fig = plot_beam_validation(result)
```

## `plot_source_validation`

Produces a side-by-side figure for `validate_source` output. Left column: zenithal projection showing the source track and which beams it crosses (highlighted) plus control beams (outlined). Right column: one panel per highlighted and control beam showing `power(t)`, the predicted in-beam window (shaded), and the predicted peak (dashed line).

```python
from casm_vis_analysis.beam_validation import plot_source_validation

fig = plot_source_validation(
    result,
    time_tz="America/Los_Angeles",   # recommended for OVRO operations
    output_path=None,                 # save to file if given
)
```

Returns a Matplotlib Figure.

## `print_source_validation_summary`

Prints a compact text summary to stdout: one line per beam the source crossed, with transit entry/exit/peak times, pointing direction, and PASS/FAIL tag.

```python
from casm_vis_analysis.beam_validation import print_source_validation_summary

print_source_validation_summary(result, time_tz="America/Los_Angeles")
```

Example output:
```
SUN crosses 3 beams during the data window:
  [PASS] beam  42  alt=75.3°  az=180.0°   in-beam 11:15-13:45  peak 12:30  ratio=  8.42
  [FAIL] beam  17  alt=60.1°  az=195.0°   in-beam 11:00-13:30  peak 12:15  ratio=  1.23
```

## `validate_source_at_time`

Standalone variant that reads fresh visibilities internally rather than requiring a pre-loaded `data` argument. Useful for cross-time tests such as applying a daytime Sun cal to a nighttime Cas A window without re-running the full fringe-stop pipeline.

```python
from casm_vis_analysis.beam_validation import validate_source_at_time

result = validate_source_at_time(
    "/path/to/weights_int8.h5",
    cal_weights=cal,             # CalibrationWeights or path to HDF5
    source="cas-a",
    time_start="2026-05-16 22:00",
    time_end="2026-05-17 02:00",
    time_tz="America/Los_Angeles",
    layout=None,                 # None: resolves via $CASM_LAYOUT_CSV
    inactive_antennas=(3,),
    data_root="/mnt",
    fmt="layout_64ant",
    freq_band_mhz=(405, 433),
    n_control_beams=1,
    rfi_mask_version=2,          # recommended
    fwhm_factor=1.0,             # recommended
    pass_ratio=5.0,              # recommended
)
```

The function reads visibilities via `casm_io.read_visibilities`, applies the static RFI mask at `rfi_mask_version`, and delegates to `validate_source`. No new primitives. The return dict is identical to `validate_source`.

## Notes

- The `ratio` metric in `per_beam_metrics` works on the *excursion above the per-beam baseline*, not on absolute power. Cross-baseline coherent sums carry a direction-dependent DC bias from the natural visibility's DC being rotated by the per-baseline phase. For faint sources (Cas A, Tau A) this bias dominates absolute power, so the metric compares `(peak_in - median_out)` against the out-of-window noise spread.
- The `casm-validate-bf-weights` CLI mirrors this API. See [cli_reference.md](cli_reference.md).
