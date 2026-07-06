# Applying SVD Calibration: beam_power_vs_time

`beam_power_vs_time` computes coherent cross-baseline beam power through time for a set of source-tracking or fixed pointings. Its primary use is verifying that a calibration solution phases the array up correctly before committing to a beamformed observation.

The function is in `casm_vis_analysis.beam_power`.

## What it computes

For each pointing direction, at each time sample:

```
P_cross(t, f) = 2 * Re( sum_{i<j} w_i(t,f) * conj(w_j(t,f)) * V_ij(t,f) )
```

where `w_k(t,f) = cal_k(f) * exp(sign * 2*pi*i * f * tau_k(t))`. The result is averaged over the frequency band and returned as a `(T,)` real array per source.

Auto-power is not included. It does not depend on pointing direction, so it contributes a constant floor that would mask the directional response. The returned series is the part that responds to where you're pointing.

## Signature

```python
from casm_vis_analysis.beam_power import beam_power_vs_time

result = beam_power_vs_time(
    data,                     # VisibilityResult or dict with vis, freq_mhz, time_unix
    ant,                      # AntennaMapping
    sources=["sun", "cas-a", "cyg-a", ("control", 30.0, 180.0)],
    cal_weights=cal,          # CalibrationWeights from bf_weights_generator
    freq_band_mhz=(405, 433), # restrict the freq-mean to this band
    sign=-1,                  # must match the sign used in fringe_stop
)
```

`sources` entries can be:
- A string name resolved via `source_altaz` (tracking, time-varying pointing)
- A tuple `(label, alt_deg, az_deg)` for a fixed pointing (control beam)

Mixing the two forms is allowed.

### Cal frequency alignment

Cal weights are assumed to align channel-for-channel with `data["freq_mhz"]` in descending order (the native CASM ordering). If the cal frequencies are ascending, the function flips them automatically. If the channel counts match but the frequency values do not (band-shift mismatch), the function raises `ValueError`. This check was added specifically to catch the class of error where a cal from one observation window is applied to data from a different configuration.

## Output dict

```python
result["time_unix"]             # (T,) float64
result["power"]["sun"]          # (T,) real — cross-baseline power toward Sun
result["power"]["cas-a"]        # (T,) real
result["alt_deg"]["sun"]        # (T,) — source altitude per time sample
result["az_deg"]["sun"]         # (T,)
result["freq_band_used_mhz"]    # (lo, hi) after RFI mask + band cut
result["n_chan_used"]           # int
```

## Typical workflow

```python
from casm_io.correlator import read_visibilities, load_format, AntennaMapping
from casm_vis_analysis.fringe_stop import fringe_stop
from casm_vis_analysis.beam_power import beam_power_vs_time, plot_beam_power

fmt = load_format("layout_64ant")
ant = AntennaMapping.load("/path/to/layout.csv").with_inactive([3])

data = read_visibilities(
    time_start="2026-05-16 11:30:00",
    time_end="2026-05-16 14:30:00",
    time_tz="America/Los_Angeles",
    data_root="/mnt",
    fmt=fmt,
)

# cal is a CalibrationWeights object from the SVD pipeline
result = beam_power_vs_time(
    data, ant,
    sources=["sun", "cas-a", "cyg-a", "tau-a"],
    cal_weights=cal,
    freq_band_mhz=(405, 433),
    sign=-1,
)

fig = plot_beam_power(result, time_tz="America/Los_Angeles")
```

## `plot_beam_power`

```python
from casm_vis_analysis.beam_power import plot_beam_power

fig = plot_beam_power(
    result,
    show_alt=True,        # add vertical markers at each source's transit
    time_tz="America/Los_Angeles",
    xlim_unix=None,       # (t0, t1) to zoom; None = full range
    output_path=None,     # save to file if given
)
```

Returns a Matplotlib Figure. When `output_path` is not None, saves at 140 dpi.

## Memory

The function builds a `(T, F_band, n_baselines_active)` complex64 array in memory. For a 21-antenna active set and 6 hours at 1093 channels, that is approximately 70 MB. Multiple sources iterate over this array without re-reading it.

## Integration with static-vis subtraction

Run `subtract_static_visibility` before calling `beam_power_vs_time` if you suspect correlated cross-talk or ground-pickup contamination. The subtraction removes the direction-independent floor, improving the contrast between the source beam and the off-source noise. See [offsource.md](offsource.md).
