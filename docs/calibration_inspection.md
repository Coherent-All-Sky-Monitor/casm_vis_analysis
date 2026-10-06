# Saved array calibration diagnostics

`casm_vis_analysis.calibration_inspection` applies an existing calibration to
recorded visibilities. It does not solve, generate beam weights, or contact
hardware. The monitor's canonical solve driver still owns calibration products.

## API

`save_inspection(data, mapping, cal, out_dir, source='sun', background=None,
reference_cal=None, subtract_static=True)` takes a compact visibility triangle
from `calibration_checks.read_selected`. Antenna IDs are distinct from compact
packet indices. It saves all unique cross-baselines and per-antenna spectra.

- Baselines are stored as `V(first antenna, second antenna)`, in increasing
  antenna-ID order, conjugating the stored packet pair when needed.
- Raw, static-subtracted, geometric model, fringe-stopped and SVD-corrected
  stages are available. Geometric model phase is the opposite of the applied
  fringe-stop correction. The display does not fit a straight-line delay.
- SVD correction uses unit phasors from the loaded applied weights:
  `V_ij_fs * w_i * conj(w_j)`. Flagged/zero calibration weights stay missing.
  Loaded calibration flags are **True = good**.
- Spectra keep every native channel: phase is the angle of a complex average,
  amplitude is mean magnitude. Curves average source integrations above 10°.
- Previews average all samples into at most 180 time by 256 frequency cells.
  Complex values and magnitudes are averaged separately. Cal correction precedes
  frequency averaging; selecting a display reference never re-solves phases.
- `antennas.npz`, `pair_A_B.npz`, and `inspection.json` are immutable diagnostic
  outputs. The application attaches frozen-layout and actual-read metadata.

## Amplitude experiments

For antenna i, channel f, the voltage amplitude is proportional to:

| Scheme | Formula |
| --- | --- |
| Uniform | `1` |
| Inv-var | `1 / N_i(f)` |
| Inv-std | `1 / sqrt(N_i(f))` |
| Matched | `s_i(f) / N_i(f)` |

`N_i` is the time median of the **raw** off-source auto-power, estimating voltage
variance, not time variance of auto-power. `s_i` is a coherent voltage-response
estimate from `log(abs(mean(V_ij_fs))) = log(s_i) + log(s_j)`, fitted by least
squares over all valid cross-baselines after static subtraction. At least three
antennas and a full-rank design are required. Autos do not enter that fit.
The response model assumes a dominant coherent source; diffuse sky, RFI and
extended-source structure can bias it. It is not an absolute flux calibration.

Each scheme is independently max-normalized across antennas per channel.
Missing/non-positive noise estimates stay unavailable. Matched is unavailable
without static subtraction. The phase-only solve and its rank-1 metric do not
change when a scheme is selected. The applied correction is
`a_i * exp(-1j * phi_i)`; geometric steering is separate.

The UI previews rounding real/imaginary components at int8 scale 127. This is
before geometric steering, not a prediction of every final beam's quantization
or the downstream detector's integer truncation. None of these experiments
modifies the saved phase calibration or an existing beam product.

## Independent checks

`render_history(datasets, calibrations, reference, out_dir, *, labels=None)` saves
one figure per common reference baseline. The upper panel compares each day's
fringe-stopped phase. Two lower panels show selected-solution and reference
residuals separately, with a shared date-only legend and large colour markers.
The same candidate applies to every day; no static template is reused across
days. Original native-channel means and residuals remain in the NPZ files.

`history_figure(freq, payload, days, reference_label, target_label, *,
candidate_label='Selected solution', reference_weights_label='Reference weights')`
can redraw those saved arrays without reading visibilities. It displays only
`candidate` and `reference` residual keys; historical alternate-subtraction
arrays remain in the source file. The caller supplies accurate calibration
labels; this library does not infer deployment state.

## Stationary transit review

From October 5, 2026, `stationary_check` and `compare_amplitude_transit` use
`calibration_transit` for common backgrounds, power scales and white plots.
The latter applies saved amplitude profiles to the same fixed pointing and
matched antenna/channel/time selection. Phase/calibration objects remain unchanged.

For non-solar targets, background samples lie beyond the configured ±90 minutes
by default and have geometric Sun altitude below 0° at the integration timestamp.
At least two integrations must survive on each side. Insufficient coverage
raises an error; daytime samples are never substituted. Sun self-checks use
outer windows and warn about remaining solar response. Neither rule screens
other sources or RFI. This background is separate from the solve's night template.
Altitude uses installed IERS data offline; no network update is attempted.

For voltage weights `w_i(f)`, the raw signed cross sum is
`2 Re sum(i<j) w_i conj(w_j) V_ij`, including fixed geometric steering.
The default **Weighted mean** divides each channel by
`D(f) = 2 sum(i<j) |w_i| |w_j| = (sum_i |w_i|)^2 - sum_i |w_i|^2`.
An arbitrary common voltage-weight scale cancels. Receiver gains and relative
antenna weighting remain; this is not a Jy conversion or a sensitivity ranking.
**Raw sum** retains the original scaling. Both displays remove the selected
background's time mean in every channel before taking the broadband mean.
The lower comparison is `(power - background mean) / std(background, ddof=1)`;
sky and interference contribute to that standard deviation.

Dynamic spectra use symmetric 99th-percentile absolute-power colour limits.
Colorbar extensions identify saturated tails; all values remain in the saved
arrays, power curve and statistics. Gaps between selected native channels stay
blank instead of extending neighbouring pixels across them. Each scheme gets a selectable waterfall
and power curve. Optional **Peak time vs frequency** has one panel: the strongest
three-integration-smoothed feature per admitted channel and the ideal
equal-amplitude array prediction. It does not identify a source or measure an
instrumental clock delay. There is no duplicate residual-time panel.

Schema 2 NPZs retain raw summed `spectrum` `(time, selected_frequency)` and add
`pair_weight_sum` `(selected_frequency,)`, `off_mask` and `sun_alt_deg` `(time,)`.
Calibration NPZ timing metrics and `light_curve` use the weighted mean; amplitude
NPZ `light_curve` retains the raw spectrum mean for compatibility. The rendered
curve is derived by `prepare_series`, not copied from legacy metrics.
For saved fixed-pointing products, changing the background only changes a
time-constant channel offset. Recentring cancels the old subtracted template to
saved precision, so legacy arrays can be reviewed without raw visibility reads.
The caller must recover the actual saved weight amplitudes before displaying
a legacy weighted mean; missing normalization allows only the raw-sum view.

Tests: `tests/test_calibration_transit.py`, `tests/test_calibration_checks.py`
and `tests/test_calibration_inspection.py`, using synthetic fixtures.
