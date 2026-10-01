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

`render_history(datasets, calibrations, reference, out_dir)` saves one readable
two-row figure per common reference baseline: fringe-stopped phase with fixed
calibration predictions, and residual phase. Every day uses the same candidate;
it does not fit away inter-day changes or silently reuse a static template.

`compare_amplitude_transit` uses saved amplitude profiles with a fixed transit
pointing and a common antenna/channel/time selection. It saves each scheme's
signed cross-power dynamic spectrum and a comparison of absolute power and
source excess / measured off-window RMS. The latter includes sky/RFI variation,
not just thermal noise. These plots do not establish a detection or a superior
calibration. Phase/calibration objects remain unchanged.
