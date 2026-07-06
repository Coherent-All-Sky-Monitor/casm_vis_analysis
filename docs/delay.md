# Delay Fitting and Correction

`casm_vis_analysis.delay` fits and corrects residual delays in fringe-stopped visibilities. This is a **diagnostic tool only**. Delay-corrected visibilities are not consumed by SVD beamforming, which operates on `vis_for_calibration` (fringe-stopped only) from `fringe_stop()`.

## Models

Two models are registered in `DELAY_MODELS`:

| Model | Recommended | Use case |
|---|---|---|
| `"linear"` | **Yes** | Normal delay fitting. Coherent delay-domain search; not affected by RFI gaps the way phase-unwrap methods are. |
| `"per_freq_phasor"` | Debugging only | Non-linear bandpass or multi-delay baselines. Corrects per-channel phase independently; cannot extrapolate to other frequencies or decompose into per-antenna delays. |

Use `"linear"` unless you have a baseline with clear non-linear instrumental phase across the band.

## Primary interface

```python
from casm_vis_analysis.delay import fit_delay, apply_delay, DELAY_MODELS

params = fit_delay(
    fs["vis_stopped"],     # (T, F, n_bl) fringe-stopped
    fs["freq_mhz"],        # (F,)
    time_mask=fs["time_mask"],   # True = use this sample
    freq_mask=fs["freq_mask"],   # True = good channel
    model="linear",              # recommended
    tau_max_ns=1000.0,           # search half-width in ns
    tau_step_ns=0.5,             # coarse grid step; refined by quadratic interp
    return_coherence=False,      # set True to get coherence_curves for plotting
)

vis_corrected = apply_delay(
    fs["vis_stopped"],
    fs["freq_mhz"],
    params,
    model="linear",
)
```

## `fit_delay` output (linear model)

| Key | Type | Description |
|---|---|---|
| `delay_ns` | float or ndarray(n_bl) | Fitted delay per baseline |
| `slope` | float or ndarray(n_bl) | Phase slope in rad/MHz |
| `intercept` | float or ndarray(n_bl) | Phase intercept in rad |
| `r_squared` | float or ndarray(n_bl) | Post-fit coherence^2; 1.0 = perfect alignment |
| `peak_to_secondary_ratio` | float or ndarray(n_bl) | Confidence in chosen delay; values near 1.0 mean ambiguous fit |
| `low_quality` | bool or ndarray(n_bl) | True if `r_squared < 0.7` or `peak_to_secondary_ratio < 2.0` |
| `tau_nyquist_ns` | float | Nyquist delay from channel spacing (same for all baselines) |
| `model` | str | `"linear"` |
| `coherence_curves` | ndarray(n_grid, n_bl) | Present only if `return_coherence=True` |
| `tau_grid_ns` | ndarray(n_grid) | Present only if `return_coherence=True` |

## How the linear fit works

The fit maximises `|sum_f vis_avg(f) * exp(-2*pi*i*tau*f)|` over a tau grid. Because it works in the delay domain rather than unwrapping phase, RFI gaps only reduce SNR; they cannot bias the fitted slope by a multiple of 2*pi as gap-crossing unwrap errors do. This matters for long cross-SNAP baselines where each 10 MHz RFI band can hide multiple phase wraps.

The coarse grid peak is refined by quadratic interpolation, giving sub-step precision without a denser grid.

The Nyquist limit for CASM (~30.5 kHz channels) is approximately 16 us, far beyond any cable delay, so there is no aliasing concern in practice. The search is bounded by `tau_max_ns` (default 1 us).

## Antenna delay decomposition

After fitting per-baseline delays you can decompose them into per-antenna contributions using a least-squares solve:

```python
from casm_vis_analysis.delay import (
    build_delay_design_matrix,
    solve_antenna_delays,
)

# baseline_pairs is a list of (i, j) antenna-position-array indices
A = build_delay_design_matrix(n_ant, baseline_pairs)

# weights = 1/r_squared or similar confidence metric
ant_delays = solve_antenna_delays(
    params["delay_ns"],     # (n_bl,)
    A,                      # (n_bl, n_ant)
    weights=None,           # optional per-baseline weights
    ref_ant_idx=0,          # reference antenna is pinned to delay=0
)
# Returns ndarray(n_ant,) — per-antenna delay relative to ref
```

The design matrix encodes `tau_ij = tau_j - tau_i` for each baseline. The reference antenna column is removed before solving; its delay is set to zero.

## Per-frequency phasor model

```python
from casm_vis_analysis.delay import compute_per_freq_phasor, apply_per_freq_phasor

phasor_phase = compute_per_freq_phasor(
    fs["vis_stopped"],
    time_mask=fs["time_mask"],
)  # shape (F,) or (F, n_bl)

vis_corrected = apply_per_freq_phasor(fs["vis_stopped"], phasor_phase)
```

This corrects per-channel phase independently. It captures non-linear instrumental phase but cannot be decomposed into per-antenna delays and cannot be applied to data at a different frequency resolution. Use it to diagnose whether a linear model is failing before drawing conclusions about cable lengths.

## Diagnostic plots

```python
from casm_vis_analysis.plotting.delay_diag import (
    plot_coherence_curves,
    plot_delay_vs_baseline_length,
)

# Requires return_coherence=True in fit_delay
params = fit_delay(..., return_coherence=True)

fig1 = plot_coherence_curves(params, target_labels=fs["target_labels"])
# One panel per baseline; red border = low_quality

fig2 = plot_delay_vs_baseline_length(params, bl_lengths_m, target_labels=fs["target_labels"])
# Scatter of delay vs baseline length; twin axis shows cable-length equivalent
```
