"""Delay models for diagnostic correction of fringe-stopped visibilities.

This module provides delay fitting and correction as a diagnostic tool.
Delay-corrected visibilities are NOT consumed by SVD beamforming
(which operates on fringe-stopped data directly).

Registry pattern: each model provides (fit, apply) function pair.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Linear delay model: phase(f) = slope * f + intercept
# ---------------------------------------------------------------------------

def linear_fit(vis_fs, freq_mhz, time_mask=None, freq_mask=None):
    """Fit linear delay from phase vs frequency slope.

    Parameters
    ----------
    vis_fs : ndarray, shape (T, F) or (T, F, n_bl)
        Fringe-stopped visibilities.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    time_mask : ndarray of bool, shape (T,), optional
        Mask selecting time samples to average over.
    freq_mask : ndarray of bool, shape (F,), optional
        Mask selecting good frequency channels for fitting (True = good).
        Phase is unwrapped and fit only on good channels, but the
        correction is applied to ALL channels.

    Returns
    -------
    params : dict
        slope: rad/MHz, intercept: rad, delay_ns: float, r_squared: float.
        For multi-baseline input, each value is an array over baselines.
    """
    if time_mask is not None:
        vis_avg = np.mean(vis_fs[time_mask], axis=0)
    else:
        vis_avg = np.mean(vis_fs, axis=0)

    # vis_avg: (F,) or (F, n_bl)
    multi_bl = vis_avg.ndim > 1
    if not multi_bl:
        vis_avg = vis_avg[:, np.newaxis]  # (F, 1)

    n_bl = vis_avg.shape[1]
    phase = np.angle(vis_avg)  # (F, n_bl)

    # Select good channels for fitting
    if freq_mask is not None:
        good_idx = np.where(freq_mask)[0]
        freq_fit = freq_mhz[good_idx]
        phase_fit = phase[good_idx]
    else:
        freq_fit = freq_mhz
        phase_fit = phase

    # Unwrap phase along frequency axis (only on good channels)
    phase_uw = np.unwrap(phase_fit, axis=0)

    slopes = np.empty(n_bl)
    intercepts = np.empty(n_bl)
    r_squared = np.empty(n_bl)

    for i in range(n_bl):
        coeffs = np.polyfit(freq_fit, phase_uw[:, i], 1)
        slopes[i] = coeffs[0]
        intercepts[i] = coeffs[1]
        # R-squared
        fit_vals = np.polyval(coeffs, freq_fit)
        ss_res = np.sum((phase_uw[:, i] - fit_vals) ** 2)
        ss_tot = np.sum((phase_uw[:, i] - np.mean(phase_uw[:, i])) ** 2)
        r_squared[i] = 1.0 - ss_res / (ss_tot + 1e-30)

    # delay_ns = slope / (2*pi) * 1e3  (slope is rad/MHz)
    delay_ns = slopes / (2 * np.pi) * 1e3

    if not multi_bl:
        slopes, intercepts, delay_ns, r_squared = (
            slopes[0], intercepts[0], delay_ns[0], r_squared[0]
        )

    return {
        "slope": slopes,
        "intercept": intercepts,
        "delay_ns": delay_ns,
        "r_squared": r_squared,
        "model": "linear",
    }


def linear_apply(vis, freq_mhz, fit_params):
    """Apply linear delay correction.

    Parameters
    ----------
    vis : ndarray, shape (T, F) or (T, F, n_bl)
        Visibilities to correct.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    fit_params : dict
        From linear_fit: slope, intercept.

    Returns
    -------
    vis_corrected : ndarray, same shape as vis
    """
    slope = np.asarray(fit_params["slope"])
    intercept = np.asarray(fit_params["intercept"])

    # phase correction: -(slope * freq + intercept)
    if slope.ndim == 0:
        phase_corr = -(slope * freq_mhz + intercept)  # (F,)
        correction = np.exp(1j * phase_corr)[np.newaxis, :, np.newaxis]
    else:
        # (F, n_bl)
        phase_corr = -(slope[np.newaxis, :] * freq_mhz[:, np.newaxis]
                        + intercept[np.newaxis, :])
        correction = np.exp(1j * phase_corr)[np.newaxis, :, :]

    return vis * correction


# ---------------------------------------------------------------------------
# Per-frequency phasor model: independent phase correction per channel
# ---------------------------------------------------------------------------

def phasor_fit(vis_fs, freq_mhz, time_mask=None, freq_mask=None):
    """Fit per-frequency phasor from time-averaged visibility.

    Parameters
    ----------
    vis_fs : ndarray, shape (T, F) or (T, F, n_bl)
        Fringe-stopped visibilities.
    freq_mhz : ndarray, shape (F,)
        Frequency axis (unused but kept for API consistency).
    time_mask : ndarray of bool, shape (T,), optional
        Mask selecting time samples to average over.

    Returns
    -------
    params : dict
        phasor_phase: phase per frequency channel, shape (F,) or (F, n_bl).
    """
    if time_mask is not None:
        vis_avg = np.mean(vis_fs[time_mask], axis=0)
    else:
        vis_avg = np.mean(vis_fs, axis=0)

    return {
        "phasor_phase": np.angle(vis_avg),
        "model": "per_freq_phasor",
    }


def phasor_apply(vis, freq_mhz, fit_params):
    """Apply per-frequency phasor correction.

    Parameters
    ----------
    vis : ndarray, shape (T, F) or (T, F, n_bl)
        Visibilities to correct.
    freq_mhz : ndarray, shape (F,)
        Frequency axis (unused but kept for API consistency).
    fit_params : dict
        From phasor_fit: phasor_phase.

    Returns
    -------
    vis_corrected : ndarray, same shape as vis
    """
    phasor = np.exp(-1j * fit_params["phasor_phase"])
    if phasor.ndim == 1:
        correction = phasor[np.newaxis, :, np.newaxis]
    else:
        correction = phasor[np.newaxis, :, :]
    return vis * correction


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DELAY_MODELS = {
    "linear": {"fit": linear_fit, "apply": linear_apply},
    "per_freq_phasor": {"fit": phasor_fit, "apply": phasor_apply},
}


def fit_delay(vis_fs, freq_mhz, time_mask=None, freq_mask=None, model="linear"):
    """Fit delay using registered model.

    Parameters
    ----------
    vis_fs : ndarray
        Fringe-stopped visibilities.
    freq_mhz : ndarray
        Frequency axis in MHz.
    time_mask : ndarray of bool, optional
        Time sample mask.
    freq_mask : ndarray of bool, optional
        Frequency channel mask (True = good). For linear model, fitting is
        restricted to good channels. Phasor model ignores this.
    model : str
        Model name (default 'linear').

    Returns
    -------
    params : dict
        Model-specific fit parameters.
    """
    if model not in DELAY_MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(DELAY_MODELS)}")
    return DELAY_MODELS[model]["fit"](vis_fs, freq_mhz, time_mask, freq_mask)


def apply_delay(vis, freq_mhz, fit_params, model="linear"):
    """Apply delay correction using registered model.

    Parameters
    ----------
    vis : ndarray
        Visibilities to correct.
    freq_mhz : ndarray
        Frequency axis in MHz.
    fit_params : dict
        From fit_delay.
    model : str
        Model name (default 'linear').

    Returns
    -------
    vis_corrected : ndarray
    """
    if model not in DELAY_MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(DELAY_MODELS)}")
    return DELAY_MODELS[model]["apply"](vis, freq_mhz, fit_params)


# ---------------------------------------------------------------------------
# Antenna-based delay decomposition
# ---------------------------------------------------------------------------

def build_delay_design_matrix(n_ant, baseline_pairs):
    """Build design matrix A where tau_ij = tau_j - tau_i.

    Parameters
    ----------
    n_ant : int
        Number of antennas.
    baseline_pairs : list of (int, int)
        List of (i, j) antenna index pairs.

    Returns
    -------
    A : ndarray, shape (n_baselines, n_ant)
        Design matrix.
    """
    n_bl = len(baseline_pairs)
    A = np.zeros((n_bl, n_ant))
    for k, (i, j) in enumerate(baseline_pairs):
        A[k, i] = -1.0
        A[k, j] = 1.0
    return A


def solve_antenna_delays(baseline_delays, design_matrix, weights=None,
                         ref_ant_idx=0):
    """Solve for per-antenna delays from baseline delays.

    Uses least-squares with a reference antenna constraint (delay=0).

    Parameters
    ----------
    baseline_delays : ndarray, shape (n_baselines,)
        Measured delay per baseline.
    design_matrix : ndarray, shape (n_baselines, n_ant)
        From build_delay_design_matrix.
    weights : ndarray, shape (n_baselines,), optional
        Per-baseline weights (e.g., 1/variance).
    ref_ant_idx : int
        Reference antenna index (delay set to 0).

    Returns
    -------
    ant_delays : ndarray, shape (n_ant,)
        Per-antenna delays.
    """
    A = design_matrix.copy()
    n_ant = A.shape[1]

    # Remove reference antenna column and solve
    keep = [i for i in range(n_ant) if i != ref_ant_idx]
    A_red = A[:, keep]

    if weights is not None:
        W = np.diag(np.sqrt(weights))
        A_red = W @ A_red
        baseline_delays = W @ baseline_delays

    result, _, _, _ = np.linalg.lstsq(A_red, baseline_delays, rcond=None)

    ant_delays = np.zeros(n_ant)
    for idx, col in enumerate(keep):
        ant_delays[col] = result[idx]

    return ant_delays
