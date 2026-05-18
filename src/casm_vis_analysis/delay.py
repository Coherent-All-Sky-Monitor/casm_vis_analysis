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

def linear_fit(vis_fs, freq_mhz, time_mask=None, freq_mask=None,
               tau_max_ns=1000.0, tau_step_ns=0.5,
               r_squared_threshold=0.7,
               peak_ratio_threshold=2.0,
               peak_exclusion_bins=None,
               return_coherence=False):
    """Fit linear delay via delay-domain coherent-sum search.

    Maximises ``|sum_f vis_avg(f) * exp(-i * 2*pi*tau*f)|`` over a tau
    grid. Does not unwrap — so missing channels (RFI gaps) cannot bias
    the slope by a multiple of 2*pi, even when the true phase wraps
    several times inside a gap. This matters for long-delay (cross-SNAP)
    baselines where each ~10 MHz RFI band can hide multiple wraps.

    Validity / unambiguous range
    ----------------------------
    For a uniform frequency grid with spacing ``dnu`` (Hz), the delay
    domain has Nyquist limit::

        tau_nyquist_ns = 1 / (2 * dnu_Hz) * 1e9

    Delays beyond this alias to ``tau +/- k/dnu`` for integer k. For
    CASM (~30.5 kHz channels) this is ~16 us, four orders of magnitude
    larger than any cable delay, so aliasing is not a concern in
    practice. The search range is further capped by ``tau_max_ns``
    (default 1 us) — fits at the edge of the search range are flagged
    as low quality via ``peak_to_secondary_ratio``.

    Robust to RFI mask gaps and low-SNR channels by construction: gaps
    just lower the SNR of the coherent sum, they cannot create
    spurious peaks at wrong delays the way unwrap-based methods do.

    Parameters
    ----------
    vis_fs : ndarray, shape (T, F, n_bl)
        Fringe-stopped visibilities.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    time_mask : ndarray of bool, shape (T,), optional
        Mask selecting time samples to average over.
    freq_mask : ndarray of bool, shape (F,), optional
        ``True`` = good channel. Search uses only good channels.
    tau_max_ns : float
        Half-width of the delay search range in ns. Capped at the
        Nyquist delay implied by the channel spacing.
    tau_step_ns : float
        Coarse delay grid resolution in ns. The peak is then refined
        with a quadratic interpolation, so sub-step precision is fine.
    r_squared_threshold : float
        Below this, the fit is flagged ``low_quality=True``. Catches
        baselines whose phase isn't well described by a single linear
        delay (heavy reflections, severe non-linear instrumental
        phase, etc.).
    peak_ratio_threshold : float
        Below this, ``low_quality=True``. Catches ambiguous fits where
        the chosen peak is not clearly the best candidate.
    peak_exclusion_bins : int, optional
        Half-width (in grid bins) of the window around the chosen peak
        that is excluded when finding the secondary peak — prevents
        the peak's own main lobe and first sidelobe ring from being
        picked as a separate candidate. Defaults to twice the first-
        null width of the delay-space sinc response,
        ``ceil(2 * 1e3 / B_dominant_MHz / tau_step_ns)``, where
        ``B_dominant_MHz`` is the longest contiguous good-channel
        segment. Using the dominant segment (not the total span) is
        important when heavy RFI gaps produce a tall structured
        sidelobe ring around the main peak.
    return_coherence : bool
        If ``True``, the per-baseline coherence curves and tau grid
        are included in the result for plotting. ``coherence_curves``
        has shape ``(n_grid, n_bl)``; ``tau_grid_ns`` has shape
        ``(n_grid,)``. Default ``False`` to keep the result small.

    Returns
    -------
    params : dict
        slope : rad/MHz
        intercept : rad
        delay_ns : float
        r_squared : float
            Coherence ratio (post-fit)^2 / (sum |v|)^2. 1.0 means all
            samples align perfectly after de-rotation.
        peak_to_secondary_ratio : float
            ``coherence[peak] / max(coherence[far from peak])``. >>1
            means the chosen delay is unambiguous; ~1 means the
            algorithm essentially flipped a coin between two
            candidates (low SNR, multiple delay components, or peak
            at search-range edge).
        low_quality : bool
            True if ``r_squared < r_squared_threshold`` or
            ``peak_to_secondary_ratio < peak_ratio_threshold``.
        tau_nyquist_ns : float
            Nyquist delay implied by the channel spacing. Reported
            once (scalar), same for all baselines.
        model : str
            ``"linear"``.

        For multi-baseline input, the per-baseline values are arrays
        over baselines.
    """
    if time_mask is not None:
        vis_avg = np.mean(vis_fs[time_mask], axis=0)
    else:
        vis_avg = np.mean(vis_fs, axis=0)

    multi_bl = vis_avg.ndim > 1
    if not multi_bl:
        vis_avg = vis_avg[:, np.newaxis]  # (F, 1)
    n_bl = vis_avg.shape[1]

    if freq_mask is not None:
        good_idx = np.where(freq_mask)[0]
    else:
        good_idx = np.arange(len(freq_mhz))
    f_g = freq_mhz[good_idx]
    n_good = len(f_g)

    # Cap search range at the Nyquist delay implied by channel spacing.
    df = float(np.abs(np.median(np.diff(freq_mhz))))  # MHz/channel
    tau_nyq_ns = 0.5 / df * 1e3 if df > 0 else tau_max_ns
    tau_max = float(min(tau_max_ns, tau_nyq_ns))
    n_grid = max(int(2 * tau_max / tau_step_ns) + 1, 11)
    tau_grid = np.linspace(-tau_max, tau_max, n_grid)
    slope_grid = 2.0 * np.pi * tau_grid * 1e-3  # rad/MHz

    # Exclusion window around the peak when finding the secondary peak.
    # The relevant main-lobe width is set by the LONGEST CONTIGUOUS run
    # of good channels (not the total span), because that segment
    # dominates the delay-space response. With heavy RFI gaps and an
    # asymmetric mask, the FT of the gap pattern produces a tall first
    # sidelobe ring within ~2 first-nulls of the peak; we exclude that
    # ring too so the "secondary" reflects an alternative delay
    # candidate rather than the main peak's own structured shoulder.
    if peak_exclusion_bins is None:
        if freq_mask is not None:
            mask_bool = np.asarray(freq_mask, dtype=bool)
        else:
            mask_bool = np.ones(len(freq_mhz), dtype=bool)
        # Run-length encode True segments
        if mask_bool.any():
            edges = np.diff(mask_bool.astype(np.int8))
            starts = np.where(edges == 1)[0] + 1
            ends = np.where(edges == -1)[0] + 1
            if mask_bool[0]:
                starts = np.concatenate([[0], starts])
            if mask_bool[-1]:
                ends = np.concatenate([ends, [len(mask_bool)]])
            # Width in MHz: count channels in segment * df
            seg_widths_mhz = (ends - starts) * df
            dominant_mhz = float(seg_widths_mhz.max())
        else:
            dominant_mhz = 1.0
        first_null_ns = 1e3 / dominant_mhz if dominant_mhz > 0 else tau_step_ns
        # 2 * first-null = main lobe + first sidelobe ring
        peak_exclusion_bins = max(
            1, int(np.ceil(2.0 * first_null_ns / tau_step_ns))
        )

    # Phasor matrix shared across baselines: (n_grid, n_good).
    # n_grid * n_good * 16 B; e.g. 4001 * 2258 * 16 = ~144 MB. Chunk if larger.
    bytes_full = n_grid * n_good * 16
    if bytes_full > 200_000_000:
        chunk = max(1, int(200_000_000 / (n_good * 16)))
    else:
        chunk = n_grid

    slopes = np.empty(n_bl)
    intercepts = np.empty(n_bl)
    delay_ns_out = np.empty(n_bl)
    r_squared = np.empty(n_bl)
    peak_to_secondary = np.empty(n_bl)
    coherence_curves = np.empty((n_grid, n_bl)) if return_coherence else None

    for i in range(n_bl):
        v_g = vis_avg[good_idx, i]

        # Coherent sum at each trial slope.
        coherence = np.empty(n_grid)
        for start in range(0, n_grid, chunk):
            end = min(start + chunk, n_grid)
            phasor = np.exp(-1j * slope_grid[start:end, None] * f_g[None, :])
            coherence[start:end] = np.abs(phasor @ v_g)

        if coherence_curves is not None:
            coherence_curves[:, i] = coherence

        peak = int(np.argmax(coherence))
        peak_value = float(coherence[peak])

        # Quadratic-interp refinement around the peak.
        if 0 < peak < n_grid - 1:
            y0, y1, y2 = coherence[peak - 1: peak + 2]
            denom = y0 - 2.0 * y1 + y2
            if denom != 0:
                shift = 0.5 * (y0 - y2) / denom
                tau_ns = tau_grid[peak] + shift * (tau_grid[1] - tau_grid[0])
            else:
                tau_ns = tau_grid[peak]
        else:
            tau_ns = tau_grid[peak]

        # Peak-to-secondary ratio: confidence in the chosen peak.
        # Mask out a window around the global peak (its parabolic
        # shoulders are not separate candidates) and find the next
        # highest value.
        lo = max(0, peak - peak_exclusion_bins)
        hi = min(n_grid, peak + peak_exclusion_bins + 1)
        secondary = np.concatenate([coherence[:lo], coherence[hi:]])
        if secondary.size > 0:
            secondary_value = float(secondary.max())
        else:
            secondary_value = 0.0
        if secondary_value > 0:
            peak_to_secondary[i] = peak_value / secondary_value
        else:
            peak_to_secondary[i] = np.inf

        slope_i = 2.0 * np.pi * tau_ns * 1e-3
        v_rot = v_g * np.exp(-1j * slope_i * f_g)
        intercept_i = float(np.angle(np.sum(v_rot)))

        # R-squared: post-fit coherence fraction. 1.0 means all
        # samples align perfectly in the complex plane after de-rotation.
        coh_after = float(np.abs(np.sum(v_rot * np.exp(-1j * intercept_i))))
        coh_total = float(np.sum(np.abs(v_g)))
        r_squared[i] = (coh_after / coh_total) ** 2 if coh_total > 0 else 0.0

        slopes[i] = slope_i
        intercepts[i] = intercept_i
        delay_ns_out[i] = tau_ns

    low_quality = (r_squared < r_squared_threshold) | (
        peak_to_secondary < peak_ratio_threshold
    )

    if not multi_bl:
        slopes, intercepts, delay_ns_out, r_squared, peak_to_secondary, low_quality = (
            slopes[0], intercepts[0], delay_ns_out[0],
            r_squared[0], peak_to_secondary[0], bool(low_quality[0]),
        )

    result = {
        "slope": slopes,
        "intercept": intercepts,
        "delay_ns": delay_ns_out,
        "r_squared": r_squared,
        "peak_to_secondary_ratio": peak_to_secondary,
        "low_quality": low_quality,
        "tau_nyquist_ns": float(tau_nyq_ns),
        "model": "linear",
    }
    if return_coherence:
        if not multi_bl:
            result["coherence_curves"] = coherence_curves[:, 0]
        else:
            result["coherence_curves"] = coherence_curves
        result["tau_grid_ns"] = tau_grid
    return result


def linear_apply(vis, freq_mhz, fit_params):
    """Apply linear delay correction.

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_bl)
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

def phasor_fit(vis_fs, freq_mhz, time_mask=None, freq_mask=None, **_kwargs):
    """Fit per-frequency phasor from time-averaged visibility.

    Parameters
    ----------
    vis_fs : ndarray, shape (T, F, n_bl)
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
    vis : ndarray, shape (T, F, n_bl)
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
# Bare-phasor API (ported from casm-bf-imaging delay_fit.py for callers
# that want the angle array directly, not wrapped in a fit_params dict).
# ---------------------------------------------------------------------------


def compute_per_freq_phasor(vis_fs, time_mask=None, freq_mask=None):
    """Bare per-frequency phasor angle.

    Returns ``angle(mean_t(vis_fs))`` directly. Equivalent to
    ``phasor_fit(...)['phasor_phase']`` but skips the dict wrapper.

    Parameters
    ----------
    vis_fs : ndarray, shape (T, F) or (T, F, n_bl)
    time_mask : ndarray of bool, shape (T,), optional
    freq_mask : ndarray of bool, shape (F,), optional
        Currently unused (kept for API symmetry with the bf-imaging form).

    Returns
    -------
    phasor_phase : ndarray, shape (F,) or (F, n_bl)
    """
    if time_mask is not None:
        vis_avg = np.mean(vis_fs[time_mask], axis=0)
    else:
        vis_avg = np.mean(vis_fs, axis=0)
    return np.angle(vis_avg)


def apply_per_freq_phasor(vis, phasor_phase):
    """Apply a bare per-frequency phasor (angle array).

    Counterpart to :func:`compute_per_freq_phasor`. Equivalent to
    ``phasor_apply(vis, freq_mhz, {'phasor_phase': phasor_phase})``.
    """
    phasor = np.exp(-1j * phasor_phase)
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


def fit_delay(vis_fs, freq_mhz, time_mask=None, freq_mask=None, model="linear",
              **kwargs):
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
    **kwargs
        Forwarded to the model's fit function (e.g. ``tau_max_ns``,
        ``return_coherence`` for ``model='linear'``).

    Returns
    -------
    params : dict
        Model-specific fit parameters.
    """
    if model not in DELAY_MODELS:
        raise ValueError(f"Unknown model '{model}'. Available: {list(DELAY_MODELS)}")
    return DELAY_MODELS[model]["fit"](
        vis_fs, freq_mhz, time_mask, freq_mask, **kwargs
    )


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
