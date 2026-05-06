"""Antenna position fitting via solar fringe-stopping.

For each antenna, scan trial positions (x or y) and find the one that minimizes
circular variance (maximizes phase coherence) after fringe-stopping + delay
correction. This verifies/corrects antenna positions in the CASM array.
"""

import csv

import numpy as np

from casm_io.constants import C_LIGHT_M_S

from casm_vis_analysis.fringe_stop import geometric_delay
from casm_vis_analysis.sources import find_transit_window


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def circular_variance_metric(vis, freq_mask=None, time_mask=None):
    """Circular variance: median over freq of (1 - |mean_over_time(e^{j*angle(V)})|).

    Lower is better (more coherent). 0 = perfect coherence, ~1 = random.

    Parameters
    ----------
    vis : ndarray, shape (T, F)
        Complex visibilities for a single baseline.
    freq_mask : ndarray of bool, shape (F,), optional
        True = good channel.
    time_mask : ndarray of bool, shape (T,), optional
        True = use this time sample.

    Returns
    -------
    score : float
    """
    if time_mask is not None:
        vis = vis[time_mask]
    phasors = np.exp(1j * np.angle(vis))
    mean_phasor = np.mean(phasors, axis=0)  # (F,)
    coherence = np.abs(mean_phasor)  # (F,)
    cvar = 1.0 - coherence
    if freq_mask is not None:
        cvar = cvar[freq_mask]
    return float(np.median(cvar))


def phase_stdev_metric(vis, freq_mask=None, time_mask=None):
    """Phase standard deviation: median over freq of std_over_time(angle(V)).

    Lower is better. 0 = constant phase, ~pi/sqrt(3) = uniform random.

    Parameters
    ----------
    vis : ndarray, shape (T, F)
        Complex visibilities for a single baseline.
    freq_mask : ndarray of bool, shape (F,), optional
        True = good channel.
    time_mask : ndarray of bool, shape (T,), optional
        True = use this time sample.

    Returns
    -------
    score : float
    """
    if time_mask is not None:
        vis = vis[time_mask]
    phase = np.angle(vis)  # (T, F)
    # Circular std: use phasor method
    phasors = np.exp(1j * phase)
    R = np.abs(np.mean(phasors, axis=0))  # (F,)
    # Circular std = sqrt(-2 * ln(R)), clamped for numerical safety
    R = np.clip(R, 1e-10, 1.0)
    circ_std = np.sqrt(-2.0 * np.log(R))
    if freq_mask is not None:
        circ_std = circ_std[freq_mask]
    return float(np.median(circ_std))


POSITION_METRICS = {
    "circvar": circular_variance_metric,
    "stdev": phase_stdev_metric,
}


# ---------------------------------------------------------------------------
# Parabola fit for uncertainty estimation
# ---------------------------------------------------------------------------

def fit_parabola_uncertainty(x_grid, scores):
    """Fit parabola around minimum to estimate best position and uncertainty.

    Uses ±10 points around the grid minimum and propagates uncertainty from
    the polyfit covariance matrix.

    Parameters
    ----------
    x_grid : ndarray, shape (N,)
        Trial x-positions.
    scores : ndarray, shape (N,)
        Metric scores at each trial position.

    Returns
    -------
    x_best : float
        Parabolic minimum x-position.
    sigma_x : float
        1-sigma uncertainty on x_best from fit covariance.
    coeffs : ndarray, shape (3,)
        Polynomial coefficients [a, b, c] for a*x^2 + b*x + c.
    """
    i_min = int(np.argmin(scores))

    # Use ±10 points around minimum for a well-constrained parabola
    i_lo = max(0, i_min - 10)
    i_hi = min(len(scores), i_min + 11)  # exclusive
    x_sub = x_grid[i_lo:i_hi]
    s_sub = scores[i_lo:i_hi]

    if len(x_sub) < 3:
        # Not enough points for a parabola; return grid minimum
        return float(x_grid[i_min]), np.inf, np.array([0.0, 0.0, float(scores[i_min])])

    coeffs, cov = np.polyfit(x_sub, s_sub, 2, cov=True)
    a, b, c = coeffs

    if a <= 0:
        # Not a proper minimum; return grid minimum
        return float(x_grid[i_min]), np.inf, coeffs

    x_best = -b / (2 * a)
    # Propagate covariance: x_best = -b/(2a)
    # dx/da = b/(2a^2), dx/db = -1/(2a)
    dxda = b / (2 * a**2)
    dxdb = -1.0 / (2 * a)
    sigma_x = np.sqrt(dxda**2 * cov[0, 0] + dxdb**2 * cov[1, 1]
                       + 2 * dxda * dxdb * cov[0, 1])

    return float(x_best), float(sigma_x), coeffs


# ---------------------------------------------------------------------------
# Sign auto-detection
# ---------------------------------------------------------------------------

def auto_detect_sign(vis_bl, freq_mhz, source_enu, baseline_enu,
                     time_mask=None, freq_mask=None):
    """Try +1 and -1 fringe-stop signs, return the one with higher coherence.

    Parameters
    ----------
    vis_bl : ndarray, shape (T, F)
        Visibilities for a single baseline.
    freq_mhz : ndarray, shape (F,)
    source_enu : ndarray, shape (T, 3)
    baseline_enu : ndarray, shape (3,)
    time_mask : ndarray of bool, shape (T,), optional
    freq_mask : ndarray of bool, shape (F,), optional

    Returns
    -------
    sign : int
        +1 or -1.
    """
    tau_s = geometric_delay(source_enu, baseline_enu)  # (T,)
    freq_hz = freq_mhz * 1e6

    best_sign = -1
    best_score = np.inf
    for sign in [-1, +1]:
        phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        vis_fs = vis_bl * np.exp(1j * phase)
        score = circular_variance_metric(vis_fs, freq_mask=freq_mask,
                                         time_mask=time_mask)
        if score < best_score:
            best_score = score
            best_sign = sign

    return best_sign


# ---------------------------------------------------------------------------
# Time window selection
# ---------------------------------------------------------------------------

def choose_time_windows(time_unix, source_name, min_alt_deg=10.0,
                        fit_frac=0.5, score_frac=0.8):
    """Choose inner (fit) and outer (score) time windows within transit.

    Parameters
    ----------
    time_unix : ndarray, shape (T,)
    source_name : str
    min_alt_deg : float
    fit_frac : float
        Fraction of transit window for delay fitting (inner).
    score_frac : float
        Fraction of transit window for metric scoring (outer).

    Returns
    -------
    time_mask_fit : ndarray of bool, shape (T,)
    time_mask_score : ndarray of bool, shape (T,)
    transit_info : dict
        Keys: i_start, i_end, n_transit.
    """
    i_start, i_end = find_transit_window(source_name, time_unix,
                                         min_alt_deg=min_alt_deg)
    n_transit = i_end - i_start + 1
    mid = (i_start + i_end) / 2.0

    # Inner window for delay fitting
    half_fit = int(n_transit * fit_frac / 2)
    i_fit_start = max(i_start, int(mid - half_fit))
    i_fit_end = min(i_end, int(mid + half_fit))

    # Outer window for scoring
    half_score = int(n_transit * score_frac / 2)
    i_score_start = max(i_start, int(mid - half_score))
    i_score_end = min(i_end, int(mid + half_score))

    time_mask_fit = np.zeros(len(time_unix), dtype=bool)
    time_mask_fit[i_fit_start:i_fit_end + 1] = True

    time_mask_score = np.zeros(len(time_unix), dtype=bool)
    time_mask_score[i_score_start:i_score_end + 1] = True

    return time_mask_fit, time_mask_score, {
        "i_start": i_start,
        "i_end": i_end,
        "n_transit": n_transit,
    }


# ---------------------------------------------------------------------------
# Single-baseline position scan (x or y)
# ---------------------------------------------------------------------------

def scan_position_single_baseline(vis_bl, freq_mhz, source_enu, ref_pos,
                                  target_pos_base, pos_grid, axis, sign,
                                  time_mask_fit, time_mask_score,
                                  freq_mask=None, metric="circvar"):
    """Scan trial positions along one axis for one baseline and find the best.

    For each trial position, builds the baseline, computes geometric delay,
    fringe-stops, fits linear delay, applies correction, computes metric.

    Parameters
    ----------
    vis_bl : ndarray, shape (T, F)
        Raw visibilities for this baseline.
    freq_mhz : ndarray, shape (F,)
    source_enu : ndarray, shape (T, 3)
    ref_pos : ndarray, shape (3,)
        Reference antenna ENU position.
    target_pos_base : ndarray, shape (3,)
        Target antenna position (the non-scanned coordinates are kept fixed).
    pos_grid : ndarray, shape (N,)
        Trial positions along the scan axis.
    axis : int
        0 = x (East), 1 = y (North).
    sign : int
        Fringe-stop sign convention.
    time_mask_fit : ndarray of bool, shape (T,)
        Time mask for delay fitting.
    time_mask_score : ndarray of bool, shape (T,)
        Time mask for metric scoring.
    freq_mask : ndarray of bool, shape (F,), optional
    metric : str
        Metric name from POSITION_METRICS.

    Returns
    -------
    result : dict
        best_pos, sigma_pos, best_score, best_delay_ns, scores, pos_grid,
        coeffs, grid_best_pos.
    """
    metric_fn = POSITION_METRICS[metric]
    freq_hz = freq_mhz * 1e6
    scores = np.empty(len(pos_grid))

    from casm_vis_analysis.delay import fit_delay, apply_delay

    for i, pos_trial in enumerate(pos_grid):
        target_pos = target_pos_base.copy()
        target_pos[axis] = pos_trial
        bl_enu = target_pos - ref_pos  # (3,)

        # Geometric delay: (T,)
        tau_s = geometric_delay(source_enu, bl_enu)

        # Inline fringe-stop: phase = sign * 2*pi * tau * freq
        phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        vis_fs = vis_bl * np.exp(1j * phase)  # (T, F)

        # Linear delay fit on fit window
        vis_fs_3d = vis_fs[:, :, np.newaxis]  # (T, F, 1) for fit_delay API
        fit_params = fit_delay(vis_fs_3d, freq_mhz, time_mask=time_mask_fit,
                               freq_mask=freq_mask, model="linear")
        vis_corr_3d = apply_delay(vis_fs_3d, freq_mhz, fit_params, model="linear")
        vis_corr = vis_corr_3d[:, :, 0]  # back to (T, F)

        scores[i] = metric_fn(vis_corr, freq_mask=freq_mask,
                              time_mask=time_mask_score)

    # Parabola fit for best position and uncertainty
    best_pos, sigma_pos, coeffs = fit_parabola_uncertainty(pos_grid, scores)

    # Get delay at best position
    i_best = int(np.argmin(scores))
    pos_at_min = pos_grid[i_best]
    target_pos_best = target_pos_base.copy()
    target_pos_best[axis] = pos_at_min
    bl_best = target_pos_best - ref_pos
    tau_best = geometric_delay(source_enu, bl_best)
    phase_best = sign * 2 * np.pi * tau_best[:, np.newaxis] * freq_hz[np.newaxis, :]
    vis_fs_best = vis_bl * np.exp(1j * phase_best)
    vis_fs_best_3d = vis_fs_best[:, :, np.newaxis]
    fit_best = fit_delay(vis_fs_best_3d, freq_mhz, time_mask=time_mask_fit,
                         freq_mask=freq_mask, model="linear")
    best_delay_ns = float(np.atleast_1d(fit_best["delay_ns"])[0])

    axis_name = "x" if axis == 0 else "y"
    return {
        f"best_{axis_name}": best_pos,
        f"sigma_{axis_name}": sigma_pos,
        "best_score": float(scores[i_best]),
        "best_delay_ns": best_delay_ns,
        "scores": scores,
        "pos_grid": pos_grid,
        "coeffs": coeffs,
        f"grid_best_{axis_name}": float(pos_at_min),
    }


def scan_x_single_baseline(vis_bl, freq_mhz, source_enu, ref_pos,
                           y_target, z_target, x_grid, sign,
                           time_mask_fit, time_mask_score,
                           freq_mask=None, metric="circvar"):
    """Scan trial x-positions for one baseline. Wrapper around scan_position_single_baseline."""
    target_pos_base = np.array([0.0, y_target, z_target])
    result = scan_position_single_baseline(
        vis_bl, freq_mhz, source_enu, ref_pos,
        target_pos_base, x_grid, axis=0, sign=sign,
        time_mask_fit=time_mask_fit, time_mask_score=time_mask_score,
        freq_mask=freq_mask, metric=metric,
    )
    # Map generalized keys back to x-specific keys for backward compatibility
    result["x_grid"] = result.pop("pos_grid")
    return result


# ---------------------------------------------------------------------------
# Fit all antennas
# ---------------------------------------------------------------------------

def fit_all_antennas(vis, freq_mhz, source_enu, ref_pos, target_positions,
                     target_labels, time_mask_fit, time_mask_score,
                     freq_mask=None, x_range=(-4, 4), x_step=0.05,
                     sign=-1, metric="circvar", axis=0):
    """Scan position along one axis for all target antennas.

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_bl)
        Raw visibilities (ref x targets).
    freq_mhz : ndarray, shape (F,)
    source_enu : ndarray, shape (T, 3)
    ref_pos : ndarray, shape (3,)
        Reference antenna ENU position.
    target_positions : ndarray, shape (n_bl, 3)
        Current target antenna positions.
    target_labels : list of str
    time_mask_fit : ndarray of bool, shape (T,)
    time_mask_score : ndarray of bool, shape (T,)
    freq_mask : ndarray of bool, shape (F,), optional
    x_range : tuple of (float, float)
        Range of trial offsets relative to current position.
    x_step : float
        Step size for trial positions.
    sign : int
    metric : str
    axis : int
        0 = x (East), 1 = y (North).

    Returns
    -------
    results : list of dict
        One dict per target antenna with scan results + label.
    """
    axis_name = "x" if axis == 0 else "y"
    results = []
    for bl_idx in range(vis.shape[2]):
        vis_bl = vis[:, :, bl_idx]
        cur_pos = target_positions[bl_idx, axis]
        pos_grid = np.arange(cur_pos + x_range[0],
                             cur_pos + x_range[1] + x_step / 2,
                             x_step)

        print(f"  Scanning {target_labels[bl_idx]}: "
              f"{axis_name} = [{pos_grid[0]:.2f}, {pos_grid[-1]:.2f}] m, "
              f"{len(pos_grid)} trials")

        result = scan_position_single_baseline(
            vis_bl, freq_mhz, source_enu, ref_pos,
            target_pos_base=target_positions[bl_idx].copy(),
            pos_grid=pos_grid, axis=axis, sign=sign,
            time_mask_fit=time_mask_fit,
            time_mask_score=time_mask_score,
            freq_mask=freq_mask,
            metric=metric,
        )
        result["label"] = target_labels[bl_idx]
        result[f"current_{axis_name}"] = float(cur_pos)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Write corrected layout CSV
# ---------------------------------------------------------------------------

def write_corrected_layout(original_path, output_path, antenna_ids,
                           fitted_x=None, fitted_y=None):
    """Read layout CSV, update x and/or y columns for fitted antennas, write new CSV.

    Parameters
    ----------
    original_path : str or Path
        Path to original antenna layout CSV.
    output_path : str or Path
        Path to write corrected CSV.
    antenna_ids : list of int
        Antenna IDs that were fitted.
    fitted_x : list or ndarray of float, optional
        Best-fit x-positions for each antenna.
    fitted_y : list or ndarray of float, optional
        Best-fit y-positions for each antenna.
    """
    with open(original_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Auto-detect column names (support both "antenna"/"x" and "antenna_id"/"x_m")
    ant_col = "antenna_id" if "antenna_id" in fieldnames else "antenna"
    x_col = "x_m" if "x_m" in fieldnames else "x"
    y_col = "y_m" if "y_m" in fieldnames else "y"

    x_map = dict(zip(antenna_ids, fitted_x)) if fitted_x is not None else {}
    y_map = dict(zip(antenna_ids, fitted_y)) if fitted_y is not None else {}

    for row in rows:
        aid = int(row[ant_col])
        if aid in x_map:
            row[x_col] = f"{x_map[aid]:.6f}"
        if aid in y_map:
            row[y_col] = f"{y_map[aid]:.6f}"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Cross-plank reference selection
# ---------------------------------------------------------------------------

def select_cross_plank_refs(antenna_ids, rows, positions):
    """Select reference antennas that maximize row separation per target.

    For each antenna, picks the reference from the row with the greatest
    |y| separation, minimizing cross-talk contamination.

    Parameters
    ----------
    antenna_ids : list of int
        All active antenna IDs.
    rows : dict of {int: str}
        Mapping antenna_id → row label (e.g., "N21", "C").
    positions : dict of {int: ndarray (3,)}
        Mapping antenna_id → ENU position.

    Returns
    -------
    ref_map : dict of {int: int}
        Mapping target_id → ref_id for each antenna.
    """
    # Compute mean y per row
    row_labels = set(rows.values())
    row_y = {}
    for rl in row_labels:
        ants_in_row = [a for a in antenna_ids if rows[a] == rl]
        row_y[rl] = np.mean([positions[a][1] for a in ants_in_row])

    # For each row, find the furthest row and pick the best ref from it
    # (closest to x=0 among antennas in the furthest row)
    row_best_ref = {}
    for rl in row_labels:
        best_dist = -1
        best_row = None
        for rl2 in row_labels:
            if rl2 == rl:
                continue
            dist = abs(row_y[rl] - row_y[rl2])
            if dist > best_dist:
                best_dist = dist
                best_row = rl2
        # Pick antenna closest to x=0 in the best row
        candidates = [a for a in antenna_ids if rows[a] == best_row]
        best_ref = min(candidates, key=lambda a: abs(positions[a][0]))
        row_best_ref[rl] = best_ref

    ref_map = {}
    for aid in antenna_ids:
        ref_map[aid] = row_best_ref[rows[aid]]

    return ref_map
