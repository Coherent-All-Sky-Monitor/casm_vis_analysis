"""Fringe-stopping for CASM correlator visibilities.

Computes geometric delays from source direction and baseline vectors,
then applies phase corrections to stop fringes.

Convention
----------
phase = sign * 2*pi * freq_hz * tau_s
vis_stopped = vis * exp(1j * phase)

Default sign=-1 removes geometric phase from visibilities.
"""

import numpy as np

from casm_io.constants import C_LIGHT_M_S


def compute_baselines_enu(positions_enu, ref_idx, target_idxs):
    """Compute baseline vectors from reference to target antennas.

    Parameters
    ----------
    positions_enu : ndarray, shape (n_ant, 3)
        ENU positions in meters.
    ref_idx : int
        Reference antenna index.
    target_idxs : array-like of int
        Target antenna indices.

    Returns
    -------
    baselines : ndarray, shape (n_targets, 3)
        Baseline vectors (target - ref) in ENU meters.
    """
    target_idxs = np.asarray(target_idxs)
    return positions_enu[target_idxs] - positions_enu[ref_idx]


def geometric_delay(source_enu, baseline_enu):
    """Compute geometric delay for source direction and baseline(s).

    tau = (baseline . source_hat) / c

    Parameters
    ----------
    source_enu : ndarray, shape (T, 3)
        ENU unit direction vectors toward source.
    baseline_enu : ndarray, shape (3,) or (n_bl, 3)
        Baseline vector(s) in meters.

    Returns
    -------
    tau_s : ndarray
        Geometric delay in seconds.
        Shape (T,) for single baseline, (T, n_bl) for multiple.
    """
    source_enu = np.atleast_2d(source_enu)  # (T, 3)
    baseline_enu = np.atleast_2d(baseline_enu)  # (n_bl, 3)

    # dot product: (T, 3) @ (3, n_bl) -> (T, n_bl)
    tau_s = source_enu @ baseline_enu.T / C_LIGHT_M_S

    if tau_s.shape[1] == 1:
        tau_s = tau_s[:, 0]  # squeeze single baseline
    return tau_s


def fringe_stop(vis, freq_mhz, tau_s, sign=-1):
    """Apply fringe-stopping to visibilities.

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_bl)
        Raw complex visibilities.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    tau_s : ndarray, shape (T,) or (T, n_bl)
        Geometric delay in seconds per time sample (and per baseline).
    sign : int
        Sign convention. Default -1 removes geometric phase.

    Returns
    -------
    result : dict
        Keys:
        - vis_raw: original visibilities
        - vis_stopped: fringe-stopped visibilities
        - vis_for_calibration: same as vis_stopped (API contract for casm_calibration)
        - geometric_phase: phase applied, shape (T, F) or (T, F, n_bl)
        - tau_s: geometric delays
        - sign: sign used
        - freq_mhz: frequency axis
    """
    freq_hz = freq_mhz * 1e6  # (F,)
    tau_s = np.asarray(tau_s)

    if tau_s.ndim == 1:
        # Single baseline or shared delay: (T,) -> (T, F) via outer product
        phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        # Broadcast to vis shape
        correction = np.exp(1j * phase)[:, :, np.newaxis]
    else:
        # Per-baseline delays: (T, n_bl)
        # phase: (T, F, n_bl) = tau(T,1,n_bl) * freq(1,F,1)
        phase = sign * 2 * np.pi * (
            tau_s[:, np.newaxis, :] * freq_hz[np.newaxis, :, np.newaxis]
        )
        correction = np.exp(1j * phase)

    vis_stopped = vis * correction

    # Ensure geometric_phase matches vis shape (T, F, n_bl)
    if phase.ndim == 2:
        geo_phase = phase[:, :, np.newaxis]
    else:
        geo_phase = phase

    return {
        "vis_raw": vis,
        "vis_stopped": vis_stopped,
        "vis_for_calibration": vis_stopped,
        "geometric_phase": geo_phase,
        "tau_s": tau_s,
        "sign": sign,
        "freq_mhz": freq_mhz,
    }
