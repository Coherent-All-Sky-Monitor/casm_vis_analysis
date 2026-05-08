"""Fringe-stopping for CASM correlator visibilities.

Computes geometric delays from source direction and baseline vectors,
then applies phase corrections to stop fringes.

Convention
----------
phase = sign * 2*pi * freq_hz * tau_s
vis_stopped = vis * exp(1j * phase)

Default sign=-1 removes geometric phase from visibilities.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from casm_io.constants import C_LIGHT_M_S


class FringeStoppedData(TypedDict, total=False):
    """Fringe-stop output. Consumed by SVD calibration and imaging."""
    vis: np.ndarray            # original (T, F, n_bl) complex64
    vis_stopped: np.ndarray    # fringe-stopped (T, F, n_bl)
    vis_for_calibration: np.ndarray  # alias of vis_stopped (calibration contract)
    geometric_phase: np.ndarray
    tau_s: np.ndarray
    freq_mhz: np.ndarray
    time_unix: np.ndarray
    source: str
    ref_ant: int
    sign: int
    target_aids: list
    target_labels: list


# ---------------------------------------------------------------------------
# Array-level primitives (existing API; unchanged signatures)
# ---------------------------------------------------------------------------


def compute_baselines_enu(positions_enu, ref_idx, target_idxs):
    """Compute baseline vectors from reference to target antennas.

    Parameters
    ----------
    positions_enu : ndarray, shape (n_ant, 3)
    ref_idx : int
    target_idxs : array-like of int

    Returns
    -------
    baselines : ndarray, shape (n_targets, 3)
        target - ref in ENU meters.
    """
    target_idxs = np.asarray(target_idxs)
    return positions_enu[target_idxs] - positions_enu[ref_idx]


def geometric_delay(source_enu, baseline_enu):
    """Compute geometric delay tau = (b . s) / c.

    Parameters
    ----------
    source_enu : ndarray, shape (T, 3)
    baseline_enu : ndarray, shape (3,) or (n_bl, 3)

    Returns
    -------
    tau_s : ndarray, shape (T,) or (T, n_bl)
    """
    source_enu = np.atleast_2d(source_enu)
    baseline_enu = np.atleast_2d(baseline_enu)
    tau_s = source_enu @ baseline_enu.T / C_LIGHT_M_S
    if tau_s.shape[1] == 1:
        tau_s = tau_s[:, 0]
    return tau_s


def fringe_stop_array(vis, freq_mhz, tau_s, sign=-1):
    """Apply fringe-stopping to visibilities (array-level).

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_bl)
    freq_mhz : ndarray, shape (F,)
    tau_s : ndarray, shape (T,) or (T, n_bl)
    sign : int

    Returns
    -------
    dict with vis_raw, vis_stopped, vis_for_calibration, geometric_phase,
    tau_s, sign, freq_mhz.
    """
    freq_hz = freq_mhz * 1e6
    tau_s = np.asarray(tau_s)

    if tau_s.ndim == 1:
        phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        correction = np.exp(1j * phase)[:, :, np.newaxis]
    else:
        phase = sign * 2 * np.pi * (
            tau_s[:, np.newaxis, :] * freq_hz[np.newaxis, :, np.newaxis]
        )
        correction = np.exp(1j * phase)

    vis_stopped = vis * correction

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


# Backward-compat alias for callers using the old name.
fringe_stop_vis = fringe_stop_array


# ---------------------------------------------------------------------------
# Single-baseline form (ported from casm-bf-imaging)
# ---------------------------------------------------------------------------


def fringe_stop_single_baseline(vis, freq_hz, tau_s, sign=-1):
    """Fringe-stop a single baseline.

    Parameters
    ----------
    vis : ndarray, shape (T, F)
    freq_hz : ndarray, shape (F,)
    tau_s : ndarray, shape (T,)
        Geometric delay per time sample.
    sign : int

    Returns
    -------
    vis_fs : ndarray, shape (T, F)
    """
    tau_s = np.asarray(tau_s)
    phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
    return vis * np.exp(1j * phase)


def coherence_metric(vis, freq_mask=None):
    """Coherence over frequency: |mean(exp(i*phase))|.

    High when phases are aligned across frequency (i.e. fringe-stopped well).

    Parameters
    ----------
    vis : ndarray, shape (T, F) or (T, F, n_bl)
    freq_mask : ndarray of bool, optional
        True for frequencies to include.

    Returns
    -------
    coh : ndarray, shape (T,) or (T, n_bl)
    """
    if freq_mask is not None:
        vis = vis[:, freq_mask, ...]
    unit_phasors = np.exp(1j * np.angle(vis))
    return np.abs(np.nanmean(unit_phasors, axis=1))


def auto_detect_sign(vis, freq_mhz, tau_s, freq_mask=None):
    """Pick the fringe-stop sign that maximises post-stop coherence.

    Parameters
    ----------
    vis : ndarray, shape (T, F)
    freq_mhz : ndarray, shape (F,)
    tau_s : ndarray, shape (T,)
    freq_mask : ndarray of bool, optional

    Returns
    -------
    sign : int
        +1 or -1.
    """
    freq_hz = freq_mhz * 1e6
    fs_pos = fringe_stop_single_baseline(vis, freq_hz, tau_s, sign=+1)
    fs_neg = fringe_stop_single_baseline(vis, freq_hz, tau_s, sign=-1)
    coh_pos = np.nanmean(coherence_metric(fs_pos, freq_mask))
    coh_neg = np.nanmean(coherence_metric(fs_neg, freq_mask))
    return +1 if coh_pos > coh_neg else -1


# ---------------------------------------------------------------------------
# Dict-based wrapper (compose-friendly notebook API)
# ---------------------------------------------------------------------------


def _vis_dict_get(data, key):
    """Read a key from either a dict-like or a dataclass-like data container."""
    if hasattr(data, "__getitem__"):
        try:
            return data[key]
        except (KeyError, TypeError):
            pass
    return getattr(data, key)


def fringe_stop(data, ant, *, ref_ant, source, sign=-1,
                min_alt_deg=10.0, rfi_mask=None) -> FringeStoppedData:
    """Compose-friendly fringe-stop.

    Accepts the dict (or :class:`VisibilityResult`) returned by
    ``casm_io.read_visibilities`` / ``VisibilityReader.read``, the
    :class:`AntennaMapping`, the reference antenna ID, and the source
    name. Returns a :class:`FringeStoppedData` ready for SVD calibration
    and imaging.

    Reuses both inputs in place:
      * The in-memory visibility data is sliced to ref<->target baselines;
        no re-read from disk.
      * The active set comes from ``ant.active_antennas()`` and so honours
        ``ant.with_inactive([...])`` overrides set after looking at
        autocorr panels.

    Accepts ``data['vis']`` in either shape:
      * full upper-triangle ``(T, F, n_inputs*(n_inputs+1)/2)`` — sliced
        internally via ``triu_flat_index``;
      * pre-filtered ``(T, F, n_targets)`` — used as-is when the baseline
        count exactly matches ``len(active_antennas) - 1``.

    The lower-level array primitives (``compute_baselines_enu``,
    ``geometric_delay``, ``fringe_stop_array``) remain available for
    callers that need them.
    """
    from casm_io.correlator.baselines import triu_flat_index
    from casm_vis_analysis.sources import source_enu, find_transit_window

    vis = _vis_dict_get(data, "vis")
    freq_mhz = _vis_dict_get(data, "freq_mhz")
    time_unix = _vis_dict_get(data, "time_unix")

    active_sorted = sorted(ant.active_antennas())
    if ref_ant not in active_sorted:
        raise ValueError(
            f"ref_ant={ref_ant} is not in active_antennas() "
            f"({active_sorted[:5]}{'...' if len(active_sorted) > 5 else ''})"
        )
    target_aids = [a for a in active_sorted if a != ref_ant]

    # Slice ref<->target baselines if vis is full upper-triangle.
    # If shape exactly matches len(target_aids), assume pre-filtered;
    # otherwise treat as full triangle and slice. Order matters because
    # tiny mappings can produce ambiguous baseline counts.
    n_bl = vis.shape[-1]
    if n_bl == len(target_aids):
        vis_used = vis
    else:
        n_full = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)
        if n_full * (n_full + 1) // 2 != n_bl:
            raise ValueError(
                f"vis has {n_bl} baselines but expected either full "
                f"upper-triangle (n*(n+1)/2 for some n) or "
                f"{len(target_aids)} (ref+targets for the active set). "
                f"If you pre-filtered with different ref/targets, slice "
                f"externally and call fringe_stop_array() directly."
            )
        ref_pidx = ant.packet_index(ref_ant)
        target_pidxs = [ant.packet_index(aid) for aid in target_aids]
        # The upper-triangle index expects i <= j. Some targets may have
        # packet_index < ref_pidx, so always pass (min, max). The vis
        # element at (i, j) is conjugate-symmetric of (j, i); for the
        # cross-correlation magnitude this doesn't matter but we DO need
        # to conjugate when the order is flipped so the geometric phase
        # convention stays right.
        bl_idxs = []
        conjugate_mask = []
        for p in target_pidxs:
            i, j = sorted((ref_pidx, p))
            bl_idxs.append(triu_flat_index(n_full, i, j))
            conjugate_mask.append(p < ref_pidx)
        vis_used = vis[:, :, bl_idxs]
        if any(conjugate_mask):
            mask = np.array(conjugate_mask, dtype=bool)
            vis_used = vis_used.copy()    # writable
            vis_used[:, :, mask] = np.conj(vis_used[:, :, mask])

    df = ant.dataframe
    positions = np.array([
        df.loc[df["antenna_id"] == a, ["x_m", "y_m", "z_m"]].values[0]
        for a in active_sorted
    ])
    ref_pos_idx = active_sorted.index(ref_ant)
    target_pos_idxs = [active_sorted.index(a) for a in target_aids]
    bl_enu = compute_baselines_enu(positions, ref_pos_idx, target_pos_idxs)

    s_enu = source_enu(source, time_unix)
    tau_s = geometric_delay(s_enu, bl_enu)

    fs = fringe_stop_array(vis_used, freq_mhz, tau_s, sign=sign)

    # Resolve the per-channel mask before NaN-filling. Precedence:
    #   1. explicit rfi_mask= kwarg (RFIMask, bool array, or None-ignore)
    #   2. data['freq_mask'] populated by apply_rfi_mask() — True = flagged
    #   3. all-good
    # Internal `freq_mask` is True = GOOD (matches fit_delay convention).
    from casm_vis_analysis.rfi import _freq_mask_for_channel
    if rfi_mask is None:
        _flag = _freq_mask_for_channel(data)
        freq_mask = np.ones(len(freq_mhz), dtype=bool) if _flag is None else (~_flag)
    elif callable(rfi_mask):
        freq_mask = np.asarray(rfi_mask(freq_mhz), dtype=bool)
    else:
        freq_mask = np.asarray(rfi_mask, dtype=bool)
    if freq_mask.shape != (len(freq_mhz),):
        raise ValueError(
            f"rfi_mask shape {freq_mask.shape} doesn't match freq axis "
            f"({len(freq_mhz)},). Pass an RFIMask, a bool array of "
            f"length {len(freq_mhz)}, or None."
        )

    # Note: we do NOT NaN-fill fs['vis_stopped'] at flagged channels.
    # Reason: fit_delay / plot_phase_vs_freq / plot_fringe_diag aren't
    # NaN-aware (np.unwrap and np.polyfit return all-NaN once a NaN
    # appears in the input), and silently breaking those plotters is
    # worse than letting RFI residuals show as scatter. Downstream
    # stages that should respect the mask (subband SVD, save_calibration)
    # read fs['freq_mask'] and skip / zero flagged channels themselves.
    # If you specifically want NaN gaps in your fringe-stop diagnostic,
    # do it at plot time:
    #     vis_for_diag = fs['vis_stopped'].copy()
    #     vis_for_diag[:, ~fs['freq_mask'], :] = np.nan + 1j*np.nan

    target_labels = [
        f"Ant {ref_ant}|S{ant.snap_adc(ref_ant)[0]}A{ant.snap_adc(ref_ant)[1]} "
        f"x Ant {aid}|S{s}A{a}"
        for aid, (s, a) in zip(target_aids,
                               (ant.snap_adc(aid) for aid in target_aids))
    ]

    # Transit-window time mask: True where the source is above min_alt_deg.
    # Downstream stages (especially SVD calibration) use this to time-average
    # only when the source is up; otherwise the matrix gets diluted with
    # low-SNR samples and lambda_1/lambda_2 collapses.
    try:
        i0, i1 = find_transit_window(source, time_unix, min_alt_deg=min_alt_deg)
        time_mask = np.zeros(len(time_unix), dtype=bool)
        time_mask[i0:i1 + 1] = True
    except ValueError:
        # Source never rises -- keep all samples and let the user decide.
        time_mask = np.ones(len(time_unix), dtype=bool)

    return {
        "vis": vis_used,
        "vis_stopped": fs["vis_stopped"],
        "vis_for_calibration": fs["vis_stopped"],
        "geometric_phase": fs["geometric_phase"],
        "tau_s": tau_s,
        "freq_mhz": freq_mhz,
        "time_unix": time_unix,
        "time_mask": time_mask,
        "freq_mask": freq_mask,
        "source": source,
        "ref_ant": ref_ant,
        "sign": sign,
        "target_aids": target_aids,
        "target_labels": target_labels,
    }


# Note: the public name `fringe_stop` is now the dict-based wrapper above.
# The legacy array-level entry point is `fringe_stop_array` (or its
# backward-compat alias `fringe_stop_vis`). `runners.py` uses
# `fringe_stop_array` explicitly.
