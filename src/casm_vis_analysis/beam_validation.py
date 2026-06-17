"""Validate deployed beamforming weights against the calibrator data.

The deploy pipeline writes a SNAP int8 HDF5 with N stationary beams
pointed at fixed (alt, az). The validation question is: "if I take the
visibility data this cal was solved against, apply the cal, and beam-
form at one of the deployed pointings, do I see source power rise as
the source transits through that beam?"

This module is the integration point: it reads the deployed weights
file, predicts source-beam transits geometrically, then beamforms the
in-memory visibilities at the same pointings using
:func:`beam_power_vs_time`. A pass means the cal/geo combination is
producing real beams on the sky, not just a coherent sum toward the
calibrator direction.

Public API
----------
* :class:`BeamHit` — one source crossing through one beam.
* :func:`load_beams_from_int8` — pointings + array geometry from HDF5.
* :func:`find_source_beam_transits` — geometric prediction.
* :func:`validate_beam_weights` — top-level orchestrator.
* :func:`plot_beam_validation` — multi-panel diagnostic figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import json
import numpy as np

from .beam_power import beam_power_vs_time
from .sources import source_altaz


@dataclass
class BeamHit:
    """One contiguous interval where a source was within one beam FWHM."""
    beam_idx: int
    beam_alt_deg: float
    beam_az_deg: float
    beam_name: str
    source: str
    entry_unix: float
    exit_unix: float
    peak_unix: float
    peak_alt_deg: float
    peak_az_deg: float
    min_dist_deg: float
    duration_min: float


def load_beams_from_int8(int8_h5_path) -> dict:
    """Read beam pointings + active-array geometry from a SNAP int8 HDF5.

    Returns
    -------
    dict with keys
        ``alt_deg`` : (n_beams,) float — beam altitudes
        ``az_deg``  : (n_beams,) float — beam azimuths
        ``names``   : list of n_beams beam name strings
        ``n_beams`` : int
        ``active_positions_enu`` : (n_active, 3) ENU positions (m)
        ``fwhm_ew_deg`` : float — diffraction-limited E-W FWHM
        ``fwhm_ns_deg`` : float — diffraction-limited N-S FWHM
        ``csv_path`` : str — provenance from the file
    """
    import h5py

    with h5py.File(int8_h5_path, "r") as f:
        # Schema/version/shape guards: refuse to load files whose layout
        # we don't recognise rather than silently producing bad geometry.
        ft = f.attrs.get("format_type", "")
        if isinstance(ft, bytes):
            ft = ft.decode()
        if ft != "int8_snap_weights":
            raise ValueError(
                f"{int8_h5_path}: not an int8 weights file "
                f"(format_type={ft!r}, expected 'int8_snap_weights')."
            )
        version = str(f.attrs.get("version", "unknown"))
        if version not in {"1.0", "2.0"}:
            raise ValueError(
                f"{int8_h5_path}: unsupported schema v{version}"
            )
        for key in ("pointings/alt_deg", "pointings/az_deg",
                    "array_config/positions_enu",
                    "array_config/active_mask"):
            if key not in f:
                raise ValueError(
                    f"{int8_h5_path}: missing dataset {key!r}"
                )

        alt_deg = f["pointings/alt_deg"][:]
        az_deg = f["pointings/az_deg"][:]
        names = json.loads(f["pointings"].attrs["names"])
        positions = f["array_config/positions_enu"][:]
        active = f["array_config/active_mask"][:]
        csv_path = str(f["array_config"].attrs.get("csv_path", ""))

    active_pos = np.asarray(positions, dtype=np.float64)[np.asarray(active, dtype=bool)]

    # FWHM at the array's geometric design freq. Lazy import to avoid
    # hard package coupling — this is the only bf_weights symbol we
    # touch, and it's a pure function of antenna positions.
    from bf_weights_generator import compute_beam_fwhm
    fwhm_ew, fwhm_ns = compute_beam_fwhm(active_pos)

    return {
        "alt_deg": np.asarray(alt_deg, dtype=np.float64),
        "az_deg": np.asarray(az_deg, dtype=np.float64),
        "names": list(names),
        "n_beams": int(len(alt_deg)),
        "active_positions_enu": active_pos,
        "fwhm_ew_deg": float(fwhm_ew),
        "fwhm_ns_deg": float(fwhm_ns),
        "csv_path": csv_path,
    }


def find_source_beam_transits(
    beams: Mapping,
    sources: Iterable[str],
    time_unix,
    *,
    fwhm_factor: float = 1.0,
) -> list[BeamHit]:
    """Predict which (beam, source) pairs cross within one FWHM during the window.

    Parameters
    ----------
    beams : dict from :func:`load_beams_from_int8`.
    sources : iterable of source name strings (resolved via
        :func:`casm_vis_analysis.sources.source_altaz`).
    time_unix : ndarray (T,) — Unix timestamps to evaluate over.
    fwhm_factor : float, default 1.0 — "in-beam" condition is
        ``elliptical distance < fwhm_factor`` in units of half-FWHM.
        Use 0.5 for "within half-power", 1.0 for "within FWHM".

    Returns
    -------
    list of :class:`BeamHit`, one per contiguous in-beam interval.
    """
    time_unix = np.asarray(time_unix, dtype=np.float64)
    if time_unix.size < 2:
        return []
    half_ew = beams["fwhm_ew_deg"] / 2.0
    half_ns = beams["fwhm_ns_deg"] / 2.0
    bn_alt = np.asarray(beams["alt_deg"], dtype=np.float64)
    bn_az = np.asarray(beams["az_deg"], dtype=np.float64)
    names = beams["names"]

    hits: list[BeamHit] = []
    for src in sources:
        s_alt, s_az = source_altaz(src, time_unix)
        s_alt = np.asarray(s_alt, dtype=np.float64)
        s_az = np.asarray(s_az, dtype=np.float64)

        for bi in range(len(bn_alt)):
            d_alt = s_alt - bn_alt[bi]
            # Wrap az delta into [-180, 180].
            d_az = ((s_az - bn_az[bi] + 180.0) % 360.0) - 180.0
            cos_alt = np.cos(np.deg2rad(bn_alt[bi]))
            d_az_sky = d_az * cos_alt
            r = np.sqrt((d_az_sky / half_ew) ** 2 + (d_alt / half_ns) ** 2)

            in_beam = r < fwhm_factor
            if not np.any(in_beam):
                continue

            idxs = np.where(in_beam)[0]
            splits = np.where(np.diff(idxs) > 1)[0] + 1
            for grp in np.split(idxs, splits):
                i_entry, i_exit = grp[0], grp[-1]
                i_peak = grp[np.argmin(r[grp])]
                hits.append(BeamHit(
                    beam_idx=int(bi),
                    beam_alt_deg=float(bn_alt[bi]),
                    beam_az_deg=float(bn_az[bi]),
                    beam_name=str(names[bi]),
                    source=str(src),
                    entry_unix=float(time_unix[i_entry]),
                    exit_unix=float(time_unix[i_exit]),
                    peak_unix=float(time_unix[i_peak]),
                    peak_alt_deg=float(s_alt[i_peak]),
                    peak_az_deg=float(s_az[i_peak]),
                    min_dist_deg=float(
                        r[i_peak] * max(half_ew, half_ns)
                    ),
                    duration_min=float(
                        (time_unix[i_exit] - time_unix[i_entry]) / 60.0
                    ),
                ))
    return hits


def _select_beams_for_validation(
    hits: list[BeamHit],
    beams: Mapping,
    sources: Iterable[str],
    time_unix,
    *,
    max_hit_panels: int = 12,
    n_control_beams: int = 2,
) -> tuple[list[int], list[int]]:
    """Pick which beams to actually beamform at.

    Returns ``(hit_beam_idxs, control_beam_idxs)``.
      * ``hit_beam_idxs`` covers the most-prominent source crossings,
        capped at ``max_hit_panels`` for plot readability, sorted by
        longest in-beam residency.
      * ``control_beam_idxs`` are beams MAXIMALLY distant from every
        source track at every time sample. We compute, per beam, the
        minimum elliptical distance (in FWHM units) to any source
        position over the data window, then pick the beams with the
        largest min-distance. This avoids "control" beams that
        accidentally sit under a source track and pick up sidelobes.
    """
    by_beam: dict[int, list[BeamHit]] = {}
    for h in hits:
        by_beam.setdefault(h.beam_idx, []).append(h)

    # Hit beams ranked by total in-beam duration (descending).
    ranked = sorted(
        by_beam.items(),
        key=lambda kv: -sum(h.duration_min for h in kv[1]),
    )
    hit_idxs = [bi for bi, _ in ranked[:max_hit_panels]]
    if n_control_beams <= 0:
        return hit_idxs, []

    # For every (beam, time) compute elliptical distance to each
    # source. Per beam, take the min over time and over sources —
    # that's the closest the beam ever gets to any source track.
    half_ew = beams["fwhm_ew_deg"] / 2.0
    half_ns = beams["fwhm_ns_deg"] / 2.0
    bn_alt = np.asarray(beams["alt_deg"], dtype=np.float64)
    bn_az = np.asarray(beams["az_deg"], dtype=np.float64)
    time_unix = np.asarray(time_unix, dtype=np.float64)

    # Stack source tracks: (n_src, T)
    src_tracks = []
    for src in sources:
        s_alt, s_az = source_altaz(src, time_unix)
        src_tracks.append((np.asarray(s_alt), np.asarray(s_az)))

    n_beams = beams["n_beams"]
    min_dist = np.full(n_beams, np.inf)
    for bi in range(n_beams):
        if bi in by_beam:
            continue   # not eligible for control duty
        cos_alt = np.cos(np.deg2rad(bn_alt[bi]))
        for s_alt, s_az in src_tracks:
            # Only consider time samples where the source is above the
            # horizon — below-horizon "tracks" pile near alt=0 and
            # spuriously near beams at low alt.
            up = s_alt > 0
            if not np.any(up):
                continue
            d_alt = s_alt[up] - bn_alt[bi]
            d_az = ((s_az[up] - bn_az[bi] + 180.0) % 360.0) - 180.0
            d_az_sky = d_az * cos_alt
            r = np.sqrt((d_az_sky / half_ew) ** 2 + (d_alt / half_ns) ** 2)
            min_dist[bi] = min(min_dist[bi], float(r.min()))

    # Pick the top-n beams by largest min-distance.
    eligible = [bi for bi in range(n_beams) if bi not in by_beam and np.isfinite(min_dist[bi])]
    eligible.sort(key=lambda bi: -min_dist[bi])
    control_idxs = sorted(eligible[:n_control_beams])
    return hit_idxs, control_idxs


def validate_beam_weights(
    int8_h5,
    data,
    ant,
    *,
    cal_weights,
    sources: Iterable[str] = ("sun", "cas-a", "cyg-a", "tau-a"),
    freq_band_mhz: tuple = (405.0, 433.0),
    max_hit_panels: int = 12,
    n_control_beams: int = 2,
    fwhm_factor: float = 1.0,
    pass_ratio: float = 5.0,
) -> dict:
    """End-to-end beam-weights validation.

    For every (source, beam) the source crosses through during the
    visibility window, beamform the in-memory vis at the beam's exact
    pointing (using ``cal_weights``) and compare peak in-window power
    to median out-of-window power.

    Parameters
    ----------
    int8_h5 : path-like
        Deployed SNAP int8 weights file.
    data : VisibilityResult
        Already-loaded visibilities (from
        :func:`casm_io.read_visibilities`). Must include ``vis``,
        ``freq_mhz``, ``time_unix``, and ideally ``freq_mask``.
    ant : :class:`casm_io.AntennaMapping`
    cal_weights : :class:`bf_weights_generator.CalibrationWeights`
        Same calibration that was applied when the int8 weights were
        built. Per-antenna cal is applied during beamforming here.
    sources : iterable of str
        Source names to test. Default covers the 4 brightest CASM
        sources at OVRO.
    freq_band_mhz : (lo, hi)
        Cleanband for the freq-mean of the beam power.
    max_hit_panels : int
        Cap on the number of source-hit beams to beamform/plot.
        Beams ranked by total in-beam dwell time.
    n_control_beams : int
        Number of no-hit beams to also beamform, for null reference.
    fwhm_factor : float
        "In-beam" radius in units of half-FWHM (1.0 = full FWHM).
    pass_ratio : float
        A beam passes if peak_in_window / median_out_of_window >=
        pass_ratio.

    Returns
    -------
    dict with keys
        ``beams``, ``hits``, ``selected_beam_idxs``,
        ``control_beam_idxs``, ``power``, ``time_unix``,
        ``freq_band_used_mhz``, ``n_chan_used``, ``per_beam_metrics``.
    """
    beams = load_beams_from_int8(int8_h5)
    time_unix = np.asarray(data["time_unix"], dtype=np.float64)
    hits = find_source_beam_transits(
        beams, list(sources), time_unix, fwhm_factor=fwhm_factor
    )

    sel_hits, sel_ctrl = _select_beams_for_validation(
        hits, beams, sources=list(sources), time_unix=time_unix,
        max_hit_panels=max_hit_panels,
        n_control_beams=n_control_beams,
    )
    sel_idxs = sel_hits + sel_ctrl

    # Build the (label, alt, az) triples that beam_power_vs_time wants.
    pointings = []
    for bi in sel_idxs:
        nm = beams["names"][bi] if bi < len(beams["names"]) else f"beam_{bi}"
        pointings.append((f"beam_{bi}_{nm}", float(beams["alt_deg"][bi]), float(beams["az_deg"][bi])))

    bp = beam_power_vs_time(
        data, ant,
        sources=pointings,
        cal_weights=cal_weights,
        freq_band_mhz=freq_band_mhz,
    )

    # Map labels back to beam indices.
    power_by_idx: dict[int, np.ndarray] = {}
    for bi, (label, _, _) in zip(sel_idxs, pointings):
        power_by_idx[bi] = np.asarray(bp["power"][label], dtype=np.float64)

    # Per-beam metrics.
    metrics: dict[int, dict] = {}
    for bi in sel_idxs:
        p = power_by_idx[bi]
        beam_hits = [h for h in hits if h.beam_idx == bi]
        if beam_hits:
            in_window = np.zeros(len(time_unix), dtype=bool)
            for h in beam_hits:
                in_window |= (time_unix >= h.entry_unix) & (time_unix <= h.exit_unix)
            peak_in = float(p[in_window].max()) if in_window.any() else float("nan")
            med_out = float(np.median(p[~in_window])) if (~in_window).any() else float("nan")
            ratio = peak_in / abs(med_out) if med_out and np.isfinite(med_out) else float("inf")
            metrics[bi] = {
                "expected_hit": True,
                "peak_in_window": peak_in,
                "median_out_window": med_out,
                "ratio": ratio,
                "pass": bool(ratio >= pass_ratio),
                "sources": sorted({h.source for h in beam_hits}),
            }
        else:
            peak_abs = float(np.max(np.abs(p)))
            med = float(np.median(p))
            metrics[bi] = {
                "expected_hit": False,
                "peak_abs": peak_abs,
                "median": med,
                "pass": True,    # no expectation
                "sources": [],
            }

    return {
        "beams": beams,
        "hits": hits,
        "selected_beam_idxs": sel_hits,
        "control_beam_idxs": sel_ctrl,
        "power": power_by_idx,
        "time_unix": time_unix,
        "freq_band_used_mhz": bp["freq_band_used_mhz"],
        "n_chan_used": bp["n_chan_used"],
        "per_beam_metrics": metrics,
    }


def plot_beam_validation(result: Mapping, *, output_path=None, time_tz="UTC"):
    """Multi-panel figure: power(t) per selected beam with transit windows.

    One subplot per selected beam (source-hit beams first, then any
    controls). Per-source colored shaded regions show predicted in-beam
    intervals; vertical line marks the predicted peak; pass/fail badge
    in the corner. Returns a matplotlib Figure.

    ``time_tz`` is an IANA timezone name (e.g. ``"America/Los_Angeles"``)
    or ``"UTC"``; the x-axis is DST-aware and labeled with the actual
    abbreviation in effect (PDT/PST/UTC).
    """
    import matplotlib.pyplot as plt
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(time_tz) if time_tz != "UTC" else timezone.utc

    beams = result["beams"]
    hits = result["hits"]
    sel = list(result["selected_beam_idxs"]) + list(result["control_beam_idxs"])
    metrics = result["per_beam_metrics"]
    times_unix = np.asarray(result["time_unix"])
    times_dt = [datetime.fromtimestamp(t, tz=tz) for t in times_unix]

    # Per-source color map for the shaded transit windows.
    src_colors: dict[str, str] = {}
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    sources_seen: list[str] = []
    for h in hits:
        if h.source not in src_colors:
            src_colors[h.source] = palette[len(src_colors) % len(palette)]
            sources_seen.append(h.source)

    n = len(sel)
    if n == 0:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No beams selected for validation",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    ncols = 2 if n > 4 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 2.4 * nrows),
                             sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax_i, bi in enumerate(sel):
        ax = axes_flat[ax_i]
        p = result["power"][bi]
        m = metrics[bi]
        beam_alt = beams["alt_deg"][bi]
        beam_az = beams["az_deg"][bi]
        beam_nm = beams["names"][bi] if bi < len(beams["names"]) else f"beam_{bi}"

        ax.plot(times_dt, p, color="0.2", lw=1.2)

        beam_hits = [h for h in hits if h.beam_idx == bi]
        for h in beam_hits:
            entry_dt = datetime.fromtimestamp(h.entry_unix, tz=tz)
            exit_dt = datetime.fromtimestamp(h.exit_unix, tz=tz)
            peak_dt = datetime.fromtimestamp(h.peak_unix, tz=tz)
            color = src_colors[h.source]
            ax.axvspan(entry_dt, exit_dt, color=color, alpha=0.18)
            ax.axvline(peak_dt, color=color, lw=1, alpha=0.6, ls="--")

        if m.get("expected_hit"):
            badge = ("PASS" if m["pass"] else "FAIL") + f"  ratio={m['ratio']:.1f}"
            badge_color = "C2" if m["pass"] else "C3"
        else:
            badge = "control"
            badge_color = "0.5"
        ax.text(0.99, 0.95,
                f"beam {bi}  alt={beam_alt:.0f}° az={beam_az:.0f}°\n{badge}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=badge_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=badge_color, alpha=0.9))
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("power")

    for ax in axes_flat[n:]:
        ax.axis("off")
    tz_label = (times_dt[0].strftime("%Z") if times_dt else "") or "UTC"
    for ax in axes_flat[-ncols:]:
        if ax.get_visible():
            ax.set_xlabel(f"{tz_label} time")

    # Top legend: one entry per source seen.
    if sources_seen:
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(color=src_colors[s], alpha=0.4, label=s) for s in sources_seen
        ]
        fig.legend(handles=legend_handles, loc="upper center",
                   ncol=len(sources_seen), fontsize=9, frameon=False)

    fig.suptitle(
        f"Beam-weights validation: {result['n_chan_used']} chan, "
        f"{result['freq_band_used_mhz'][0]:.1f}-{result['freq_band_used_mhz'][1]:.1f} MHz",
        y=0.995,
    )
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if output_path is not None:
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------
# Per-source variant. Cleaner output than ``validate_beam_weights`` when
# you just want to ask "did this one source produce a visible transit
# through every beam it was supposed to cross?"
# ---------------------------------------------------------------------


def validate_source(
    int8_h5,
    data,
    ant,
    *,
    source: str,
    cal_weights,
    freq_band_mhz: tuple | None = None,
    n_control_beams: int = 1,
    fwhm_factor: float = 1.0,
    pass_ratio: float = 5.0,
) -> dict:
    """One-source variant of :func:`validate_beam_weights`.

    Identifies every beam the named source crosses through during the
    visibility window, beamforms the in-memory vis at each beam's exact
    pointing using ``cal_weights``, and adds one or more far-from-track
    control beams (zero source contamination) for null reference.

    Parameters
    ----------
    int8_h5 : path-like
        Deployed SNAP int8 weights file.
    data : VisibilityResult
        Already-loaded visibilities. Must include ``vis``, ``freq_mhz``,
        ``time_unix``; ``freq_mask`` is honoured if present.
    ant : :class:`casm_io.AntennaMapping`
    source : str
        One source name, e.g. ``"sun"``, ``"cas-a"``, ``"cyg-a"``,
        ``"tau-a"``.
    cal_weights : :class:`bf_weights_generator.CalibrationWeights`
    freq_band_mhz, n_control_beams, fwhm_factor, pass_ratio :
        See :func:`validate_beam_weights`.

    Returns
    -------
    dict with keys
        ``source`` : str
        ``beams`` : output of :func:`load_beams_from_int8`
        ``hits`` : list of :class:`BeamHit` (only this source)
        ``hit_beam_idxs`` : list of beam indices the source crosses
        ``control_beam_idxs`` : list of control beam indices
        ``power`` : dict[beam_idx -> ndarray(T,)]
        ``per_beam_metrics`` : dict[beam_idx -> dict]
        ``time_unix`` : ndarray(T,)
        ``freq_band_used_mhz`` : (lo, hi)
        ``n_chan_used`` : int
    """
    beams = load_beams_from_int8(int8_h5)
    time_unix = np.asarray(data["time_unix"], dtype=np.float64)

    hits = find_source_beam_transits(
        beams, [source], time_unix, fwhm_factor=fwhm_factor
    )
    # Hit beams sorted CHRONOLOGICALLY by earliest entry time —
    # reads top-to-bottom in time order in the per-beam panels.
    by_beam: dict[int, list[BeamHit]] = {}
    for h in hits:
        by_beam.setdefault(h.beam_idx, []).append(h)
    hit_beam_idxs = sorted(
        by_beam.keys(),
        key=lambda bi: min(h.entry_unix for h in by_beam[bi]),
    )

    # Far-from-track controls (no source contamination).
    _, ctrl_beam_idxs = _select_beams_for_validation(
        hits, beams, sources=[source], time_unix=time_unix,
        max_hit_panels=0,             # we don't need its hit picks
        n_control_beams=n_control_beams,
    )

    pointings = []
    sel_idxs = list(hit_beam_idxs) + list(ctrl_beam_idxs)
    for bi in sel_idxs:
        pointings.append(
            (f"beam_{bi}",
             float(beams["alt_deg"][bi]),
             float(beams["az_deg"][bi]))
        )
    if not pointings:
        # Source never crosses any beam during the window. Still pick
        # control beams so the user can see the noise floor.
        return {
            "source": source,
            "beams": beams,
            "hits": [],
            "hit_beam_idxs": [],
            "control_beam_idxs": list(ctrl_beam_idxs),
            "power": {},
            "per_beam_metrics": {},
            "time_unix": time_unix,
            "freq_band_used_mhz": tuple(freq_band_mhz) if freq_band_mhz is not None else None,
            "n_chan_used": 0,
        }

    bp = beam_power_vs_time(
        data, ant, sources=pointings,
        cal_weights=cal_weights, freq_band_mhz=freq_band_mhz,
    )

    power_by_idx: dict[int, np.ndarray] = {}
    for bi in sel_idxs:
        power_by_idx[bi] = np.asarray(bp["power"][f"beam_{bi}"], dtype=np.float64)

    # Cross-baseline coherent sums carry a per-beam DC bias that
    # varies with pointing direction (the natural visibility's DC
    # gets rotated by the per-baseline phase, summing to a pointing-
    # dependent offset). For bright calibrators that bias is small
    # next to the source signal; for faint sources (Cas A, Tau A) it
    # dominates the absolute power. So the metric works on the
    # *excursion above the per-beam baseline*, not the absolute
    # power.
    #
    #   peak_excursion_in_window = max(p[in]) - median(p[out])
    #   noise_excursion_out      = robust_std of (p[out] - median)
    #   ratio = peak_excursion_in / noise_excursion_out
    #
    # This is direction-invariant and self-calibrating per beam.
    metrics: dict[int, dict] = {}
    for bi in sel_idxs:
        p = power_by_idx[bi]
        beam_hits = [h for h in hits if h.beam_idx == bi]
        if beam_hits:
            in_window = np.zeros(len(time_unix), dtype=bool)
            for h in beam_hits:
                in_window |= (time_unix >= h.entry_unix) & (time_unix <= h.exit_unix)
            if not in_window.any() or not (~in_window).any():
                metrics[bi] = {
                    "expected_hit": True, "ratio": float("nan"),
                    "pass": False, "reason": "in/out window empty",
                }
                continue
            p_in = p[in_window]
            p_out = p[~in_window]
            baseline = float(np.median(p_out))                 # per-beam DC
            peak_excursion = float(p_in.max() - baseline)
            # Robust std (1.4826 * MAD) of the out-of-window values.
            mad_out = float(np.median(np.abs(p_out - baseline)))
            noise = max(1.4826 * mad_out, 1e-30)
            ratio = peak_excursion / noise
            metrics[bi] = {
                "expected_hit": True,
                "peak_in_window": float(p_in.max()),
                "median_out_window": baseline,
                "peak_excursion": peak_excursion,
                "noise_mad": noise,
                "ratio": ratio,                                # excursion-σ
                "pass": bool(ratio >= pass_ratio),
            }
        else:
            baseline = float(np.median(p))
            mad = float(np.median(np.abs(p - baseline)))
            noise = max(1.4826 * mad, 1e-30)
            metrics[bi] = {
                "expected_hit": False,
                "peak_abs": float(np.max(np.abs(p))),
                "median": baseline,
                "noise_mad": noise,
                "pass": True,
            }

    return {
        "source": source,
        "beams": beams,
        "hits": hits,
        "hit_beam_idxs": list(hit_beam_idxs),
        "control_beam_idxs": list(ctrl_beam_idxs),
        "power": power_by_idx,
        "per_beam_metrics": metrics,
        "time_unix": time_unix,
        "freq_band_used_mhz": bp["freq_band_used_mhz"],
        "n_chan_used": bp["n_chan_used"],
    }


def _draw_zenith_projection(
    ax,
    beams: Mapping,
    source: str,
    time_unix: np.ndarray,
    *,
    highlight_beams: Iterable[int],
    control_beams: Iterable[int],
):
    """Zenithal projection: all beams as ellipses (faded), the source's
    track over ``time_unix`` as a colored line, hit beams highlighted,
    control beams outlined."""
    from matplotlib.patches import Ellipse, Patch
    from matplotlib.lines import Line2D

    bn_alt = np.asarray(beams["alt_deg"], dtype=np.float64)
    bn_az = np.asarray(beams["az_deg"], dtype=np.float64)
    fwhm_ew = beams["fwhm_ew_deg"]
    fwhm_ns = beams["fwhm_ns_deg"]

    za = 90.0 - bn_alt
    az_rad = np.deg2rad(bn_az)
    xs = za * np.sin(az_rad)
    ys = za * np.cos(az_rad)

    margin = max(fwhm_ew, fwhm_ns) * 1.2
    lim = max(np.max(np.abs(xs)), np.max(np.abs(ys)), 90.0) + margin

    theta = np.linspace(0, 2 * np.pi, 256)
    for alt in [15, 30, 45, 60, 75]:
        r = 90 - alt
        if r <= lim * 1.2:
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color="gray", lw=0.5, ls="--", alpha=0.4)
            ax.text(0.4, -r - 0.3, f"{alt}°", ha="left", va="top",
                    fontsize=7, color="gray", alpha=0.6)
    for az in np.arange(0, 360, 45):
        a_r = np.deg2rad(az)
        ax.plot([0, lim * np.sin(a_r)], [0, lim * np.cos(a_r)],
                color="gray", lw=0.5, ls="--", alpha=0.4)

    off = lim + 1.1
    ax.text(0, off, "N", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.text(off, 0, "E", ha="left", va="center", fontsize=12, fontweight="bold")
    ax.text(0, -off, "S", ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(-off, 0, "W", ha="right", va="center", fontsize=12, fontweight="bold")

    hit_set = set(int(b) for b in highlight_beams)
    ctrl_set = set(int(b) for b in control_beams)

    for i in range(len(bn_alt)):
        if za[i] < fwhm_ns:
            w = h = max(fwhm_ew, fwhm_ns); angle = 0
        else:
            w = fwhm_ew; h = fwhm_ns; angle = -bn_az[i]
        if i in hit_set:
            face, edge, alpha, lw, z = "C0", "C0", 0.50, 1.6, 5
        elif i in ctrl_set:
            face, edge, alpha, lw, z = "none", "C2", 1.0, 2.2, 6
        else:
            face, edge, alpha, lw, z = "white", "0.65", 0.55, 0.6, 1
        ax.add_patch(Ellipse((xs[i], ys[i]), width=w, height=h, angle=angle,
                             facecolor=face, edgecolor=edge,
                             alpha=alpha, lw=lw, zorder=z))
        # Hits/controls always get a bold black number. For the
        # background grid, only label every beam if the grid is small
        # enough that labels won't overlap (<= 64). For larger grids,
        # leave the background unlabeled — the highlighted ones still
        # carry the meaningful numbers.
        if i in hit_set or i in ctrl_set:
            ax.annotate(str(i), (xs[i], ys[i]), fontsize=10,
                        ha="center", va="center",
                        fontweight="bold", color="k", zorder=10)
        elif len(bn_alt) <= 64:
            ax.annotate(str(i), (xs[i], ys[i]), fontsize=6,
                        ha="center", va="center",
                        color="0.4", zorder=2)

    s_alt, s_az = source_altaz(source, time_unix)
    s_alt = np.asarray(s_alt, dtype=np.float64)
    s_az = np.asarray(s_az, dtype=np.float64)
    above = s_alt > 0
    if np.any(above):
        s_za = 90.0 - s_alt
        s_az_rad = np.deg2rad(s_az)
        sx = s_za * np.sin(s_az_rad)
        sy = s_za * np.cos(s_az_rad)
        idxs = np.where(above)[0]
        if len(idxs) > 1:
            d_az = np.abs(np.diff(s_az[idxs]))
            breaks = np.where(d_az > 90)[0] + 1
            segs = np.split(idxs, breaks)
        else:
            segs = [idxs]
        for j, seg in enumerate(segs):
            if len(seg) < 2:
                continue
            ax.plot(sx[seg], sy[seg], color="C3", lw=2.2, alpha=0.9,
                    label=f"{source} track" if j == 0 else None)
            # Direction arrows: 3 evenly spaced along the segment so
            # the eye picks up the start->end direction at a glance.
            seg_len = len(seg)
            if seg_len >= 4:
                marker_idxs = [int(seg_len * f) for f in (0.25, 0.55, 0.85)]
                for mi_local in marker_idxs:
                    if mi_local + 1 >= seg_len:
                        continue
                    p_idx = seg[mi_local]
                    n_idx = seg[mi_local + 1]
                    ax.annotate(
                        "",
                        xy=(sx[n_idx], sy[n_idx]),
                        xytext=(sx[p_idx], sy[p_idx]),
                        arrowprops=dict(
                            arrowstyle="->", color="C3",
                            lw=1.8, alpha=0.95,
                            mutation_scale=18,
                        ),
                        zorder=12,
                    )
        # Mark start/end of the track inside the data window. Start =
        # green circle (filled). End = red square. Pairs with the
        # arrows so the direction is unambiguous.
        ax.plot(sx[idxs[0]], sy[idxs[0]], "o", color="#0a8a0a",
                ms=8, mec="k", mew=0.8, zorder=13,
                label="window start")
        ax.plot(sx[idxs[-1]], sy[idxs[-1]], "s", color="#c01010",
                ms=8, mec="k", mew=0.8, zorder=13,
                label="window end")

    ax.plot(0, 0, "r+", markersize=10, markeredgewidth=2, zorder=10)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("← W   E →")
    ax.set_ylabel("← S   N →")

    legend_elements = [
        Line2D([0], [0], color="C3", lw=2, label=f"{source} track"),
        Line2D([0], [0], marker="o", color="#0a8a0a", lw=0,
               markeredgecolor="k", markersize=8, label="start (window)"),
        Line2D([0], [0], marker="s", color="#c01010", lw=0,
               markeredgecolor="k", markersize=8, label="end (window)"),
        Patch(facecolor="C0", alpha=0.5, edgecolor="C0", lw=1.5,
              label="hit beams"),
        Patch(facecolor="none", edgecolor="C2", lw=2, label="control beam"),
        Patch(facecolor="lightgray", alpha=0.25, edgecolor="gray",
              label="other beams"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              fontsize=8, framealpha=0.9)


def plot_source_validation(
    result: Mapping,
    *,
    time_tz: str = "America/Los_Angeles",
    output_path=None,
):
    """Side-by-side figure for :func:`validate_source`'s output.

    Left column: zenithal projection showing the source's track over
    the data window with the beams it crosses highlighted (and the
    control beam(s) outlined).

    Right column: one panel per highlighted/control beam, stacked
    vertically. X-axis is local time in ``time_tz`` (default OVRO
    Pacific). Each panel:

    * power(t) line in black
    * source transit window shaded (blue)
    * predicted peak time as a dashed vertical line
    * panel title: ``beam N  alt=X°  az=Y°    [PASS/FAIL/CONTROL]``

    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(time_tz)
    source = result["source"]
    beams = result["beams"]
    hits = result["hits"]
    hit_idxs = list(result["hit_beam_idxs"])
    ctrl_idxs = list(result["control_beam_idxs"])
    panels = hit_idxs + ctrl_idxs
    n_panels = max(len(panels), 1)

    # Make the figure wide enough that the zenith projection on the
    # left is large enough to read every beam's number, and tall
    # enough that each per-beam panel on the right has breathing room.
    panel_h = 2.2
    fig_h = max(11.0, panel_h * n_panels + 1.8)
    fig = plt.figure(figsize=(20.0, fig_h))
    gs = GridSpec(n_panels, 2, width_ratios=[1.4, 1.4], figure=fig,
                  wspace=0.18, hspace=0.40)

    ax_zen = fig.add_subplot(gs[:, 0])
    _draw_zenith_projection(
        ax_zen, beams, source, result["time_unix"],
        highlight_beams=hit_idxs, control_beams=ctrl_idxs,
    )
    win_start = datetime.fromtimestamp(result["time_unix"][0], tz=tz)
    win_end = datetime.fromtimestamp(result["time_unix"][-1], tz=tz)
    ax_zen.set_title(
        f"{source.upper()} track + beam grid\n"
        f"{win_start:%Y-%m-%d %H:%M}  to  {win_end:%H:%M} {time_tz}",
        fontsize=11,
    )

    if not panels:
        # Nothing crossed any beam — show the zenith projection only.
        fig.suptitle(
            f"Beam validation — {source.upper()}: no beams crossed during window",
            fontsize=12, fontweight="bold", y=0.995,
        )
        if output_path is not None:
            fig.savefig(output_path, dpi=140, bbox_inches="tight")
        return fig

    times_dt = [datetime.fromtimestamp(t, tz=tz) for t in result["time_unix"]]

    for k, bi in enumerate(panels):
        ax = fig.add_subplot(gs[k, 1])
        p = result["power"][bi]
        ax.plot(times_dt, p, color="0.15", lw=1.2)
        beam_alt = beams["alt_deg"][bi]
        beam_az = beams["az_deg"][bi]
        is_ctrl = bi in ctrl_idxs

        if not is_ctrl:
            for h in [h for h in hits if h.beam_idx == bi]:
                entry = datetime.fromtimestamp(h.entry_unix, tz=tz)
                exit_ = datetime.fromtimestamp(h.exit_unix, tz=tz)
                peak = datetime.fromtimestamp(h.peak_unix, tz=tz)
                ax.axvspan(entry, exit_, color="C0", alpha=0.20,
                           label=f"{source} in-beam")
                ax.axvline(peak, color="C0", lw=1, ls="--", alpha=0.7)
            m = result["per_beam_metrics"][bi]
            badge = ("PASS" if m["pass"] else "FAIL") + f"  ratio={m['ratio']:.1f}"
            badge_color = "C2" if m["pass"] else "C3"
        else:
            badge = "CONTROL"
            badge_color = "0.4"

        ax.set_title(
            f"beam {bi}   alt={beam_alt:.0f}°  az={beam_az:.0f}°    [{badge}]",
            color=badge_color, fontsize=10, loc="left",
        )
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("power")

        if k < n_panels - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.set_xlabel(f"local time ({time_tz})")
        # Local-time axis formatter
        from matplotlib.dates import DateFormatter
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=tz))

    fig.suptitle(
        f"Beam validation — {source.upper()} — "
        f"{result['n_chan_used']} chan, "
        f"{result['freq_band_used_mhz'][0]:.1f}-"
        f"{result['freq_band_used_mhz'][1]:.1f} MHz",
        fontsize=12, fontweight="bold", y=0.995,
    )
    if output_path is not None:
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    return fig


def print_source_validation_summary(
    result: Mapping,
    *,
    time_tz: str = "America/Los_Angeles",
) -> None:
    """One-line-per-beam summary of which beams the source crossed and
    whether the visibility-domain beam picked up a real transit.
    Times printed in ``time_tz`` (default OVRO Pacific)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(time_tz)
    src = result["source"]
    hits = result["hits"]
    print(f"{src.upper()} crosses {len(result['hit_beam_idxs'])} beams "
          f"during the data window:")
    for bi in result["hit_beam_idxs"]:
        m = result["per_beam_metrics"][bi]
        beam_hits = [h for h in hits if h.beam_idx == bi]
        for h in beam_hits:
            entry = datetime.fromtimestamp(h.entry_unix, tz=tz)
            exit_ = datetime.fromtimestamp(h.exit_unix, tz=tz)
            peak = datetime.fromtimestamp(h.peak_unix, tz=tz)
            tag = "PASS" if m["pass"] else "FAIL"
            beams = result["beams"]
            print(f"  [{tag}] beam {bi:3d}  "
                  f"alt={beams['alt_deg'][bi]:5.1f}°  "
                  f"az={beams['az_deg'][bi]:5.1f}°   "
                  f"in-beam {entry:%H:%M}–{exit_:%H:%M}  "
                  f"peak {peak:%H:%M}  "
                  f"ratio={m['ratio']:6.2f}")
    if result["control_beam_idxs"]:
        print(f"Control beams (far from {src} track):")
        for bi in result["control_beam_idxs"]:
            m = result["per_beam_metrics"][bi]
            beams = result["beams"]
            print(f"  [ctrl] beam {bi:3d}  "
                  f"alt={beams['alt_deg'][bi]:5.1f}°  "
                  f"az={beams['az_deg'][bi]:5.1f}°   "
                  f"peak_abs={m['peak_abs']:.2e}")


def validate_source_at_time(
    int8_h5,
    cal_weights,
    *,
    source: str,
    time_start,
    time_end,
    time_tz: str = "America/Los_Angeles",
    layout=None,
    inactive_antennas: Iterable[int] = (),
    data_root: str = "/mnt",
    data_dir=None,
    fmt="layout_64ant",
    freq_band_mhz: tuple | None = None,
    n_control_beams: int = 1,
    rfi_mask_version: int = 2,
    fwhm_factor: float = 1.0,
    pass_ratio: float = 5.0,
    verbose: bool = False,
) -> dict:
    """Standalone per-source validation: read fresh visibilities for
    an arbitrary date/time range, apply previously-saved cal weights,
    and run :func:`validate_source` against a deployed int8 weights
    file.

    Use this to:

    * Apply a Sun-derived cal to a *night-time* window and check Cas A /
      Cyg A / Tau A beams.
    * Test cal stability over hours/days — load yesterday's cal, point
      at today's data, see if the beams still light up.
    * Sweep across multiple sources without re-running fringe-stop /
      SVD: same cal, different time slices.

    The function delegates to ``casm_io.read_visibilities`` for data
    discovery (auto-scans ``data_root`` for ``visibilities_*`` dirs),
    ``casm_vis_analysis.RFIMask.from_static`` for masking, and
    :func:`validate_source` for the actual validation. No new
    primitives — this is composition only.

    Parameters
    ----------
    int8_h5 : path-like
        Deployed SNAP int8 weights file (the beams to validate).
    cal_weights : path-like or :class:`bf_weights_generator.CalibrationWeights`
        Either a path to a cal HDF5 (loaded via
        :func:`bf_weights_generator.load_calibration_weights`) or an
        already-built ``CalibrationWeights`` instance.
    source : str
        Source name (e.g. ``"sun"``, ``"cas-a"``, ``"cyg-a"``, ``"tau-a"``).
    time_start, time_end : str or datetime
        Window to read.  Strings accepted by ``casm_io.read_visibilities``.
    time_tz : str
        Timezone for ``time_start``/``time_end``. Default OVRO Pacific.
    layout : path-like, optional
        Antenna layout CSV for :class:`casm_io.AntennaMapping`. If None,
        falls back to ``$CASM_LAYOUT_CSV`` / canonical resolver.
    inactive_antennas : iterable of int
        Antenna IDs to disable at runtime via
        :meth:`AntennaMapping.with_inactive`.
    data_root, data_dir, fmt :
        Forwarded to :func:`casm_io.read_visibilities`.
    freq_band_mhz, n_control_beams, fwhm_factor, pass_ratio :
        Forwarded to :func:`validate_source`.
    rfi_mask_version : int
        Static RFI mask version applied to the freshly-read data.
    verbose : bool
        Show casm_io's data-discovery progress.

    Returns
    -------
    dict — same shape as :func:`validate_source`.
    """
    from casm_io.correlator import (
        AntennaMapping, load_format, read_visibilities,
    )
    from .rfi import RFIMask, apply_rfi_mask

    fmt_obj = load_format(fmt) if isinstance(fmt, str) else fmt

    ant = AntennaMapping.load(layout)
    if inactive_antennas:
        ant = ant.with_inactive(list(inactive_antennas))

    data = read_visibilities(
        time_start=time_start,
        time_end=time_end,
        time_tz=time_tz,
        data_root=data_root,
        data_dir=data_dir,
        fmt=fmt_obj,
        verbose=verbose,
    )
    apply_rfi_mask(data, RFIMask.from_static(version=rfi_mask_version))

    if isinstance(cal_weights, (str, Path)):
        from bf_weights_generator import load_calibration_weights
        cal_obj = load_calibration_weights(str(cal_weights))
    else:
        cal_obj = cal_weights

    return validate_source(
        int8_h5=int8_h5,
        data=data, ant=ant, cal_weights=cal_obj,
        source=source,
        freq_band_mhz=freq_band_mhz,
        n_control_beams=n_control_beams,
        fwhm_factor=fwhm_factor,
        pass_ratio=pass_ratio,
    )
