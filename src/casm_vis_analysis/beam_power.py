"""Beam-power-vs-time diagnostic for source-tracking validation.

Given a visibility dataset (already in memory), an :class:`AntennaMapping`,
and per-antenna calibration weights, compute the coherent beam power
through time for each of several source-tracking pointings (or fixed
control directions) and a single broadband channel-averaged number.

The intended use: after running SVD calibration toward Sun, apply those
gains to the same data and beamform toward Sun, Cas A, Cyg A, Tau A,
and a fixed off-source control direction. The Sun beam should be
strong while Sun is up; the others should sit near the noise floor.
This is the cheapest "did the calibration phase the array up correctly"
test before the B0329+54 pulsar fold.

Math
----
For active antennas (active subset after `with_inactive`), with cal
weight ``c_k(f)`` and per-antenna geometric phasor toward the source,
``g_k(t, f) = exp(+2*pi*i * f * tau_k(t))`` (where
``tau_k(t) = pos_k . s_hat(t) / c``), the F-engine convention is
``b(t, f) = sum_k w_k(t, f) * v_k(t, f)`` with
``w_k(t, f) = c_k(f) * g_k(t, f)``. Beam power is
``P(t, f) = |b|^2 = sum_ij w_i conj(w_j) V_ij``.

Splitting auto and cross:
``P_cross(t, f) = 2 * Re(sum_{i<j} w_i conj(w_j) V_ij)``
``P_auto(t, f)  = sum_i |w_i|^2 V_ii  =  sum_i V_ii``  (phase-only weights)

The cross-baseline term is what changes with pointing direction. We
return the freq-mean of ``P_cross(t, f)`` over good channels in the
specified band — that's the "beam intensity" at each time.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from casm_io.constants import C_LIGHT_M_S
from casm_io.correlator.baselines import triu_flat_index

from .sources import source_altaz, source_enu


def beam_power_vs_time(
    data,
    ant,
    sources: Iterable,
    *,
    cal_weights=None,
    freq_band_mhz: tuple | None = None,
    sign: int = -1,
) -> dict:
    """Coherent beam power per time sample for each source-tracking pointing.

    Parameters
    ----------
    data : VisibilityResult or dict-like
        Output of :func:`casm_io.correlator.read_visibilities`. Must
        carry ``vis`` (T, F, n_bl), ``freq_mhz`` (F,), ``time_unix`` (T,),
        and optionally ``freq_mask`` (F,) which is honoured if present.
    ant : :class:`casm_io.AntennaMapping`
        Active set determines which antennas participate. ``with_inactive``
        overrides apply.
    sources : iterable
        Each entry is either:

        * a string source name resolved via :func:`source_altaz`
          (``"sun"``, ``"cas_a"``, ``"tau_a"``, ``"cyg_a"``, ``"cas-a"``...)
        * a tuple ``(label, alt_deg, az_deg)`` for a stationary
          control beam at a fixed pointing.

        Mixing the two forms is allowed.
    cal_weights : :class:`bf_weights_generator.CalibrationWeights`, optional
        Per-antenna gains. If given, they're applied to the visibilities
        as ``V_ij' = cal_i(f) * conj(cal_j(f)) * V_ij(f)``. Cal frequencies
        are assumed to align with ``data['freq_mhz']`` (descending) — the
        usual case after the SVD pipeline. Mismatched ordering is
        handled by flipping ascending cal to descending.
    freq_band_mhz : tuple, optional
        ``(lo, hi)`` MHz inclusive. Restricts the freq-mean to channels
        in this band. Default uses every unflagged channel.
    sign : int, default -1
        Sign of the per-baseline geometric phase. ``-1`` cancels the
        natural baseline phase ``exp(+2*pi*i * f * (b_ij . s_hat)/c)``
        so the cross-baseline sum adds coherently for the source
        direction. Matches :func:`fringe_stop`'s default. Use ``+1`` to
        explicitly look for the source's anti-phase image.

    Returns
    -------
    dict with keys:
        ``time_unix`` : (T,) float64
        ``power`` : dict[str, ndarray(T,)] — per-source coherent beam
            power (cross-only, freq-averaged in band), real-valued.
        ``alt_deg`` : dict[str, ndarray(T,)] — pointing altitude per
            time sample. For named sources this tracks the source;
            for fixed pointings it's constant.
        ``az_deg``  : dict[str, ndarray(T,)] — same for azimuth.
        ``freq_band_used_mhz`` : tuple — ``(min, max)`` of channels
            actually averaged after applying RFI mask + freq_band cut.
        ``n_chan_used`` : int

    Notes
    -----
    * Cross-only power is direction-dependent; auto-power isn't, so
      it's omitted. The returned series is therefore *not* the full
      tied-array power — it's the part that responds to pointing.
    * Memory: builds a per-baseline cal'd vis array of shape
      ``(T, F_band, n_baselines_active)`` complex64. For a 21-antenna
      active set, 6 h at 1093 channels, that's ~70 MB.
    """
    # ``data`` is either a VisibilityResult (attribute access) or a plain
    # dict; pull the four fields uniformly via getattr-with-fallback.
    def _field(name, default=...):
        if hasattr(data, name):
            return getattr(data, name)
        try:
            return data[name]
        except (KeyError, TypeError):
            if default is ...:
                raise ValueError(f"data missing required field {name!r}")
            return default

    vis = np.asarray(_field("vis"))
    freq_mhz_all = np.asarray(_field("freq_mhz"), dtype=np.float64)
    time_unix = np.asarray(_field("time_unix"), dtype=np.float64)
    # Convention reminder: apply_rfi_mask() stores True=flagged(bad).
    # Invert here so ``good_chan_mask[i] is True`` => channel i is usable.
    raw_mask = _field("freq_mask", default=None)
    if raw_mask is None:
        good_chan_mask = np.ones(vis.shape[1], dtype=bool)
    else:
        good_chan_mask = ~np.asarray(raw_mask, dtype=bool)

    T, F, n_bl_total = vis.shape
    n_inputs = int((-1 + (1 + 8 * n_bl_total) ** 0.5) / 2)
    if n_inputs * (n_inputs + 1) // 2 != n_bl_total:
        raise ValueError(
            f"vis last axis ({n_bl_total}) is not a valid upper-triangle "
            f"count for any integer n_inputs."
        )

    # --- Channel mask: RFI + freq_band_mhz (both True=keep) ---
    chan_mask = good_chan_mask.copy()
    if freq_band_mhz is not None:
        lo, hi = sorted(freq_band_mhz)
        chan_mask &= (freq_mhz_all >= lo) & (freq_mhz_all <= hi)
    if not chan_mask.any():
        raise ValueError("No frequency channels survive the RFI mask + freq_band_mhz cut.")
    freq_mhz = freq_mhz_all[chan_mask]
    F_used = freq_mhz.size

    # --- Active antenna positions and the active cross-baseline list ---
    active = sorted(ant.active_antennas())
    n_active = len(active)
    if n_active < 2:
        raise ValueError(f"Need >=2 active antennas, got {n_active}")

    df = ant.dataframe
    positions = np.array([
        df.loc[df["antenna_id"] == a, ["x_m", "y_m", "z_m"]].values[0]
        for a in active
    ])  # (n_active, 3)
    pidx = np.array([ant.packet_index(a) for a in active], dtype=int)

    # Cross-only baselines: one (i, j) per pair with i < j (in the active set).
    bl_pairs = [(i, j) for i in range(n_active) for j in range(i + 1, n_active)]
    n_bl_active = len(bl_pairs)
    bl_flat_idx = np.empty(n_bl_active, dtype=np.int64)
    bl_conj = np.zeros(n_bl_active, dtype=bool)
    for k, (i, j) in enumerate(bl_pairs):
        p_i, p_j = int(pidx[i]), int(pidx[j])
        i_min, i_max = (p_i, p_j) if p_i <= p_j else (p_j, p_i)
        bl_flat_idx[k] = triu_flat_index(n_inputs, i_min, i_max)
        bl_conj[k] = (p_i > p_j)   # need to conjugate to get V_{i,j} from V_{j,i}

    # --- Slice vis to active baselines and good channels ---
    vis_band = vis[:, chan_mask, :][:, :, bl_flat_idx].astype(np.complex64)
    if bl_conj.any():
        vis_band[:, :, bl_conj] = np.conj(vis_band[:, :, bl_conj])

    # --- Apply per-antenna calibration to each baseline ---
    # cal weight for antenna k is cal_k(f) where f indexes the cal's freqs.
    # We assume cal freqs align channel-for-channel with data['freq_mhz'].
    # If cal freqs are ascending, flip to descending to match.
    if cal_weights is not None:
        cal_w = np.asarray(cal_weights.weights)              # (n_cal_ant, n_chan)
        cal_freqs = np.asarray(cal_weights.frequencies_hz)   # (n_chan,)
        if cal_freqs.size > 1 and cal_freqs[1] > cal_freqs[0]:
            cal_w = cal_w[:, ::-1]
        if cal_w.shape[1] != F:
            raise ValueError(
                f"cal_weights has {cal_w.shape[1]} channels but vis has {F}; "
                f"can't align."
            )
        # Length agreement is necessary but not sufficient — verify the
        # channel centres actually line up (catches band-shift mismatches
        # that silently produce garbage beams).
        cal_freqs_mhz_check = cal_freqs / 1e6
        if cal_freqs_mhz_check.size > 1 and cal_freqs_mhz_check[1] < cal_freqs_mhz_check[0]:
            cal_freqs_mhz_check = cal_freqs_mhz_check[::-1]
        data_freq_mhz_check = freq_mhz_all
        if data_freq_mhz_check.size > 1 and data_freq_mhz_check[1] < data_freq_mhz_check[0]:
            data_freq_mhz_check = data_freq_mhz_check[::-1]
        if not np.allclose(cal_freqs_mhz_check, data_freq_mhz_check, atol=0.01):
            raise ValueError(
                f"cal weights frequency axis doesn't match data: "
                f"cal range {cal_freqs_mhz_check[0]:.1f}-{cal_freqs_mhz_check[-1]:.1f} MHz vs "
                f"data range {data_freq_mhz_check[0]:.1f}-{data_freq_mhz_check[-1]:.1f} MHz"
            )
        cal_band = cal_w[:, chan_mask]  # (n_cal_ant, F_used)

        # Map active antenna_id -> cal row.
        cal_ant_ids = list(int(a) for a in cal_weights.ant_ids)
        cal_row_for_active = []
        for aid in active:
            try:
                cal_row_for_active.append(cal_ant_ids.index(int(aid)))
            except ValueError as exc:
                raise ValueError(
                    f"Active antenna {aid} not found in cal_weights.ant_ids "
                    f"{cal_ant_ids}"
                ) from exc
        cal_per_active = cal_band[cal_row_for_active]   # (n_active, F_used)

        # Per-baseline cal phasor: cal_i * conj(cal_j) for each (i, j) pair.
        i_idx = np.array([i for (i, j) in bl_pairs])
        j_idx = np.array([j for (i, j) in bl_pairs])
        cal_pair = cal_per_active[i_idx] * np.conj(cal_per_active[j_idx])
        # cal_pair shape: (n_bl, F_used). Broadcast to (T, F_used, n_bl)
        # so V_ij' = cal_i * conj(cal_j) * V_ij elementwise.
        vis_band *= cal_pair.T[None, :, :].astype(np.complex64)

    # --- Precompute per-baseline ENU baseline vectors b_ij = pos_j - pos_i ---
    bl_enu = np.array([positions[j] - positions[i] for (i, j) in bl_pairs])  # (n_bl, 3)

    # --- For each source, compute per-baseline geometric phase
    # exp(sign * 2*pi*i * f * tau_ij(t)) where tau_ij(t) = b_ij . s_hat(t) / c ---
    freq_hz = freq_mhz * 1e6                           # (F_used,)
    out_power: dict[str, np.ndarray] = {}
    out_alt: dict[str, np.ndarray] = {}
    out_az:  dict[str, np.ndarray] = {}

    for src in sources:
        if isinstance(src, str):
            label = src
            alt_deg, az_deg = source_altaz(src, time_unix)
            s_enu = source_enu(src, time_unix)            # (T, 3)
        else:
            label, alt0, az0 = src
            alt_deg = np.full(T, float(alt0))
            az_deg  = np.full(T, float(az0))
            alt_rad = np.deg2rad(alt_deg)
            az_rad  = np.deg2rad(az_deg)
            s_enu = np.column_stack([
                np.sin(az_rad) * np.cos(alt_rad),
                np.cos(az_rad) * np.cos(alt_rad),
                np.sin(alt_rad),
            ])
        # tau_ij(t) = (b_ij . s_hat(t)) / c, shape (T, n_bl)
        tau_ij = (s_enu @ bl_enu.T) / C_LIGHT_M_S      # (T, n_bl)
        # phasor[t, f, b] = exp(sign * 2*pi*i * freq_hz[f] * tau_ij[t, b])
        # Memory: (T, F_used, n_bl) complex64. We compute on the fly per
        # source so the peak is one of these tensors at a time.
        phase = sign * 2.0 * np.pi * freq_hz[None, :, None] * tau_ij[:, None, :]
        phasor = np.exp(1j * phase).astype(np.complex64)
        # Coherent cross-baseline sum: P_cross(t, f) = 2 * Re(sum_b phasor * vis)
        coh = np.sum(phasor * vis_band, axis=2)        # (T, F_used) complex
        p_cross = 2.0 * np.real(coh)                   # (T, F_used)
        # Average over freq in band (only good channels by construction).
        out_power[label] = np.mean(p_cross, axis=1)    # (T,)
        out_alt[label] = alt_deg
        out_az[label] = az_deg

    return {
        "time_unix": time_unix,
        "power": out_power,
        "alt_deg": out_alt,
        "az_deg": out_az,
        "freq_band_used_mhz": (float(freq_mhz.min()), float(freq_mhz.max())),
        "n_chan_used": int(F_used),
    }


def plot_beam_power(
    result: Mapping,
    *,
    ax=None,
    show_alt: bool = True,
    output_path=None,
    time_tz: str = "America/Los_Angeles",
    xlim_unix: tuple | None = None,
):
    """Plot the per-source power time series from :func:`beam_power_vs_time`.

    Adds vertical markers at each named source's transit (peak alt) when
    ``show_alt`` is True, and a twin-axis altitude trace.

    Parameters
    ----------
    time_tz : str
        IANA timezone for the x-axis. Defaults to OVRO local
        ("America/Los_Angeles"). Pass "UTC" to plot in UTC.
    xlim_unix : (float, float), optional
        Zoom the x-axis to this Unix-time window. ``None`` = show
        whatever ``result['time_unix']`` covers.

    Returns the Matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(time_tz)
    times_unix = np.asarray(result["time_unix"])
    times_dt = [datetime.fromtimestamp(t, tz=tz) for t in times_unix]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k, (label, p) in enumerate(result["power"].items()):
        ax.plot(times_dt, p, color=color_cycle[k % len(color_cycle)], label=label)
        if show_alt:
            alt = result["alt_deg"].get(label)
            if alt is not None and np.ptp(alt) > 1.0:
                # Mark the peak-altitude time for tracking sources.
                pk = int(np.argmax(alt))
                ax.axvline(times_dt[pk], color=color_cycle[k % len(color_cycle)],
                           ls="--", lw=0.8, alpha=0.5)

    # Format ticks as HH:MM in the requested tz so they're unambiguous
    # (default %d %H:%M reads as "day hour:min" and is easy to misread).
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    if xlim_unix is not None:
        x0 = datetime.fromtimestamp(float(xlim_unix[0]), tz=tz)
        x1 = datetime.fromtimestamp(float(xlim_unix[1]), tz=tz)
        ax.set_xlim(x0, x1)

    date_str = times_dt[0].strftime("%Y-%m-%d")
    ax.set_xlabel(f"Local time ({time_tz}, {date_str})")
    ax.set_ylabel("Coherent cross-baseline power (a.u.)")
    ax.set_title(
        f"Beamformer power vs time, {result['n_chan_used']} chan, "
        f"{result['freq_band_used_mhz'][0]:.1f}-{result['freq_band_used_mhz'][1]:.1f} MHz"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
    return fig
