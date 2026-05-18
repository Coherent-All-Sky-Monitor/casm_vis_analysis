"""Off-source visibility subtraction.

Estimate the static (direction-independent) per-baseline visibility from a
quiet sky window and subtract it from the observation. Removes correlated
cross-talk, common-mode RFI, ground-pickup pedestals, and similar
contamination that does not fringe with hour angle.

Six functions, six jobs:

* :func:`find_quiet_windows` — scan a time axis for contiguous intervals
  where every named sky source is below a per-source altitude cap.
* :func:`average_visibility` — time-average ``data['vis']`` over a window
  (mask, index range, or unix-time range), respecting ``data['freq_mask']``.
* :func:`subtract_static_visibility` — subtract a static ``(F, n_bl)``
  array from ``data['vis']`` and return a new dict.
* :func:`plot_offsource_diagnostic` — visually confirm that the quiet
  window is genuinely quiet and the static estimate is plausible.
* :func:`save_static_visibility` / :func:`load_static_visibility` —
  persist the ``(F, n_bl)`` static estimate to NPZ so the quiet-window
  read can be done once per night and reused across observations.

Two-step nightly workflow (handles the "quiet window is at night but my
science target is on Sun at noon" problem)::

    # --- One-off, ~30 min of data, write a cache file. ---
    quiet = read_visibilities(
        time_start='2026-05-09 22:29', time_end='2026-05-09 23:20', ...,
    )
    apply_rfi_mask(quiet, RFIMask.from_static())
    windows = find_quiet_windows(
        quiet['time_unix'],
        altitude_caps={'sun': 0, 'tau-a': 0, 'cyg-a': 20, 'cas-a': 15},
    )
    static = average_visibility(quiet, time_mask=windows[0]['mask'])
    fig = plot_offsource_diagnostic(quiet, static, windows[0]['mask'])
    save_static_visibility(
        '/path/static_2026-05-09.npz', static,
        freq_mhz=quiet['freq_mhz'],
        window_unix=(windows[0]['t_start'], windows[0]['t_end']),
        altitudes=windows[0]['altitudes'],
    )
    del quiet                                # free the quiet-window read

    # --- Later, on the actual science data. ---
    data = read_visibilities(time_start='2026-05-09 11:00', ..., ...)
    static_cached = load_static_visibility('/path/static_2026-05-09.npz')
    data_clean = subtract_static_visibility(data, static_cached['static_vis'])
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from .sources import source_altaz


# --------------------------------------------------------------------- #
# 0. One-shot end-to-end builder (date in, ready-to-save result out)     #
# --------------------------------------------------------------------- #

# Sensible defaults at OVRO. Cas A is circumpolar (min alt ~6°), so an
# "all sources below horizon" policy never qualifies; instead we cap
# Cas A and Cyg A at low altitudes where the primary beam attenuates
# them, and require Sun and Tau A actually below the horizon.
DEFAULT_ALTITUDE_CAPS_OVRO = {
    "sun":   0.0,
    "tau-a": 0.0,
    "cyg-a": 20.0,
    "cas-a": 15.0,
}


def build_static_visibility(
    date: str,
    *,
    fmt,
    data_root: str = "/mnt",
    time_tz: str = "America/Los_Angeles",
    altitude_caps: Mapping[str, float] | None = None,
    min_duration_s: float = 15 * 60,
    max_duration_s: float = 60 * 60,
    rfi_mask=None,
    verbose: bool = True,
    _read_fn=None,
) -> dict:
    """Date in → static-vis estimate out. End-to-end orchestrator.

    Workflow inside the function:

    1. Scan a 24-hour grid (1-min cadence) of the given date for the
       first contiguous window where every named source is below its
       altitude cap.
    2. Trim that window to ``max_duration_s`` if longer.
    3. Call ``casm_io.read_visibilities`` for that exact window.
    4. Apply ``rfi_mask`` if given.
    5. Time-average the kept samples (the whole read is already inside
       the quiet window).

    The returned dict carries everything that
    :func:`plot_offsource_diagnostic` and :func:`save_static_visibility`
    need — the user can verify visually, then persist.

    Parameters
    ----------
    date : str
        Calendar date in ``"YYYY-MM-DD"`` form (OVRO local).
    fmt : casm_io VisibilityFormat
        Passed straight through to ``read_visibilities``.
    data_root : str
        Root directory containing ``visibilities_*`` subdirs.
    time_tz : str
        IANA timezone of ``date``. Default ``"America/Los_Angeles"``.
    altitude_caps : dict, optional
        ``{source: max_alt_deg}``. Defaults to
        :data:`DEFAULT_ALTITUDE_CAPS_OVRO`.
    min_duration_s : float
        Reject windows shorter than this. Default 15 min.
    max_duration_s : float
        Trim windows longer than this. Default 60 min — keeps the
        quiet-window read small (~6 GB for 64-ant / 3072-chan).
    rfi_mask : :class:`RFIMask`, optional
        If given, ``apply_rfi_mask`` runs on the quiet read before
        averaging, so flagged channels NaN out cleanly in the static.
    verbose : bool, default True
        Print the chosen window + per-source altitudes.
    _read_fn : callable, optional
        Injection seam for tests. Defaults to
        ``casm_io.correlator.read_visibilities``.

    Returns
    -------
    dict with keys:
      ``date``        — input string
      ``data``        — full VisibilityResult from the quiet read
      ``static_vis``  — ``(F, n_bl)`` complex average
      ``quiet_mask``  — bool ``(T,)``, all True (the entire read IS the
                        quiet window — the mask is kept for plot API
                        consistency with the manual workflow)
      ``window_unix`` — ``(t_start, t_end)`` float
      ``altitudes``   — ``{source: (min, mean, max)}`` over the window
      ``freq_mhz``    — ``(F,)`` frequency axis
    """
    from datetime import datetime, date as _date_cls
    from zoneinfo import ZoneInfo

    caps = dict(altitude_caps) if altitude_caps else dict(DEFAULT_ALTITUDE_CAPS_OVRO)

    # --- 1. Geometric search for a quiet window on the given date. ---
    tz = ZoneInfo(time_tz)
    d = (date if isinstance(date, _date_cls)
         else _date_cls.fromisoformat(date))
    midnight = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=tz)
    grid = np.arange(midnight.timestamp(),
                     midnight.timestamp() + 24 * 3600, 60.0)
    candidates = find_quiet_windows(
        grid, altitude_caps=caps, min_duration_s=min_duration_s,
    )
    if not candidates:
        raise RuntimeError(
            f"No quiet window on {date} (tz={time_tz}) longer than "
            f"{min_duration_s:.0f}s for altitude_caps={caps}. "
            f"Loosen the caps or pick a different date."
        )

    w = candidates[0]
    t_start = float(w["t_start"])
    t_end = float(min(w["t_end"], t_start + max_duration_s))

    if verbose:
        s = datetime.fromtimestamp(t_start, tz=tz)
        e = datetime.fromtimestamp(t_end,   tz=tz)
        print(f"Proposed quiet window: {s:%Y-%m-%d %H:%M %Z}  ->  "
              f"{e:%H:%M %Z}   ({(t_end - t_start) / 60:.0f} min)")
        for name, (mn, mean, mx) in w["altitudes"].items():
            print(f"  {name:8s} alt  min={mn:5.1f}°  mean={mean:5.1f}°  "
                  f"max={mx:5.1f}°")

    # --- 2. Read just that window. ---
    if _read_fn is None:
        from casm_io.correlator import read_visibilities as _read_fn

    s = datetime.fromtimestamp(t_start, tz=tz)
    e = datetime.fromtimestamp(t_end,   tz=tz)
    quiet = _read_fn(
        time_start=s.strftime("%Y-%m-%d %H:%M:%S"),
        time_end=e.strftime("%Y-%m-%d %H:%M:%S"),
        time_tz=time_tz, data_root=data_root, fmt=fmt,
    )

    # --- 3. RFI mask (optional) → 4. average. ---
    if rfi_mask is not None:
        from .rfi import apply_rfi_mask
        apply_rfi_mask(quiet, rfi_mask)

    T = np.asarray(_field(quiet, "time_unix")).shape[0]
    quiet_mask = np.ones(T, dtype=bool)
    static_vis = average_visibility(quiet, time_mask=quiet_mask)

    return {
        "date": str(date),
        "data": quiet,
        "static_vis": static_vis,
        "quiet_mask": quiet_mask,
        "window_unix": (t_start, t_end),
        "altitudes": w["altitudes"],
        "freq_mhz": np.asarray(_field(quiet, "freq_mhz"), dtype=np.float64),
    }


# --------------------------------------------------------------------- #
# 1. Window finder                                                       #
# --------------------------------------------------------------------- #

def find_quiet_windows(
    time_unix,
    *,
    altitude_caps: Mapping[str, float] | None = None,
    min_duration_s: float = 0.0,
) -> list[dict]:
    """Find contiguous time intervals where all named sources are
    simultaneously below their altitude caps.

    Parameters
    ----------
    time_unix : array-like, shape (T,)
        UNIX timestamps to evaluate.
    altitude_caps : dict, optional
        Map source name -> max altitude in degrees. Sources are kept
        below this altitude inside the returned windows. Default:
        ``{'sun': 0, 'cas-a': 0, 'cyg-a': 0, 'tau-a': 0}`` (everything
        below the horizon).
    min_duration_s : float, default 0
        Discard windows shorter than this.

    Returns
    -------
    list of dict, one per qualifying window, each carrying:

    * ``i_start``, ``i_end`` : slice indices into ``time_unix``
    * ``t_start``, ``t_end`` : float UNIX seconds (inclusive)
    * ``duration_s``         : float
    * ``mask``               : bool array of shape (T,), True inside window
    * ``altitudes``          : dict[source -> (min, mean, max)] over window
    """
    if altitude_caps is None:
        altitude_caps = {"sun": 0.0, "cas-a": 0.0, "cyg-a": 0.0, "tau-a": 0.0}
    if not altitude_caps:
        raise ValueError("altitude_caps must contain at least one source")

    time_unix = np.asarray(time_unix, dtype=np.float64)
    T = time_unix.size

    alt = {name: source_altaz(name, time_unix)[0] for name in altitude_caps}
    quiet = np.ones(T, dtype=bool)
    for name, cap in altitude_caps.items():
        quiet &= alt[name] < float(cap)

    # Run-length encode contiguous True segments.
    edges = np.diff(quiet.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if quiet[0]:
        starts.insert(0, 0)
    if quiet[-1]:
        ends.append(T)

    out = []
    for s, e in zip(starts, ends):
        if e <= s:
            continue
        dur = float(time_unix[e - 1] - time_unix[s])
        if dur < min_duration_s:
            continue
        mask = np.zeros(T, dtype=bool)
        mask[s:e] = True
        out.append({
            "i_start": int(s),
            "i_end": int(e),
            "t_start": float(time_unix[s]),
            "t_end": float(time_unix[e - 1]),
            "duration_s": dur,
            "mask": mask,
            "altitudes": {
                name: (
                    float(alt[name][s:e].min()),
                    float(alt[name][s:e].mean()),
                    float(alt[name][s:e].max()),
                )
                for name in altitude_caps
            },
        })
    return out


# --------------------------------------------------------------------- #
# 2. Time-window average                                                 #
# --------------------------------------------------------------------- #

def average_visibility(
    data: Mapping,
    *,
    time_mask=None,
    time_range_unix: tuple | None = None,
    apply_freq_mask: bool = True,
) -> np.ndarray:
    """Time-average ``data['vis']`` over a selection.

    Exactly one of ``time_mask`` or ``time_range_unix`` must be given.

    Parameters
    ----------
    data : Mapping
        Must carry ``vis`` (T, F, n_bl) complex and ``time_unix`` (T,).
        Optional ``freq_mask`` (T-flagging-True) is honoured if
        ``apply_freq_mask=True``: flagged channels in the output are
        set to NaN so subsequent subtraction is a no-op there.
    time_mask : array-like of bool, shape (T,), optional
        Select these samples for the average.
    time_range_unix : (float, float), optional
        Inclusive ``(t_start, t_end)`` selection on ``time_unix``.
    apply_freq_mask : bool, default True
        When True and ``data['freq_mask']`` exists, output channels
        flagged True are replaced with NaN.

    Returns
    -------
    ndarray, shape (F, n_bl), complex128
        Sample-mean of selected ``vis`` rows.
    """
    if (time_mask is None) == (time_range_unix is None):
        raise ValueError(
            "Pass exactly one of time_mask or time_range_unix."
        )

    vis = np.asarray(_field(data, "vis"))
    time_unix = np.asarray(_field(data, "time_unix"), dtype=np.float64)
    if vis.ndim != 3:
        raise ValueError(f"vis must be (T, F, n_bl); got shape {vis.shape}")
    if vis.shape[0] != time_unix.size:
        raise ValueError(
            f"vis.shape[0]={vis.shape[0]} != time_unix.size={time_unix.size}"
        )

    if time_mask is not None:
        sel = np.asarray(time_mask, dtype=bool)
        if sel.shape != time_unix.shape:
            raise ValueError("time_mask shape does not match time_unix")
    else:
        t0, t1 = sorted(time_range_unix)
        sel = (time_unix >= t0) & (time_unix <= t1)

    if not sel.any():
        raise ValueError("Time selection contains zero samples.")

    avg = vis[sel].mean(axis=0).astype(np.complex128)

    if apply_freq_mask:
        raw = _field(data, "freq_mask", default=None)
        if raw is not None:
            bad = np.asarray(raw, dtype=bool)
            if bad.shape[0] != avg.shape[0]:
                raise ValueError(
                    f"freq_mask length {bad.shape[0]} != F={avg.shape[0]}"
                )
            avg[bad, :] = np.nan + 1j * np.nan
    return avg


# --------------------------------------------------------------------- #
# 3. Subtraction                                                         #
# --------------------------------------------------------------------- #

def subtract_static_visibility(
    data: Mapping,
    static_vis: np.ndarray,
) -> dict:
    """Return a new ``data``-shaped dict with ``vis - static_vis[None, :, :]``.

    Channels where ``static_vis`` is NaN pass through unchanged
    (treated as "no estimate available, leave the data alone").

    Parameters
    ----------
    data : Mapping
        Must carry ``vis`` (T, F, n_bl). All other keys are copied
        through unchanged.
    static_vis : ndarray, shape (F, n_bl), complex
        From :func:`average_visibility`.

    Returns
    -------
    dict
        Shallow copy of ``data`` with ``vis`` replaced by the cleaned
        array. ``data`` itself is not mutated.
    """
    vis = np.asarray(_field(data, "vis"))
    if static_vis.shape != vis.shape[1:]:
        raise ValueError(
            f"static_vis shape {static_vis.shape} does not match "
            f"vis last two axes {vis.shape[1:]}"
        )

    nan_mask = ~np.isfinite(static_vis)              # (F, n_bl)
    static_clean = np.where(nan_mask, 0.0 + 0.0j, static_vis)
    cleaned = vis - static_clean[None, :, :].astype(vis.dtype)

    out = dict(data) if isinstance(data, Mapping) else {
        k: getattr(data, k) for k in dir(data) if not k.startswith("_")
    }
    out["vis"] = cleaned
    return out


# --------------------------------------------------------------------- #
# 4. Persistence (save / load the static estimate)                       #
# --------------------------------------------------------------------- #

def save_static_visibility(
    path,
    static_vis: np.ndarray,
    *,
    freq_mhz,
    window_unix: tuple | None = None,
    altitudes: Mapping[str, tuple] | None = None,
    notes: str = "",
) -> None:
    """Persist a static-vis estimate to disk as a single NPZ file.

    Keeps the array together with enough metadata to verify it matches
    the observation it will later be subtracted from (frequency axis,
    quiet-window time range, source altitudes during the average).

    Parameters
    ----------
    path : str or Path
        Output ``.npz`` path. Parent directory must exist.
    static_vis : ndarray, shape (F, n_bl), complex
        From :func:`average_visibility`.
    freq_mhz : ndarray, shape (F,)
        Frequency axis the static was computed on. The consumer must
        check this matches its own ``data['freq_mhz']``.
    window_unix : (float, float), optional
        ``(t_start, t_end)`` UNIX seconds of the quiet window used.
    altitudes : dict, optional
        ``{source: (min, mean, max)}`` per-source altitude during the
        averaging window. From :func:`find_quiet_windows`.
    notes : str, optional
        Free-form comment (e.g. layout version, RFI mask version).
    """
    from pathlib import Path

    static_vis = np.asarray(static_vis)
    if static_vis.ndim != 2:
        raise ValueError(
            f"static_vis must be (F, n_bl); got shape {static_vis.shape}"
        )
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    if freq_mhz.shape[0] != static_vis.shape[0]:
        raise ValueError(
            f"freq_mhz length {freq_mhz.shape[0]} != static_vis "
            f"first axis {static_vis.shape[0]}"
        )

    payload = {
        "static_vis": static_vis.astype(np.complex64),
        "freq_mhz": freq_mhz,
        "notes": np.array(notes),
    }
    if window_unix is not None:
        payload["window_unix"] = np.asarray(window_unix, dtype=np.float64)
    if altitudes is not None:
        # Flatten {source: (min, mean, max)} into two parallel arrays.
        names = list(altitudes.keys())
        vals = np.array(
            [list(altitudes[n]) for n in names], dtype=np.float64
        )  # (n_src, 3)
        payload["altitude_sources"] = np.array(names)
        payload["altitude_min_mean_max"] = vals

    np.savez_compressed(Path(path), **payload)


def load_static_visibility(path) -> dict:
    """Read a static-vis NPZ written by :func:`save_static_visibility`.

    Returns
    -------
    dict with keys ``static_vis``, ``freq_mhz``, ``notes`` (str), and
    optionally ``window_unix`` and ``altitudes`` (dict). Pass
    ``out['static_vis']`` straight into :func:`subtract_static_visibility`.
    """
    from pathlib import Path

    with np.load(Path(path), allow_pickle=False) as f:
        out = {
            "static_vis": f["static_vis"],
            "freq_mhz": f["freq_mhz"],
            "notes": str(f["notes"]) if "notes" in f.files else "",
        }
        if "window_unix" in f.files:
            out["window_unix"] = tuple(float(x) for x in f["window_unix"])
        if "altitude_sources" in f.files:
            names = [str(n) for n in f["altitude_sources"]]
            vals = f["altitude_min_mean_max"]
            out["altitudes"] = {
                n: (float(v[0]), float(v[1]), float(v[2]))
                for n, v in zip(names, vals)
            }
    return out


# --------------------------------------------------------------------- #
# 5. Diagnostic plot                                                     #
# --------------------------------------------------------------------- #

def plot_offsource_diagnostic(
    data: Mapping,
    static_vis: np.ndarray,
    quiet_mask,
    *,
    baseline_indices: Iterable[int] | None = None,
    n_baselines: int = 4,
    freq_band_mhz: tuple | None = None,
    output_path=None,
    time_tz: str = "America/Los_Angeles",
):
    """Two-panel sanity check that the chosen quiet window is genuinely quiet.

    The top panel plots ``|V_ij(t)|`` (averaged over the in-band good
    channels) versus time for a handful of baselines, with the quiet
    window shaded. In a real quiet window the trace should sit near
    a constant level — source-up regions outside the window should
    fringe up and down. If the shaded region itself wiggles at a
    similar amplitude to the source-up regions, the window is
    contaminated and the static estimate will be biased.

    The bottom panel plots ``|static_vis|`` versus frequency for the
    same baselines. Smooth, structureless bandpasses are a good sign;
    spiky structure tracks RFI; sharp fringes-with-frequency mean an
    un-fringe-stopped source still leaks through.

    Parameters
    ----------
    data : Mapping
        Must carry ``vis`` (T, F, n_bl), ``time_unix`` (T,), and
        ``freq_mhz`` (F,). Honours ``freq_mask`` if present.
    static_vis : ndarray, shape (F, n_bl)
        Output of :func:`average_visibility` over ``quiet_mask``.
    quiet_mask : array-like of bool, shape (T,)
        Same shape as ``data['time_unix']``. The shaded region in the
        top panel.
    baseline_indices : iterable of int, optional
        Which baselines (last-axis indices of ``vis``) to plot.
        Defaults to ``n_baselines`` evenly spaced through the
        baseline axis.
    n_baselines : int, default 4
        How many baselines to draw when ``baseline_indices`` is None.
    freq_band_mhz : tuple, optional
        ``(lo, hi)`` MHz inclusive. Restricts the freq-mean in the
        top panel to channels in this band. Default uses every
        unflagged channel.
    output_path : str or Path, optional
        Write the figure here if given.
    time_tz : str, default "America/Los_Angeles"
        IANA tz for the top-panel x-axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    from zoneinfo import ZoneInfo

    vis = np.asarray(_field(data, "vis"))
    time_unix = np.asarray(_field(data, "time_unix"), dtype=np.float64)
    freq_mhz = np.asarray(_field(data, "freq_mhz"), dtype=np.float64)
    quiet_mask = np.asarray(quiet_mask, dtype=bool)
    if quiet_mask.shape != time_unix.shape:
        raise ValueError("quiet_mask shape must match time_unix")
    if static_vis.shape != vis.shape[1:]:
        raise ValueError(
            f"static_vis shape {static_vis.shape} != vis last two axes "
            f"{vis.shape[1:]}"
        )

    # Channel mask: drop RFI-flagged + restrict to band.
    raw_mask = _field(data, "freq_mask", default=None)
    if raw_mask is None:
        chan_keep = np.ones(vis.shape[1], dtype=bool)
    else:
        chan_keep = ~np.asarray(raw_mask, dtype=bool)
    if freq_band_mhz is not None:
        lo, hi = sorted(freq_band_mhz)
        chan_keep &= (freq_mhz >= lo) & (freq_mhz <= hi)
    if not chan_keep.any():
        raise ValueError("No usable channels after RFI/band cut.")

    # Pick baselines.
    n_bl = vis.shape[2]
    if baseline_indices is None:
        baseline_indices = np.linspace(
            0, n_bl - 1, min(n_baselines, n_bl), dtype=int
        ).tolist()
    baseline_indices = [int(b) for b in baseline_indices]

    # |V|(t) freq-averaged over kept channels, per chosen baseline.
    vis_sel = vis[:, chan_keep, :][:, :, baseline_indices]   # (T, F_keep, K)
    amp_t = np.mean(np.abs(vis_sel), axis=1)                  # (T, K)

    # |static|(f) per chosen baseline (no time axis).
    amp_f = np.abs(static_vis[:, baseline_indices])           # (F, K)

    # Plot.
    tz = ZoneInfo(time_tz)
    times_dt = [datetime.fromtimestamp(t, tz=tz) for t in time_unix]

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(11, 7))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for k, b in enumerate(baseline_indices):
        c = color_cycle[k % len(color_cycle)]
        ax_t.plot(times_dt, amp_t[:, k], color=c, lw=0.9,
                  label=f"baseline {b}")
        ax_f.plot(freq_mhz, amp_f[:, k], color=c, lw=0.8,
                  label=f"baseline {b}")

    # Shade the quiet window on the time panel (handles non-contiguous masks).
    edges = np.diff(quiet_mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if quiet_mask[0]:
        starts.insert(0, 0)
    if quiet_mask[-1]:
        ends.append(len(quiet_mask))
    for s, e in zip(starts, ends):
        ax_t.axvspan(times_dt[s], times_dt[e - 1], alpha=0.15,
                     color="C2", label="quiet window" if s == starts[0] else None)

    ax_t.xaxis.set_major_locator(mdates.AutoDateLocator(tz=tz))
    ax_t.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    date_str = times_dt[0].strftime("%Y-%m-%d")
    ax_t.set_xlabel(f"Local time ({time_tz}, {date_str})")
    ax_t.set_ylabel("|V(t)| freq-averaged in band  (a.u.)")
    ax_t.set_title(
        "Quiet-window check: |V|(t) per baseline — shaded interval is "
        "the proposed off-source window"
    )
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8, loc="upper right", ncol=2)

    if raw_mask is not None:
        # NaN flagged channels in static_vis appear as breaks; matplotlib
        # already handles this. Just annotate.
        n_flagged = int(np.asarray(raw_mask, dtype=bool).sum())
        ax_f.set_title(
            f"Static estimate |static_vis|(f) per baseline   "
            f"({n_flagged}/{len(raw_mask)} channels flagged → NaN)"
        )
    else:
        ax_f.set_title("Static estimate |static_vis|(f) per baseline")
    ax_f.set_xlabel("Frequency (MHz)")
    ax_f.set_ylabel("|static_vis|  (a.u.)")
    ax_f.grid(True, alpha=0.3)
    ax_f.legend(fontsize=8, loc="upper right", ncol=2)

    fig.tight_layout()
    if output_path is not None:
        from pathlib import Path
        fig.savefig(Path(output_path), dpi=140, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------- #
# 6. Helper                                                              #
# --------------------------------------------------------------------- #

def _field(data, name, default=...):
    """Read ``name`` from a VisibilityResult or plain dict."""
    if hasattr(data, name):
        return getattr(data, name)
    try:
        return data[name]
    except (KeyError, TypeError):
        if default is ...:
            raise ValueError(f"data missing required field {name!r}")
        return default
