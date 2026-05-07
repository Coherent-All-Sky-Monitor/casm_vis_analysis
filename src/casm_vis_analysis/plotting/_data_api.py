"""Compose-friendly plotter wrappers.

Each ``plot_*_data(data, ant, **knobs)`` shim unpacks the dict returned by
``read_visibilities``, builds antenna labels from the
:class:`AntennaMapping`, and forwards to the array-level plotter.

Display behaviour: when no ``output_path`` / ``output_dir`` is given, the
wrappers call ``plt.show()`` so figures render in Jupyter inline AND in
scripts (in script mode, opens a GUI window if a display is available;
under Agg, the call is a no-op). The returned ``Figure`` object remains
valid for the caller to inspect or save manually.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from casm_vis_analysis.plotting.autocorr import plot_autocorr as _plot_autocorr_array
from casm_vis_analysis.plotting.waterfall import plot_waterfall as _plot_waterfall_array


def _vis_dict_get(data, key):
    if hasattr(data, "__getitem__"):
        try:
            return data[key]
        except (KeyError, TypeError):
            pass
    return getattr(data, key)


def _labels_from_mapping(ant, *, include_inactive=False):
    """Return (packet_indices, antenna_labels, snap_adc_labels) for active set."""
    df = ant.dataframe
    aids = (df["antenna_id"].tolist() if include_inactive
            else ant.active_antennas())
    packet_indices = [ant.packet_index(aid) for aid in aids]
    snap_adc_labels = [
        f"S{s}A{a}" for s, a in (ant.snap_adc(aid) for aid in aids)
    ]
    antenna_labels = []
    for aid in aids:
        s, a = ant.snap_adc(aid)
        grid = ""
        if "row" in df.columns and "col" in df.columns:
            r = df.loc[df.antenna_id == aid].iloc[0]
            grid = f"({r.row}-{r.col})" if r.get("row") else ""
        antenna_labels.append(f"S{s}A{a} {grid}".strip())
    return packet_indices, antenna_labels, snap_adc_labels


def plot_autocorr_data(data, ant, *, include_inactive=False, **kwargs):
    """Autocorrelation power spectra from a ``data`` dict + ``ant`` mapping.

    Mirrors the ``run_autocorr`` runner style: **one figure per SNAP**,
    each with the bold "SNAP N" header at the top, panel titles in
    ``S0A0: N21E5`` form (SNAP-ADC ``:`` plank+col), and no grid lines.
    Pass ``show_grid=True`` to bring grids back.

    Parameters
    ----------
    data : VisibilityResult or dict-like
        Must expose ``vis`` and ``freq_mhz``. ``time_unix`` is forwarded
        if present (used by the plotter's small time-range header).
    ant : AntennaMapping
    include_inactive : bool
        Plot non-functional ADCs too. Default False.
    **kwargs
        Forwarded to the array-level :func:`plot_autocorr`. Common knobs:
        ``time_avg``, ``freq_mask``, ``output_path``, ``ncols``, ``scale``,
        ``show_grid``. ``snap_label`` is set automatically per group.

    Returns
    -------
    list[Figure]
        One ``matplotlib.figure.Figure`` per SNAP (sorted by SNAP id).
    """
    from casm_io.correlator.baselines import triu_flat_index

    vis = _vis_dict_get(data, "vis")
    freq_mhz = _vis_dict_get(data, "freq_mhz")
    time_unix = None
    try:
        time_unix = _vis_dict_get(data, "time_unix")
    except (AttributeError, KeyError):
        pass

    # Default: no grid lines (matches the runner's clean look).
    kwargs.setdefault("show_grid", False)

    # Determine n_inputs from baseline count (nbl = n*(n+1)/2).
    n_bl = vis.shape[-1]
    n_sig = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)

    df = ant.dataframe
    aids = (df["antenna_id"].tolist() if include_inactive
            else ant.active_antennas())
    has_grid_cols = "row" in df.columns and "col" in df.columns

    # Group antennas by SNAP id.
    snap_groups: dict = {}
    for aid in aids:
        snap_id, adc = ant.snap_adc(aid)
        label = f"S{snap_id}A{adc}"
        if has_grid_cols:
            r = df.loc[df.antenna_id == aid].iloc[0]
            row, col = r.get("row"), r.get("col")
            if row and col:
                label = f"{label}: {row}{col}"
        auto_idx = triu_flat_index(n_sig, ant.packet_index(aid),
                                   ant.packet_index(aid))
        snap_groups.setdefault(snap_id, ([], []))
        snap_groups[snap_id][0].append(auto_idx)
        snap_groups[snap_id][1].append(label)

    output_path = kwargs.pop("output_path", None)
    figs = []
    for snap_id, (idxs, labels) in sorted(snap_groups.items()):
        snap_vis = vis[:, :, idxs]
        # Per-SNAP output path: append _snap{N} suffix when saving.
        per_snap_path = None
        if output_path is not None:
            from pathlib import Path
            p = Path(output_path)
            per_snap_path = p.parent / f"{p.stem}_snap{snap_id}{p.suffix}"
        fig = _plot_autocorr_array(
            snap_vis, freq_mhz, labels,
            output_path=per_snap_path,
            time_unix=time_unix,
            snap_label=f"SNAP {snap_id}",
            **kwargs,
        )
        figs.append(fig)

    if output_path is None:
        plt.show()
    return figs


def plot_waterfall_data(data, ant, *, include_inactive=False, **kwargs):
    """Upper-triangle waterfall matrix from a ``data`` dict + ``ant`` mapping.

    Parameters
    ----------
    data : VisibilityResult or dict-like
        Must expose ``vis``, ``freq_mhz``, ``time_unix``.
    ant : AntennaMapping
    **kwargs
        Forwarded to :func:`plot_waterfall`. Common knobs: ``split_max``,
        ``output_dir``, ``diag_spectra``, ``pub``.
    """
    vis = _vis_dict_get(data, "vis")
    freq_mhz = _vis_dict_get(data, "freq_mhz")
    time_unix = _vis_dict_get(data, "time_unix")

    pkt, antenna_labels, snap_adc_labels = _labels_from_mapping(
        ant, include_inactive=include_inactive
    )

    n_bl = vis.shape[-1]
    n_sig = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)

    figs = _plot_waterfall_array(
        vis, freq_mhz, time_unix, n_sig,
        packet_indices=pkt,
        antenna_labels=antenna_labels,
        snap_adc_labels=snap_adc_labels,
        **kwargs,
    )
    if kwargs.get("output_dir") is None:
        plt.show()
    return figs
