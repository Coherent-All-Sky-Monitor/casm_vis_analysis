"""Compose-friendly plotter wrappers.

Each ``plot_*_data(data, ant, **knobs)`` shim unpacks the dict returned by
``read_visibilities``, builds antenna labels from the
:class:`AntennaMapping`, and forwards to the array-level plotter.
"""

from __future__ import annotations

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

    Parameters
    ----------
    data : VisibilityResult or dict-like
        Must expose ``vis`` and ``freq_mhz``.
    ant : AntennaMapping
    include_inactive : bool
        Plot non-functional ADCs too. Default False.
    **kwargs
        Forwarded to the array-level :func:`plot_autocorr`. Common knobs:
        ``time_avg``, ``freq_mask``, ``output_path``, ``ncols``,
        ``time_unix``, ``snap_label``, ``scale``.
    """
    from casm_io.correlator.baselines import triu_flat_index
    from casm_io.correlator import load_format

    vis = _vis_dict_get(data, "vis")
    freq_mhz = _vis_dict_get(data, "freq_mhz")

    # Build per-antenna autocorrelation slice indices.
    # n_signals derived from baseline count: nbl = n*(n+1)/2 -> solve for n.
    n_bl = vis.shape[-1]
    n_sig = int((-1 + (1 + 8 * n_bl) ** 0.5) / 2)
    df = ant.dataframe
    aids = (df["antenna_id"].tolist() if include_inactive
            else ant.active_antennas())
    auto_idxs = [triu_flat_index(n_sig, ant.packet_index(a),
                                 ant.packet_index(a)) for a in aids]
    snap_adc_labels = [f"S{s}A{a}"
                       for s, a in (ant.snap_adc(a) for a in aids)]

    auto_vis = vis[:, :, auto_idxs]
    return _plot_autocorr_array(
        auto_vis, freq_mhz, snap_adc_labels, **kwargs
    )


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

    return _plot_waterfall_array(
        vis, freq_mhz, time_unix, n_sig,
        packet_indices=pkt,
        antenna_labels=antenna_labels,
        snap_adc_labels=snap_adc_labels,
        **kwargs,
    )
