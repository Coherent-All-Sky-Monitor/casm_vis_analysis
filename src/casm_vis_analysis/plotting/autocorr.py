"""Autocorrelation power (dB) vs frequency grid plot."""

import numpy as np
import matplotlib.pyplot as plt


def plot_autocorr(vis, freq_mhz, antenna_labels, time_avg=True,
                  freq_mask=None, output_path=None, ncols=4,
                  time_unix=None, snap_label=None, scale="dB"):
    """Plot autocorrelation power spectra on a grid.

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_ant) or (F, n_ant) if time_avg=False
        Autocorrelation visibilities (real-valued).
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    antenna_labels : list of str
        Labels for each antenna panel.
    time_avg : bool
        If True, average over time axis first.
    freq_mask : ndarray of bool, shape (F,), optional
        Channels to mask (True = masked).
    output_path : str or Path, optional
        Save figure to this path. If None, calls plt.show().
    ncols : int
        Number of columns in grid.

    Returns
    -------
    fig : matplotlib Figure
    """
    if time_avg and vis.ndim == 3:
        power = np.mean(np.abs(vis), axis=0)  # (F, n_ant)
    elif vis.ndim == 3:
        power = np.abs(vis[0])
    else:
        power = np.abs(vis)

    if scale == "dB":
        plot_data = 10 * np.log10(power + 1e-30)
        ylabel = "Power (dB)"
    else:
        plot_data = power
        ylabel = "Power (linear)"

    n_ant = plot_data.shape[1]
    nrows = int(np.ceil(n_ant / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for i in range(n_ant):
        ax = axes[i // ncols, i % ncols]
        y = plot_data[:, i].copy()
        if freq_mask is not None:
            y[freq_mask] = np.nan
        ax.plot(freq_mhz, y, linewidth=0.5)
        ax.set_title(antenna_labels[i], fontsize=9)
        ax.grid(True, alpha=0.3)

    # Turn off unused axes
    for i in range(n_ant, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    # Header: time range (subtle) + snap label (prominent), with spacing
    # Scale header room by number of rows so short figures get more breathing room
    has_time = time_unix is not None
    has_snap = snap_label is not None
    if has_time or has_snap:
        header_inches = 0.7 if (has_time and has_snap) else 0.4
        fig_h = fig.get_size_inches()[1]
        top_margin = 1.0 - header_inches / fig_h
        if has_time:
            from casm_vis_analysis.plotting import format_time_range
            fig.text(0.5, 0.99, format_time_range(time_unix),
                     ha="center", va="top", fontsize=8, family="monospace",
                     color="0.35")
        if has_snap:
            snap_y = 0.99 - 0.35 / fig_h if has_time else 0.99
            fig.text(0.5, snap_y, snap_label,
                     ha="center", va="top", fontsize=11, fontweight="bold")
    else:
        top_margin = 0.95

    fig.supxlabel("Frequency (MHz)")
    fig.supylabel(ylabel)
    fig.tight_layout(rect=[0, 0, 1, top_margin])

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
