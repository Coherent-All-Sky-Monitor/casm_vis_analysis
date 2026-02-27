"""Autocorrelation power (dB) vs frequency grid plot."""

import numpy as np
import matplotlib.pyplot as plt


def plot_autocorr(vis, freq_mhz, antenna_labels, time_avg=True,
                  freq_mask=None, output_path=None, ncols=4):
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

    power_db = 10 * np.log10(power + 1e-30)
    n_ant = power_db.shape[1]
    nrows = int(np.ceil(n_ant / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    freq_plot = freq_mhz.copy()
    if freq_mask is not None:
        freq_plot_masked = np.where(freq_mask, np.nan, freq_plot)
    else:
        freq_plot_masked = freq_plot

    for i in range(n_ant):
        ax = axes[i // ncols, i % ncols]
        y = power_db[:, i].copy()
        if freq_mask is not None:
            y[freq_mask] = np.nan
        ax.plot(freq_plot, y, linewidth=0.5)
        ax.set_title(antenna_labels[i], fontsize=9)
        ax.grid(True, alpha=0.3)

    # Turn off unused axes
    for i in range(n_ant, nrows * ncols):
        axes[i // ncols, i % ncols].set_visible(False)

    fig.supxlabel("Frequency (MHz)")
    fig.supylabel("Power (dB)")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
