"""Phase vs frequency plot with configurable columns/lines."""

import numpy as np
import matplotlib.pyplot as plt


def plot_phase_vs_freq(panels, freq_mhz, baseline_labels=None,
                       unwrap=True, output_path=None):
    """Plot phase vs frequency for multiple processing stages.

    Parameters
    ----------
    panels : list of (str, ndarray)
        Each entry is (label, vis_array) where vis_array has shape
        (T, F) or (T, F, n_bl). Complex visibilities are time-averaged
        then phase is extracted.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    baseline_labels : list of str, optional
        Labels for each baseline subplot.
    unwrap : bool
        Whether to unwrap phase.
    output_path : str or Path, optional
        Save figure to this path.

    Returns
    -------
    fig : matplotlib Figure
    """
    # Determine number of baselines from first panel
    _, first_data = panels[0]
    if first_data.ndim == 2:
        n_bl = 1
    else:
        n_bl = first_data.shape[2]

    if baseline_labels is None:
        baseline_labels = [f"Baseline {i}" for i in range(n_bl)]

    fig, axes = plt.subplots(n_bl, 1, figsize=(10, 3 * n_bl),
                             squeeze=False, sharex=True)

    for bl_idx in range(n_bl):
        ax = axes[bl_idx, 0]
        for label, data in panels:
            if data.ndim == 2:
                vis_avg = np.mean(data, axis=0)
            else:
                vis_avg = np.mean(data[:, :, bl_idx], axis=0)

            phase = np.angle(vis_avg)
            if unwrap:
                phase = np.unwrap(phase)

            ax.plot(freq_mhz, phase, label=label, linewidth=0.8)

        ax.set_ylabel("Phase (rad)")
        ax.set_title(baseline_labels[bl_idx], fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Frequency (MHz)")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
