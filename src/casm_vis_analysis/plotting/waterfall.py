"""Upper-triangle waterfall matrix plot.

Diagonal: autocorrelation power in dB (viridis).
Upper triangle: cross-correlation phase (RdBu, -pi to pi).
Lower triangle: off.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_waterfall(vis, freq_mhz, time_unix, nsig, antenna_labels=None,
                   split_max=16, output_dir=None):
    """Plot waterfall matrix for all baselines.

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_baselines)
        Full upper-triangle visibilities (including autos).
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    time_unix : ndarray, shape (T,)
        Unix timestamps.
    nsig : int
        Number of signals/inputs.
    antenna_labels : list of str, optional
        Labels for each antenna.
    split_max : int
        Maximum number of antennas per figure. Split if nsig > split_max.
    output_dir : str or Path, optional
        Save figures to this directory.

    Returns
    -------
    figs : list of matplotlib Figure
    """
    from casm_io.correlator.baselines import triu_flat_index

    time_hours = (time_unix - time_unix[0]) / 3600.0
    if antenna_labels is None:
        antenna_labels = [f"Ant {i}" for i in range(nsig)]

    # Determine splits
    if nsig <= split_max:
        groups = [list(range(nsig))]
    else:
        groups = []
        for start in range(0, nsig, split_max):
            groups.append(list(range(start, min(start + split_max, nsig))))

    figs = []
    for g_idx, group in enumerate(groups):
        n = len(group)
        fig, axes = plt.subplots(n, n, figsize=(2.5 * n, 2 * n),
                                 squeeze=False)

        for row_local, i in enumerate(group):
            for col_local, j in enumerate(group):
                ax = axes[row_local, col_local]

                if col_local < row_local:
                    # Lower triangle: off
                    ax.set_visible(False)
                    continue

                bl_idx = triu_flat_index(nsig, i, j)

                if i == j:
                    # Diagonal: dB power
                    power_db = 10 * np.log10(np.abs(vis[:, :, bl_idx]) + 1e-30)
                    ax.pcolormesh(time_hours, freq_mhz, power_db.T,
                                  cmap="viridis", shading="auto")
                else:
                    # Upper triangle: phase
                    phase = np.angle(vis[:, :, bl_idx])
                    ax.pcolormesh(time_hours, freq_mhz, phase.T,
                                  cmap="RdBu", shading="auto",
                                  norm=Normalize(-np.pi, np.pi))

                ax.set_xticks([])
                ax.set_yticks([])

                if row_local == 0:
                    ax.set_title(antenna_labels[j], fontsize=7)
                if col_local == row_local:
                    ax.set_ylabel(antenna_labels[i], fontsize=7)

        fig.suptitle(f"Waterfall (group {g_idx + 1}/{len(groups)})", fontsize=10)
        fig.tight_layout()

        if output_dir is not None:
            from pathlib import Path
            path = Path(output_dir) / f"waterfall_group{g_idx + 1}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        figs.append(fig)

    return figs
