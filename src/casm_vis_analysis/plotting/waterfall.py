"""Upper-triangle waterfall matrix plot.

Diagonal: autocorrelation power in dB (viridis).
Upper triangle: cross-correlation phase (RdBu, -pi to pi).
Lower triangle: off.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_waterfall(vis, freq_mhz, time_unix, nsig, packet_indices,
                   antenna_labels, snap_adc_labels, split_max=16,
                   output_dir=None, diag_spectra=False, pub=False):
    """Plot waterfall matrix for active antennas.

    Parameters
    ----------
    vis : ndarray, shape (T, F, n_baselines)
        Full upper-triangle visibilities (including autos).
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    time_unix : ndarray, shape (T,)
        Unix timestamps.
    nsig : int
        Number of correlator inputs (for triu_flat_index).
    packet_indices : list of int
        Correlator input index for each active antenna.
    antenna_labels : list of str
        Full label per antenna (for diagonal titles).
    snap_adc_labels : list of str
        Short label per antenna (for cross-correlation titles).
    split_max : int
        Maximum antennas per figure.
    output_dir : str or Path, optional
        Save figures to this directory.
    diag_spectra : bool
        When True, diagonal cells show 1D time-averaged power spectrum
        instead of 2D waterfall.
    pub : bool
        When True, save as PDF at 300 DPI instead of PNG at 150 DPI.

    Returns
    -------
    figs : list of matplotlib Figure
    """
    from casm_io.correlator.baselines import triu_flat_index

    n_ant = len(packet_indices)
    time_hours = (time_unix - time_unix[0]) / 3600.0

    # Determine splits based on active antenna count
    if n_ant <= split_max:
        groups = [list(range(n_ant))]
    else:
        groups = []
        for start in range(0, n_ant, split_max):
            groups.append(list(range(start, min(start + split_max, n_ant))))

    figs = []
    for g_idx, group in enumerate(groups):
        n = len(group)
        fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n),
                                 squeeze=False)

        for row_local, i in enumerate(group):
            for col_local, j in enumerate(group):
                ax = axes[row_local, col_local]

                if col_local < row_local:
                    ax.set_visible(False)
                    continue

                # Map antenna indices to packet (correlator input) indices
                inp_i = packet_indices[i]
                inp_j = packet_indices[j]
                lo, hi = min(inp_i, inp_j), max(inp_i, inp_j)
                bl_idx = triu_flat_index(nsig, lo, hi)
                conjugate = inp_i > inp_j

                bl_vis = vis[:, :, bl_idx]
                if conjugate:
                    bl_vis = np.conj(bl_vis)

                if i == j:
                    if diag_spectra:
                        # 1D time-averaged power spectrum
                        power_db = 10 * np.log10(
                            np.mean(np.abs(bl_vis), axis=0) + 1e-30
                        )
                        ax.plot(freq_mhz, power_db, linewidth=0.5)
                        ax.set_xlabel("Freq (MHz)", fontsize=5)
                        ax.set_ylabel("Power (dB)", fontsize=5)
                        ax.grid(alpha=0.3)
                    else:
                        # Diagonal: dB power waterfall
                        power_db = 10 * np.log10(np.abs(bl_vis) + 1e-30)
                        ax.pcolormesh(time_hours, freq_mhz, power_db.T,
                                      cmap="viridis", shading="auto")
                    ax.set_title(antenna_labels[i], fontsize=6)
                else:
                    # Upper triangle: phase
                    phase = np.angle(bl_vis)
                    ax.pcolormesh(time_hours, freq_mhz, phase.T,
                                  cmap="RdBu", shading="auto",
                                  norm=Normalize(-np.pi, np.pi))
                    ax.set_title(
                        f"{snap_adc_labels[i]} \u00d7 {snap_adc_labels[j]}",
                        fontsize=5,
                    )

                ax.set_xticks([])
                ax.set_yticks([])

        # Compact single-line header
        from casm_vis_analysis.plotting import format_time_range
        group_label = f"Waterfall ({g_idx + 1}/{len(groups)})"
        header = f"{group_label}  —  {format_time_range(time_unix)}"
        fig.text(0.5, 0.995, header,
                 ha="center", va="top", fontsize=8, fontweight="bold",
                 family="monospace", color="0.3")
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if output_dir is not None:
            from pathlib import Path
            if pub:
                path = Path(output_dir) / f"waterfall_group{g_idx + 1}.pdf"
                fig.savefig(path, dpi=300, bbox_inches="tight")
            else:
                path = Path(output_dir) / f"waterfall_group{g_idx + 1}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        figs.append(fig)

    return figs
