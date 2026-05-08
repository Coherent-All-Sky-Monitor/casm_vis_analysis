"""Phase vs frequency plot with per-stage subplots.

Layout: ``(n_baselines, n_stages)`` grid. One row per baseline, one column
per processing stage. Each subplot has its own y-axis — necessary because
unwrapped raw-phase swings range over hundreds of radians while
fringe-stopped / post-delay phases sit within a few radians, and
overlaying them in a single panel collapses the latter to a flat line.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_phase_vs_freq(panels, freq_mhz, baseline_labels=None,
                       unwrap=True, output_path=None, time_unix=None,
                       time_mask=None, freq_mask=None):
    """Plot phase vs frequency for multiple processing stages.

    Parameters
    ----------
    panels : list of (str, ndarray)
        Each entry is ``(label, vis_array)`` where ``vis_array`` has shape
        ``(T, F)`` or ``(T, F, n_bl)``. Complex visibilities are
        time-averaged then phase is extracted.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    baseline_labels : list of str, optional
        Labels for each baseline subplot row.
    unwrap : bool
        Unwrap phase along the frequency axis.
    output_path : str or Path, optional
        Save figure to this path.
    time_unix : ndarray, optional
        Time axis for header annotation.
    time_mask : ndarray of bool, shape (T,), optional
        Mask selecting time samples to average over (e.g. transit window).
        When provided, only these time samples are averaged. Default: all.
    freq_mask : ndarray of bool, shape (F,), optional
        ``True`` = good. When provided, RFI-flagged channels are removed
        before unwrapping (so the unwrap doesn't accumulate junk wraps from
        noisy bands) and not plotted. Plot points are still on the full
        ``freq_mhz`` axis — flagged channels appear as gaps.

    Returns
    -------
    fig : matplotlib Figure
    """
    _, first_data = panels[0]
    if first_data.ndim == 2:
        n_bl = 1
    else:
        n_bl = first_data.shape[2]
    n_stages = len(panels)

    if baseline_labels is None:
        baseline_labels = [f"Baseline {i}" for i in range(n_bl)]

    fig, axes = plt.subplots(
        n_bl, n_stages,
        figsize=(3.2 * n_stages, 2.2 * n_bl),
        squeeze=False, sharex=True,
    )

    if freq_mask is not None:
        good = np.asarray(freq_mask, dtype=bool)
    else:
        good = None

    for bl_idx in range(n_bl):
        for st_idx, (label, data) in enumerate(panels):
            ax = axes[bl_idx, st_idx]

            if time_mask is not None and data.ndim >= 2:
                d = data[time_mask]
            else:
                d = data
            if d.ndim == 2:
                vis_avg = np.mean(d, axis=0)
            else:
                vis_avg = np.mean(d[:, :, bl_idx], axis=0)

            phase = np.angle(vis_avg)

            if good is not None:
                # Unwrap on good channels only; leave flagged channels as
                # NaN so matplotlib draws gaps.
                phase_plot = np.full_like(phase, np.nan, dtype=float)
                if unwrap:
                    phase_plot[good] = np.unwrap(phase[good])
                else:
                    phase_plot[good] = phase[good]
                ax.plot(freq_mhz, phase_plot, linewidth=0.8)
            else:
                if unwrap:
                    phase = np.unwrap(phase)
                ax.plot(freq_mhz, phase, linewidth=0.8)

            ax.grid(True, alpha=0.3)

            if bl_idx == 0:
                ax.set_title(label, fontsize=9)
            if st_idx == 0:
                ax.set_ylabel(f"{baseline_labels[bl_idx]}\nPhase (rad)",
                              fontsize=8)
            if bl_idx == n_bl - 1:
                ax.set_xlabel("Frequency (MHz)")

    if time_unix is not None:
        from casm_vis_analysis.plotting import format_time_range
        fig.text(0.5, 0.995, format_time_range(time_unix),
                 ha="center", va="top", fontsize=8, family="monospace",
                 color="0.35")
        top_margin = 0.96
    else:
        top_margin = 0.99

    fig.tight_layout(rect=[0, 0, 1, top_margin])

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
