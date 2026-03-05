"""Fringe-stopping diagnostic waterfall: configurable columns per SNAP pair."""

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def group_by_snap_pair(ref_snap, target_snaps):
    """Group target indices by SNAP pair.

    Parameters
    ----------
    ref_snap : int
        SNAP ID of the reference antenna.
    target_snaps : list of int
        SNAP IDs of each target antenna.

    Returns
    -------
    groups : dict
        {(ref_snap, target_snap): [target_indices]}
    """
    groups = {}
    for idx, snap in enumerate(target_snaps):
        key = (ref_snap, snap)
        groups.setdefault(key, []).append(idx)
    return groups


def plot_fringe_diagnostic(panels, time_unix, freq_mhz, target_labels,
                           target_snaps, ref_snap, output_dir=None,
                           split_max=20):
    """Plot fringe diagnostic waterfalls grouped by SNAP pair.

    Parameters
    ----------
    panels : list of (str, ndarray)
        Each entry is (label, data) where data has shape (T, F, n_targets).
        Data is interpreted as phase (angle taken if complex, wrapped if real).
    time_unix : ndarray, shape (T,)
        Unix timestamps.
    freq_mhz : ndarray, shape (F,)
        Frequency axis in MHz.
    target_labels : list of str
        Label per target baseline.
    target_snaps : list of int
        SNAP ID per target antenna.
    ref_snap : int
        Reference antenna SNAP ID.
    output_dir : str or Path, optional
        Save figures here.
    split_max : int, optional
        Maximum baselines per figure before splitting (default: 20).

    Returns
    -------
    figs : list of matplotlib Figure
    """
    time_hours = (time_unix - time_unix[0]) / 3600.0
    groups = group_by_snap_pair(ref_snap, target_snaps)

    figs = []
    for (rs, ts), indices in sorted(groups.items()):
        # Split large groups into parts
        if len(indices) > split_max:
            n_parts = math.ceil(len(indices) / split_max)
            chunks = [indices[i * split_max:(i + 1) * split_max]
                      for i in range(n_parts)]
        else:
            chunks = [indices]
            n_parts = 1

        for part_idx, chunk in enumerate(chunks):
            ncols = len(panels)
            nrows = len(chunk)

            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(3.5 * ncols, 2.5 * nrows),
                                     squeeze=False, sharex=True, sharey=True)

            for col, (panel_label, data) in enumerate(panels):
                axes[0, col].set_title(panel_label, fontsize=9)
                for row, bl_idx in enumerate(chunk):
                    ax = axes[row, col]
                    d = data[:, :, bl_idx]
                    if np.iscomplexobj(d):
                        d = np.angle(d)
                    else:
                        d = np.mod(d + np.pi, 2 * np.pi) - np.pi
                    ax.pcolormesh(time_hours, freq_mhz, d.T,
                                  cmap="RdBu", shading="auto",
                                  norm=Normalize(-np.pi, np.pi))
                    if col == 0:
                        ax.set_ylabel(target_labels[bl_idx], fontsize=7)

            snap_title = f"SNAP {rs} \u2192 {ts}"
            if n_parts > 1:
                snap_title += f" (part {part_idx + 1}/{n_parts})"
            if time_unix is not None:
                from casm_vis_analysis.plotting import format_time_range
                fig.text(0.5, 0.99, format_time_range(time_unix),
                         ha="center", va="top", fontsize=8, family="monospace",
                         color="0.35")
                fig.text(0.5, 0.955, snap_title,
                         ha="center", va="top", fontsize=11, fontweight="bold")
                top_margin = 0.90
            else:
                fig.suptitle(snap_title, fontsize=11, fontweight="bold")
                top_margin = 0.93
            fig.supxlabel("Time (hours)")
            fig.supylabel("Freq (MHz)")
            fig.tight_layout(rect=[0, 0, 1, top_margin])

            if output_dir is not None:
                from pathlib import Path
                if n_parts > 1:
                    fname = f"fringe_diag_snap{rs}_to_{ts}_part{part_idx + 1}.png"
                else:
                    fname = f"fringe_diag_snap{rs}_to_{ts}.png"
                path = Path(output_dir) / fname
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            figs.append(fig)

    return figs
