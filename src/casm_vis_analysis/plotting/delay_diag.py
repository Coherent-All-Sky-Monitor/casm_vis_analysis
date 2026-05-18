"""Delay-fit diagnostic plots.

Two functions, both consume the dict returned by ``fit_delay(model='linear')``:

- :func:`plot_coherence_curves` — visual confirmation of each fit. One small
  panel per baseline showing ``C(tau)`` from the delay-domain search, with a
  vertical line at the chosen delay and quality flags annotated.

- :func:`plot_delay_vs_baseline_length` — physical sanity check. Scatter of
  fitted delay against antenna baseline length, with a twin y-axis showing
  the equivalent cable-length difference at a coax velocity factor.
"""

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


C_LIGHT_M_S = 2.998e8


def plot_coherence_curves(fit_params, target_labels=None,
                          output_path=None, split_max=20):
    """Per-baseline coherence ``|C(tau)|`` from the delay-domain search.

    Each panel plots the coherence curve, marks the fitted delay with a
    vertical line, and annotates ``r_squared`` and
    ``peak_to_secondary_ratio``. Panels for baselines flagged
    ``low_quality=True`` get a red border so they jump out visually.

    Parameters
    ----------
    fit_params : dict
        Output of ``fit_delay(..., model='linear', return_coherence=True)``.
        Must contain ``coherence_curves`` (n_grid, n_bl), ``tau_grid_ns``
        (n_grid,), ``delay_ns`` (n_bl,), ``r_squared`` (n_bl,),
        ``peak_to_secondary_ratio`` (n_bl,), ``low_quality`` (n_bl,).
    target_labels : list of str, optional
        One label per baseline. Defaults to ``"baseline {i}"``.
    output_path : str or Path, optional
        Save figure(s) to this path. ``_partN`` is appended for splits.
    split_max : int
        Max baselines per figure.

    Returns
    -------
    figs : list of matplotlib Figure
    """
    if "coherence_curves" not in fit_params:
        raise KeyError(
            "fit_params has no 'coherence_curves' — call "
            "fit_delay(..., return_coherence=True)."
        )

    curves = np.atleast_2d(fit_params["coherence_curves"])
    if curves.shape[0] == fit_params["tau_grid_ns"].shape[0]:
        # (n_grid, n_bl) form — fine
        pass
    else:
        # single-baseline (n_grid,) was upgraded to (1, n_grid) by atleast_2d
        curves = curves.T

    tau_grid = np.asarray(fit_params["tau_grid_ns"])
    delay_ns = np.atleast_1d(fit_params["delay_ns"])
    r2 = np.atleast_1d(fit_params["r_squared"])
    p2s = np.atleast_1d(fit_params["peak_to_secondary_ratio"])
    flags = np.atleast_1d(fit_params["low_quality"])
    n_bl = curves.shape[1]

    if target_labels is None:
        target_labels = [f"baseline {i}" for i in range(n_bl)]

    chunks = [list(range(s, min(s + split_max, n_bl)))
              for s in range(0, n_bl, split_max)]
    n_parts = len(chunks)

    figs = []
    for part_idx, chunk in enumerate(chunks):
        n = len(chunk)
        ncols = 4
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.6 * ncols, 2.4 * nrows),
                                 squeeze=False, sharex=True)
        for ax in axes.flat:
            ax.axis("off")

        for k, bl_idx in enumerate(chunk):
            r, c = divmod(k, ncols)
            ax = axes[r, c]
            ax.axis("on")

            curve = curves[:, bl_idx]
            peak_val = curve.max()
            ax.plot(tau_grid, curve / peak_val, lw=0.8, color="C0")
            ax.axvline(delay_ns[bl_idx], color="red", lw=1.0, alpha=0.7)
            ax.set_xlim(tau_grid[0], tau_grid[-1])
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

            ax.set_title(target_labels[bl_idx], fontsize=8)
            txt = (f"τ={delay_ns[bl_idx]:.1f} ns\n"
                   f"r²={r2[bl_idx]:.3f}  pk/2nd={p2s[bl_idx]:.1f}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    fontsize=7, va="top", ha="left", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="0.7", alpha=0.85))

            if flags[bl_idx]:
                for spine in ax.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(1.6)

            if r == nrows - 1:
                ax.set_xlabel("τ (ns)")
            if c == 0:
                ax.set_ylabel("|C(τ)| / max")

        title = "Delay-search coherence curves"
        if n_parts > 1:
            title += f"  (part {part_idx + 1}/{n_parts})"
        title += "    — red border = low_quality"
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        if output_path is not None:
            p = Path(output_path)
            if n_parts > 1:
                out = p.with_name(f"{p.stem}_part{part_idx + 1}{p.suffix}")
            else:
                out = p
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)

        figs.append(fig)

    return figs


def plot_delay_vs_baseline_length(delay_ns, baseline_lengths_m,
                                  target_labels=None,
                                  target_snaps=None,
                                  target_slots=None,
                                  low_quality=None,
                                  velocity_factor=0.67,
                                  output_path=None):
    """Scatter delay vs. antenna baseline length, coloured by SNAP.

    Cable delays should not correlate with antenna separation if cables are
    routed independently of array geometry — flat scatter is the expected
    sanity-check signature. A correlation hints that cable length tracks
    antenna placement (e.g. central trunks running outward to remote ants).
    Colouring by SNAP makes the cluster-per-chassis pattern obvious.

    Twin y-axis shows the same delay re-expressed as a cable-length
    difference at the given velocity factor (default 0.67c, typical coax).

    Parameters
    ----------
    delay_ns : array-like, shape (n_bl,)
        Fitted delay per baseline (ref - target convention).
    baseline_lengths_m : array-like, shape (n_bl,)
        Physical antenna separation (m).
    target_labels : list of str, optional
        One label per baseline. Used to annotate ``low_quality`` points.
    target_snaps : array-like of int, shape (n_bl,), optional
        SNAP id of the target antenna for each baseline. When provided,
        points are coloured by SNAP and a legend is added. ``None``
        falls back to a single colour.
    target_slots : array-like of str, shape (n_bl,), optional
        Chassis slot letter (e.g. ``'A'``, ``'B'``, ``'D'``, ``'I'``)
        of the target antenna for each baseline, from the antenna
        layout CSV. When provided alongside ``target_snaps``, the
        legend reads ``"SNAP {snap}/Slot {slot}"``.
    low_quality : array-like of bool, shape (n_bl,), optional
        From ``fit_params['low_quality']``. Flagged baselines are drawn
        as hollow markers with a red edge and labelled.
    velocity_factor : float
        Coax velocity factor for the secondary axis. Default 0.67.
    output_path : str or Path, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    delay_ns = np.asarray(delay_ns)
    baseline_lengths_m = np.asarray(baseline_lengths_m)
    n = len(delay_ns)
    if low_quality is None:
        low_quality = np.zeros(n, dtype=bool)
    else:
        low_quality = np.asarray(low_quality, dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 5))

    if target_snaps is not None:
        target_snaps = np.asarray(target_snaps)
        snap_ids = sorted(np.unique(target_snaps).tolist())
        cmap = plt.get_cmap("tab10")
        colors = {s: cmap(i % cmap.N) for i, s in enumerate(snap_ids)}
        if target_slots is not None:
            target_slots = np.asarray(target_slots)
            slot_for_snap = {}
            for s in snap_ids:
                slots_here = [sl for sl in target_slots[target_snaps == s]
                              if sl is not None and str(sl) not in ("nan", "None")]
                slot_for_snap[s] = str(slots_here[0]) if slots_here else "?"
            label_for = lambda s: f"SNAP {s}/Slot {slot_for_snap[s]}"
        else:
            label_for = lambda s: f"SNAP {s}"
        for s in snap_ids:
            sel_good = (target_snaps == s) & ~low_quality
            sel_bad = (target_snaps == s) & low_quality
            if sel_good.any():
                ax.scatter(baseline_lengths_m[sel_good], delay_ns[sel_good],
                           s=32, color=colors[s], label=label_for(s),
                           edgecolor="0.2", linewidth=0.4, zorder=3)
            if sel_bad.any():
                ax.scatter(baseline_lengths_m[sel_bad], delay_ns[sel_bad],
                           s=42, facecolor=colors[s], edgecolor="red",
                           linewidth=1.4, zorder=4,
                           label=f"{label_for(s)} (low_q)")
    else:
        good = ~low_quality
        ax.scatter(baseline_lengths_m[good], delay_ns[good],
                   s=32, color="C0", label="good",
                   edgecolor="0.2", linewidth=0.4, zorder=3)
        if low_quality.any():
            ax.scatter(baseline_lengths_m[low_quality], delay_ns[low_quality],
                       s=42, facecolor="white", edgecolor="red",
                       linewidth=1.4, label="low_quality", zorder=4)

    if low_quality.any() and target_labels is not None:
        for i in np.where(low_quality)[0]:
            ax.annotate(target_labels[i],
                        (baseline_lengths_m[i], delay_ns[i]),
                        xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color="red")

    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_xlabel("Antenna baseline length (m)")
    ax.set_ylabel("Fitted delay (ns)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)

    # Twin y-axis: same data re-labelled as cable-length difference at v_f.
    # 1 ns of delay corresponds to v_f * c * 1 ns of physical cable.
    ns_to_m = velocity_factor * C_LIGHT_M_S * 1e-9
    ax2 = ax.twinx()
    y0, y1 = ax.get_ylim()
    ax2.set_ylim(y0 * ns_to_m, y1 * ns_to_m)
    ax2.set_ylabel(f"Δ cable length (m)  [v_f = {velocity_factor} c]")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(Path(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
