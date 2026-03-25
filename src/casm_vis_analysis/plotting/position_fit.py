"""Diagnostic plots for antenna position fitting (x or y)."""

import numpy as np
import matplotlib.pyplot as plt


def _detect_axis(results):
    """Detect which axis was fitted from result keys."""
    if "best_y" in results[0]:
        return "y"
    return "x"


def plot_score_curves(results, ref_label, output_dir=None, ncols=3):
    """Plot circular variance vs trial position, one subplot per antenna.

    Parameters
    ----------
    results : list of dict
        From fit_all_antennas(). Each dict has: label, pos_grid, scores,
        best_{x|y}, sigma_{x|y}, best_score, coeffs, current_{x|y}.
    ref_label : str
        Reference antenna label for suptitle.
    output_dir : str or Path, optional
        Directory to save PNG files. If None, figures are returned without saving.
    ncols : int
        Columns in subplot grid.

    Returns
    -------
    figs : list of Figure
    """
    a = _detect_axis(results)
    # Support both old "x_grid" and new "pos_grid" keys
    grid_key = "x_grid" if "x_grid" in results[0] else "pos_grid"

    n = len(results)
    max_per_page = ncols * 6  # up to 6 rows per page
    figs = []

    for page_start in range(0, n, max_per_page):
        page_results = results[page_start:page_start + max_per_page]
        n_page = len(page_results)
        nrows = int(np.ceil(n_page / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(4.5 * ncols, 3.2 * nrows),
                                 squeeze=False)

        for idx, res in enumerate(page_results):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            pos_grid = res[grid_key]
            scores = res["scores"]
            best_pos = res[f"best_{a}"]
            sigma_pos = res[f"sigma_{a}"]
            coeffs = res["coeffs"]
            current_pos = res[f"current_{a}"]

            # Scatter of scores
            ax.scatter(pos_grid, scores, s=8, alpha=0.6, color="C0", zorder=2)

            # Parabola fit curve (only over the fit region)
            if coeffs[0] > 0:
                i_min = int(np.argmin(scores))
                i_lo = max(0, i_min - 10)
                i_hi = min(len(pos_grid), i_min + 11)
                p_fit = np.linspace(pos_grid[i_lo], pos_grid[min(i_hi, len(pos_grid) - 1)], 200)
                y_fit = np.polyval(coeffs, p_fit)
                ax.plot(p_fit, y_fit, "C1-", linewidth=1.2, alpha=0.8,
                        label="parabola fit")

            # Vertical lines
            ax.axvline(best_pos, color="C3", linewidth=1.0, linestyle="--",
                       label=f"best={best_pos:.3f}")
            ax.axvline(current_pos, color="0.5", linewidth=0.8, linestyle=":",
                       label=f"current={current_pos:.2f}")

            title = f"{res['label']}\n{a}={best_pos:.3f}±{sigma_pos:.3f} m"
            ax.set_title(title, fontsize=8)
            ax.set_xlabel(f"{a} (m)", fontsize=7)
            ax.set_ylabel("score", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5, loc="upper right")
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_page, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        page_num = page_start // max_per_page + 1
        fig.suptitle(f"{a.upper()}-position scan (ref: {ref_label}) "
                     f"— page {page_num}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        if output_dir is not None:
            from pathlib import Path
            path = Path(output_dir) / f"score_curves_p{page_num}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")

        figs.append(fig)

    return figs


def plot_position_summary(results, ref_label, output_path=None):
    """Error bar plot of best position ± sigma for all antennas.

    Parameters
    ----------
    results : list of dict
    ref_label : str
    output_path : str or Path, optional

    Returns
    -------
    fig : Figure
    """
    a = _detect_axis(results)
    labels = [r["label"] for r in results]
    best_pos = np.array([r[f"best_{a}"] for r in results])
    sigma_pos = np.array([r[f"sigma_{a}"] for r in results])
    current_pos = np.array([r[f"current_{a}"] for r in results])

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 0.5), 5))

    y_pos = np.arange(len(results))
    ax.errorbar(best_pos, y_pos, xerr=sigma_pos, fmt="o", markersize=5,
                capsize=3, color="C0", label=f"fitted {a}")
    ax.scatter(current_pos, y_pos, marker="x", s=40, color="C3", zorder=3,
               label=f"current {a}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"{a} position (m)")
    ax.set_title(f"Fitted {a}-positions (ref: {ref_label})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_multiday_comparison(all_day_results, antenna_labels,
                             output_path=None, axis="x"):
    """Fitted position across days with error bars for consistency check.

    Parameters
    ----------
    all_day_results : list of (str, list[dict])
        Each entry is (obs_id, results) from fit_all_antennas().
    antenna_labels : list of str
        Antenna labels (same order across days).
    output_path : str or Path, optional
    axis : str
        "x" or "y".

    Returns
    -------
    fig : Figure
    """
    a = axis
    n_days = len(all_day_results)
    n_ant = len(antenna_labels)

    fig, axes = plt.subplots(1, 1, figsize=(max(10, n_ant * 0.8), 6))
    ax = axes if not hasattr(axes, '__len__') else axes

    x_pos = np.arange(n_ant)
    width = 0.7 / max(n_days, 1)

    for d_idx, (obs_id, results) in enumerate(all_day_results):
        best_pos = np.array([r[f"best_{a}"] for r in results])
        sigma_pos = np.array([r[f"sigma_{a}"] for r in results])
        offset = (d_idx - n_days / 2 + 0.5) * width
        ax.errorbar(x_pos + offset, best_pos, yerr=sigma_pos,
                    fmt="o", markersize=4, capsize=2,
                    label=obs_id[:10])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(antenna_labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel(f"fitted {a} (m)")
    ax.set_title(f"Multi-day {a}-position comparison")
    ax.legend(fontsize=6, ncol=min(4, n_days))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
