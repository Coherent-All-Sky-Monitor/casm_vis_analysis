"""CASM correlator visibility analysis."""

__version__ = "0.3.0"

# Compose-friendly notebook API (primary).
from casm_vis_analysis.fringe_stop import (
    fringe_stop, FringeStoppedData,
    coherence_metric, auto_detect_sign,
    fringe_stop_array, fringe_stop_single_baseline,
    compute_baselines_enu, geometric_delay,
)
from casm_vis_analysis.delay import (
    fit_delay, apply_delay,
    compute_per_freq_phasor, apply_per_freq_phasor,
)
from casm_vis_analysis.sources import (
    source_position, source_altaz, source_enu,
    source_flux, SOURCE_CATALOG, SUN_FLUX_400_DEFAULT, SUN_SPECTRAL_INDEX,
    find_transit_window,
)
from casm_vis_analysis.rfi import RFIMask
from casm_vis_analysis.plotting.autocorr import plot_autocorr as plot_autocorr_array
from casm_vis_analysis.plotting.waterfall import plot_waterfall as plot_waterfall_array
from casm_vis_analysis.plotting.fringe_diag import plot_fringe_diagnostic as plot_fringe_diag
from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq
# Compose-friendly dict wrappers (primary public API).
from casm_vis_analysis.plotting._data_api import (
    plot_autocorr_data as plot_autocorr,
    plot_waterfall_data as plot_waterfall,
)

# CLI mirrors (one-shot; re-read on each call).
from casm_vis_analysis.runners import (
    run_autocorr, run_waterfall, run_fringe_stop,
)
from casm_vis_analysis.layout import run_build_layout, run_sync_wiring

__all__ = [
    # Compose API
    "fringe_stop", "FringeStoppedData",
    "coherence_metric", "auto_detect_sign",
    "fringe_stop_array", "fringe_stop_single_baseline",
    "compute_baselines_enu", "geometric_delay",
    "fit_delay", "apply_delay",
    "compute_per_freq_phasor", "apply_per_freq_phasor",
    "source_position", "source_altaz", "source_enu",
    "source_flux", "SOURCE_CATALOG", "SUN_FLUX_400_DEFAULT",
    "SUN_SPECTRAL_INDEX", "find_transit_window",
    "RFIMask",
    "plot_autocorr", "plot_waterfall", "plot_fringe_diag",
    "plot_phase_vs_freq",
    "plot_autocorr_array", "plot_waterfall_array",
    # CLI mirrors
    "run", "run_autocorr", "run_waterfall", "run_fringe_stop",
    "run_build_layout", "run_sync_wiring",
]

_COMMANDS = {
    "casm-viz-data-span": "casm_vis_analysis.cli:data_span_main",
    "casm-autocorr": "casm_vis_analysis.cli:autocorr_main",
    "casm-waterfall": "casm_vis_analysis.cli:waterfall_main",
    "casm-fringe-stop": "casm_vis_analysis.cli:fringe_stop_main",
    "casm-fit-positions": "casm_vis_analysis.cli:fit_positions_main",
    "casm-build-layout": "casm_vis_analysis.layout.build:main",
    "casm-sync-wiring":  "casm_vis_analysis.layout.sync:main",
}


def run(cmd):
    """Run a CLI command in-process. For Jupyter use.

    Example::

        from casm_vis_analysis import run
        run("casm-autocorr --obs 2026-02-28-00:41:50 --data-dir /path --show")
    """
    import shlex

    tokens = shlex.split(cmd)
    name, argv = tokens[0], tokens[1:]

    if name not in _COMMANDS:
        raise ValueError(f"Unknown command: {name!r}. "
                         f"Available: {', '.join(_COMMANDS)}")

    module_path, func_name = _COMMANDS[name].rsplit(":", 1)
    from importlib import import_module
    func = getattr(import_module(module_path), func_name)
    func(argv)
