"""CASM correlator visibility analysis."""

__version__ = "0.1.0"

_COMMANDS = {
    "casm-viz-data-span": "casm_vis_analysis.cli:data_span_main",
    "casm-autocorr": "casm_vis_analysis.cli:autocorr_main",
    "casm-waterfall": "casm_vis_analysis.cli:waterfall_main",
    "casm-fringe-stop": "casm_vis_analysis.cli:fringe_stop_main",
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
