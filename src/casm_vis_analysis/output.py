"""Structured output directory management and NPZ saving."""

from pathlib import Path

import numpy as np


def make_output_dir(base_dir, obs_id):
    """Create structured output directories.

    Parameters
    ----------
    base_dir : str or Path
        Base output directory.
    obs_id : str
        Observation identifier.

    Returns
    -------
    dirs : dict
        Keys: 'base', 'autocorr', 'waterfall', 'fringe_stop'.
        Values: Path objects (directories are created).
    """
    base = Path(base_dir) / obs_id
    dirs = {
        "base": base,
        "autocorr": base / "autocorr",
        "waterfall": base / "waterfall",
        "fringe_stop": base / "fringe_stop",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def save_results(path, **arrays):
    """Save arrays to NPZ file.

    Parameters
    ----------
    path : str or Path
        Output file path (should end in .npz).
    **arrays
        Named arrays to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **arrays)
