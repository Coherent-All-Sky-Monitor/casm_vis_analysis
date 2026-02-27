"""Tests for casm_vis_analysis.output."""

import numpy as np
import pytest
from casm_vis_analysis.output import make_output_dir, save_results


class TestMakeOutputDir:
    def test_creates_directories(self, tmp_path):
        dirs = make_output_dir(tmp_path, "test_obs")
        assert dirs["base"].exists()
        assert dirs["autocorr"].exists()
        assert dirs["waterfall"].exists()
        assert dirs["fringe_stop"].exists()

    def test_directory_structure(self, tmp_path):
        dirs = make_output_dir(tmp_path, "2026-01-27")
        assert dirs["autocorr"] == tmp_path / "2026-01-27" / "autocorr"

    def test_idempotent(self, tmp_path):
        dirs1 = make_output_dir(tmp_path, "obs1")
        dirs2 = make_output_dir(tmp_path, "obs1")
        assert dirs1 == dirs2


class TestSaveResults:
    def test_saves_and_loads(self, tmp_path):
        path = tmp_path / "results.npz"
        arr = np.arange(10)
        save_results(path, data=arr, label=np.array("test"))

        loaded = np.load(path)
        np.testing.assert_array_equal(loaded["data"], arr)

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "results.npz"
        save_results(path, x=np.array([1, 2, 3]))
        assert path.exists()
