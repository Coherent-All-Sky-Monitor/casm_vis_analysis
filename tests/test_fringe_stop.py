"""Tests for casm_vis_analysis.fringe_stop."""

import numpy as np
import pytest
from casm_vis_analysis.fringe_stop import (
    compute_baselines_enu,
    geometric_delay,
    fringe_stop,
)


class TestComputeBaselinesENU:
    def test_shape(self, tiny_positions):
        bl = compute_baselines_enu(tiny_positions, 0, [1, 2, 3])
        assert bl.shape == (3, 3)

    def test_ref_to_self_is_zero(self, tiny_positions):
        bl = compute_baselines_enu(tiny_positions, 0, [0])
        np.testing.assert_allclose(bl, 0.0)

    def test_antisymmetric(self, tiny_positions):
        bl_01 = compute_baselines_enu(tiny_positions, 0, [1])
        bl_10 = compute_baselines_enu(tiny_positions, 1, [0])
        np.testing.assert_allclose(bl_01, -bl_10)


class TestGeometricDelay:
    def test_single_baseline_shape(self, tiny_positions, n_time):
        source = np.tile([0.0, 0.0, 1.0], (n_time, 1))  # zenith
        bl = tiny_positions[1] - tiny_positions[0]
        tau = geometric_delay(source, bl)
        assert tau.shape == (n_time,)

    def test_multi_baseline_shape(self, tiny_positions, n_time):
        source = np.tile([1.0, 0.0, 0.0], (n_time, 1))  # due East
        bl = tiny_positions[1:] - tiny_positions[0]
        tau = geometric_delay(source, bl)
        assert tau.shape == (n_time, 3)

    def test_zenith_zero_delay(self, n_time):
        """Source at zenith → zero delay for any horizontal baseline."""
        source = np.tile([0.0, 0.0, 1.0], (n_time, 1))
        bl = np.array([[100.0, 0.0, 0.0]])  # pure East baseline
        tau = geometric_delay(source, bl)
        np.testing.assert_allclose(tau, 0.0, atol=1e-15)


class TestFringeStop:
    def test_output_keys(self, tiny_vis, tiny_freq):
        n_time = tiny_vis.shape[0]
        tau = np.zeros((n_time, tiny_vis.shape[2]))
        result = fringe_stop(tiny_vis, tiny_freq, tau)
        expected_keys = {
            "vis_raw", "vis_stopped", "vis_for_calibration",
            "geometric_phase", "tau_s", "sign", "freq_mhz",
        }
        assert set(result.keys()) == expected_keys

    def test_vis_for_calibration_equals_vis_stopped(self, tiny_vis, tiny_freq):
        n_time = tiny_vis.shape[0]
        tau = np.zeros((n_time, tiny_vis.shape[2]))
        result = fringe_stop(tiny_vis, tiny_freq, tau)
        np.testing.assert_array_equal(
            result["vis_for_calibration"], result["vis_stopped"]
        )

    def test_zero_delay_identity(self, tiny_vis, tiny_freq):
        """Zero delay should not change visibilities."""
        n_time = tiny_vis.shape[0]
        tau = np.zeros((n_time, tiny_vis.shape[2]))
        result = fringe_stop(tiny_vis, tiny_freq, tau)
        np.testing.assert_allclose(result["vis_stopped"], tiny_vis, atol=1e-6)

    def test_roundtrip(self, tiny_vis, tiny_freq):
        """Fringe-stop with sign=-1 then sign=+1 should recover original."""
        n_time, n_freq, n_bl = tiny_vis.shape
        rng = np.random.default_rng(99)
        tau = rng.uniform(-1e-7, 1e-7, (n_time, n_bl))

        result1 = fringe_stop(tiny_vis, tiny_freq, tau, sign=-1)
        result2 = fringe_stop(result1["vis_stopped"], tiny_freq, tau, sign=+1)
        np.testing.assert_allclose(result2["vis_stopped"], tiny_vis, atol=1e-5)

    def test_shape_preserved(self, tiny_vis, tiny_freq):
        n_time = tiny_vis.shape[0]
        tau = np.zeros((n_time, tiny_vis.shape[2]))
        result = fringe_stop(tiny_vis, tiny_freq, tau)
        assert result["vis_stopped"].shape == tiny_vis.shape
