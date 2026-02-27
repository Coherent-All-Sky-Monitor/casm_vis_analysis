"""Tests for casm_vis_analysis.delay."""

import numpy as np
import pytest
from casm_vis_analysis.delay import (
    linear_fit,
    linear_apply,
    phasor_fit,
    phasor_apply,
    fit_delay,
    apply_delay,
    build_delay_design_matrix,
    solve_antenna_delays,
    DELAY_MODELS,
)


class TestLinearDelay:
    def test_fit_recovers_injected_slope(self, tiny_freq):
        """Inject a known linear phase slope and recover it."""
        n_time, n_freq = 10, len(tiny_freq)
        delay_ns = 5.0  # 5 ns delay
        slope_true = delay_ns * 2 * np.pi / 1e3  # rad/MHz

        phase = slope_true * tiny_freq[np.newaxis, :]
        vis = np.exp(1j * phase) * np.ones((n_time, 1))
        vis = vis[:, :, np.newaxis]  # (T, F, 1)

        params = linear_fit(vis, tiny_freq)
        np.testing.assert_allclose(params["delay_ns"], delay_ns, atol=0.1)
        assert params["r_squared"] > 0.99

    def test_apply_removes_slope(self, tiny_freq):
        n_time, n_freq = 10, len(tiny_freq)
        slope = 0.05  # rad/MHz
        intercept = 0.3  # rad

        phase = slope * tiny_freq[np.newaxis, :] + intercept
        vis = np.exp(1j * phase) * np.ones((n_time, 1))
        vis = vis[:, :, np.newaxis]

        params = linear_fit(vis, tiny_freq)
        corrected = linear_apply(vis, tiny_freq, params)

        # After correction, phase should be ~flat
        residual_phase = np.angle(np.mean(corrected, axis=0))
        phase_spread = np.std(residual_phase)
        assert phase_spread < 0.1


class TestPhasorDelay:
    def test_fit_apply_roundtrip(self, tiny_freq):
        """Phasor correction should flatten phase."""
        n_time = 10
        rng = np.random.default_rng(42)
        phase = rng.uniform(-np.pi, np.pi, len(tiny_freq))
        vis = np.exp(1j * phase[np.newaxis, :]) * np.ones((n_time, 1))
        vis = vis[:, :, np.newaxis]

        params = phasor_fit(vis, tiny_freq)
        corrected = phasor_apply(vis, tiny_freq, params)

        # After correction, phase should be ~0
        residual = np.angle(np.mean(corrected, axis=0))
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)


class TestRegistry:
    def test_all_models_registered(self):
        assert "linear" in DELAY_MODELS
        assert "per_freq_phasor" in DELAY_MODELS

    def test_fit_delay_dispatches(self, tiny_vis, tiny_freq):
        params = fit_delay(tiny_vis, tiny_freq, model="linear")
        assert "slope" in params

    def test_unknown_model_raises(self, tiny_vis, tiny_freq):
        with pytest.raises(ValueError, match="Unknown model"):
            fit_delay(tiny_vis, tiny_freq, model="nonexistent")


class TestAntennaDecomposition:
    def test_design_matrix_shape(self):
        pairs = [(0, 1), (0, 2), (1, 2)]
        A = build_delay_design_matrix(3, pairs)
        assert A.shape == (3, 3)

    def test_design_matrix_rows(self):
        pairs = [(0, 1), (0, 2)]
        A = build_delay_design_matrix(3, pairs)
        # Row 0: tau_01 = tau_1 - tau_0 → [-1, 1, 0]
        np.testing.assert_array_equal(A[0], [-1, 1, 0])
        np.testing.assert_array_equal(A[1], [-1, 0, 1])

    def test_solve_recovers_delays(self):
        """Inject known antenna delays and recover them."""
        n_ant = 4
        true_delays = np.array([0.0, 3.0, -2.0, 5.0])  # ref ant 0 = 0

        pairs = []
        bl_delays = []
        for i in range(n_ant):
            for j in range(i + 1, n_ant):
                pairs.append((i, j))
                bl_delays.append(true_delays[j] - true_delays[i])

        A = build_delay_design_matrix(n_ant, pairs)
        recovered = solve_antenna_delays(
            np.array(bl_delays), A, ref_ant_idx=0
        )
        np.testing.assert_allclose(recovered, true_delays, atol=1e-10)
