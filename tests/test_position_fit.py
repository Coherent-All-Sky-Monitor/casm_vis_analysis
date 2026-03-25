"""Tests for casm_vis_analysis.position_fit."""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from casm_vis_analysis.position_fit import (
    circular_variance_metric,
    phase_stdev_metric,
    fit_parabola_uncertainty,
    auto_detect_sign,
    scan_x_single_baseline,
    scan_position_single_baseline,
    write_corrected_layout,
    select_cross_plank_refs,
    POSITION_METRICS,
)
from casm_vis_analysis.fringe_stop import geometric_delay
from casm_io.constants import C_LIGHT_M_S


class TestCircularVarianceMetric:
    def test_perfect_coherence_near_zero(self):
        """Constant phase across time → circvar ≈ 0."""
        n_time, n_freq = 50, 32
        phase = np.linspace(0, 0.5, n_freq)  # fixed across time
        vis = np.exp(1j * phase[np.newaxis, :]) * np.ones((n_time, 1))
        score = circular_variance_metric(vis)
        assert score < 0.01

    def test_random_phase_near_one(self):
        """Random phase across time → circvar ≈ 1."""
        rng = np.random.default_rng(42)
        n_time, n_freq = 500, 32
        phase = rng.uniform(-np.pi, np.pi, (n_time, n_freq))
        vis = np.exp(1j * phase)
        score = circular_variance_metric(vis)
        assert score > 0.8

    def test_freq_mask(self):
        """Freq mask selects only some channels."""
        n_time, n_freq = 50, 32
        vis = np.exp(1j * np.zeros((n_time, n_freq)))
        freq_mask = np.zeros(n_freq, dtype=bool)
        freq_mask[10:20] = True
        score = circular_variance_metric(vis, freq_mask=freq_mask)
        assert score < 0.01

    def test_time_mask(self):
        """Time mask selects subset of time samples."""
        n_time, n_freq = 50, 32
        vis = np.exp(1j * np.zeros((n_time, n_freq)))
        time_mask = np.zeros(n_time, dtype=bool)
        time_mask[10:30] = True
        score = circular_variance_metric(vis, time_mask=time_mask)
        assert score < 0.01


class TestPhaseStdevMetric:
    def test_perfect_coherence_near_zero(self):
        n_time, n_freq = 50, 32
        vis = np.exp(1j * np.zeros((n_time, n_freq)))
        score = phase_stdev_metric(vis)
        assert score < 0.01

    def test_random_phase_large(self):
        rng = np.random.default_rng(42)
        n_time, n_freq = 500, 32
        phase = rng.uniform(-np.pi, np.pi, (n_time, n_freq))
        vis = np.exp(1j * phase)
        score = phase_stdev_metric(vis)
        assert score > 1.0  # should be ~ sqrt(2) for uniform random


class TestFitParabolaUncertainty:
    def test_known_quadratic(self):
        """Recover minimum of y = 2*(x - 3)^2 + 1."""
        x = np.linspace(0, 6, 61)
        y = 2.0 * (x - 3.0) ** 2 + 1.0
        x_best, sigma_x, coeffs = fit_parabola_uncertainty(x, y)
        assert abs(x_best - 3.0) < 0.01
        # With a clean parabola and covariance propagation, sigma should be tiny
        assert sigma_x < 0.01
        assert coeffs[0] > 0  # positive curvature

    def test_known_quadratic_with_noise(self):
        """Recover minimum with small noise — sigma should reflect noise level."""
        rng = np.random.default_rng(99)
        x = np.linspace(0, 6, 61)
        y = 2.0 * (x - 3.0) ** 2 + 1.0 + 0.01 * rng.standard_normal(len(x))
        x_best, sigma_x, coeffs = fit_parabola_uncertainty(x, y)
        assert abs(x_best - 3.0) < 0.05
        assert sigma_x < 0.1
        assert sigma_x > 0  # nonzero due to noise

    def test_minimum_at_edge(self):
        """If minimum is near edge, should still return something reasonable."""
        x = np.linspace(0, 5, 51)
        y = (x - 0.0) ** 2 + 0.5  # minimum at x=0 (edge)
        x_best, sigma_x, coeffs = fit_parabola_uncertainty(x, y)
        assert abs(x_best) < 0.5

    def test_too_few_points(self):
        """With 2 points, should return grid minimum."""
        x = np.array([1.0, 2.0])
        y = np.array([0.5, 0.3])
        x_best, sigma_x, coeffs = fit_parabola_uncertainty(x, y)
        assert x_best == 2.0
        assert sigma_x == np.inf


class TestAutoDetectSign:
    def test_detects_correct_sign(self, tiny_freq):
        """Inject geometric delay with known sign, verify detection."""
        rng = np.random.default_rng(42)
        n_time = 50
        source_enu = np.tile([0.5, 0.5, np.sqrt(0.5)], (n_time, 1))
        baseline_enu = np.array([30.0, 2.0, 0.0])

        # Compute geometric delay
        tau_s = geometric_delay(source_enu, baseline_enu)
        freq_hz = tiny_freq * 1e6

        # Create vis with geometric phase using sign=-1 convention
        true_sign = -1
        geo_phase = true_sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        # vis_raw has this geometric phase; fringe-stop with correct sign should remove it
        vis_bl = np.exp(-1j * geo_phase)  # vis that needs sign=-1 to correct

        detected = auto_detect_sign(vis_bl, tiny_freq, source_enu,
                                    baseline_enu)
        assert detected == true_sign


class TestScanXSingleBaseline:
    def test_recovers_known_position(self):
        """Create synthetic vis with known x-position and verify recovery."""
        rng = np.random.default_rng(123)
        n_time = 60
        n_freq = 32
        freq_mhz = np.linspace(468.75, 375.0, n_freq)
        freq_hz = freq_mhz * 1e6

        # Source moving through sky (simplified)
        t = np.linspace(-0.3, 0.3, n_time)
        source_enu = np.column_stack([
            np.sin(t) * np.cos(0.2),
            np.cos(t) * np.cos(0.2),
            np.ones(n_time) * np.sin(0.2),
        ])

        ref_pos = np.array([0.0, 0.0, 0.0])
        true_x = 15.0
        target_pos = np.array([true_x, 2.0, 0.0])
        bl_enu = target_pos - ref_pos

        # Compute geometric delay and create vis with that delay
        tau_s = geometric_delay(source_enu, bl_enu)
        sign = -1
        geo_phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        # Raw vis = unit amplitude with geometric phase embedded (inverted)
        vis_bl = np.exp(-1j * geo_phase)

        time_mask_fit = np.ones(n_time, dtype=bool)
        time_mask_score = np.ones(n_time, dtype=bool)

        x_grid = np.arange(true_x - 3, true_x + 3 + 0.05, 0.1)

        result = scan_x_single_baseline(
            vis_bl, freq_mhz, source_enu, ref_pos,
            y_target=2.0, z_target=0.0,
            x_grid=x_grid, sign=sign,
            time_mask_fit=time_mask_fit,
            time_mask_score=time_mask_score,
        )

        # Best x should be close to true_x
        assert abs(result["best_x"] - true_x) < 0.5, \
            f"Expected ~{true_x}, got {result['best_x']}"
        assert result["sigma_x"] < 5.0
        assert result["best_score"] < result["scores"][0]


class TestScanYSingleBaseline:
    def test_recovers_known_y_position(self):
        """Create synthetic vis with known y-position and verify recovery."""
        rng = np.random.default_rng(456)
        n_time = 80
        n_freq = 32
        freq_mhz = np.linspace(468.75, 375.0, n_freq)
        freq_hz = freq_mhz * 1e6

        # Source arc with significant N-component variation (wide hour-angle range)
        t = np.linspace(-1.0, 1.0, n_time)
        source_enu = np.column_stack([
            np.sin(t) * np.cos(0.5),     # E varies with hour angle
            -np.cos(t) * np.cos(0.5),    # N varies (negative = south)
            np.ones(n_time) * np.sin(0.5),
        ])

        ref_pos = np.array([0.0, 0.0, 0.0])
        true_y = 8.0
        target_pos = np.array([1.0, true_y, 0.0])
        bl_enu = target_pos - ref_pos

        # Compute geometric delay and create vis with that delay
        tau_s = geometric_delay(source_enu, bl_enu)
        sign = -1
        geo_phase = sign * 2 * np.pi * tau_s[:, np.newaxis] * freq_hz[np.newaxis, :]
        vis_bl = np.exp(-1j * geo_phase)

        time_mask_fit = np.ones(n_time, dtype=bool)
        time_mask_score = np.ones(n_time, dtype=bool)

        y_grid = np.arange(true_y - 3, true_y + 3 + 0.05, 0.1)

        result = scan_position_single_baseline(
            vis_bl, freq_mhz, source_enu, ref_pos,
            target_pos_base=np.array([1.0, 0.0, 0.0]),  # wrong y to start
            pos_grid=y_grid, axis=1, sign=sign,
            time_mask_fit=time_mask_fit,
            time_mask_score=time_mask_score,
        )

        assert abs(result["best_y"] - true_y) < 0.5, \
            f"Expected ~{true_y}, got {result['best_y']}"
        assert result["sigma_y"] < 5.0
        assert result["best_score"] < result["scores"][0]


class TestWriteCorrectedLayout:
    def test_roundtrip(self, tmp_path):
        """Write a CSV, update x, verify the updated value."""
        original = tmp_path / "layout.csv"
        output = tmp_path / "layout_corrected.csv"

        # Create a simple layout CSV
        with open(original, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "antenna_id", "x_m", "y_m", "z_m", "snap_id", "adc"])
            writer.writeheader()
            writer.writerow({"antenna_id": "1", "x_m": "10.0",
                             "y_m": "0.0", "z_m": "0.0",
                             "snap_id": "0", "adc": "0"})
            writer.writerow({"antenna_id": "2", "x_m": "20.0",
                             "y_m": "1.0", "z_m": "0.0",
                             "snap_id": "0", "adc": "1"})
            writer.writerow({"antenna_id": "3", "x_m": "30.0",
                             "y_m": "2.0", "z_m": "0.0",
                             "snap_id": "0", "adc": "2"})

        write_corrected_layout(original, output,
                               antenna_ids=[2], fitted_x=[22.5])

        with open(output, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert float(rows[0]["x_m"]) == 10.0  # unchanged
        assert abs(float(rows[1]["x_m"]) - 22.5) < 1e-5  # updated
        assert float(rows[2]["x_m"]) == 30.0  # unchanged


class TestWriteCorrectedLayoutY:
    def test_updates_y_column(self, tmp_path):
        """Write a CSV, update y, verify the updated value."""
        original = tmp_path / "layout.csv"
        output = tmp_path / "layout_corrected.csv"

        with open(original, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "antenna", "x", "y", "z", "snap", "adc"])
            writer.writeheader()
            writer.writerow({"antenna": "1", "x": "10.0",
                             "y": "5.0", "z": "0.0",
                             "snap": "0", "adc": "0"})
            writer.writerow({"antenna": "2", "x": "20.0",
                             "y": "6.0", "z": "0.0",
                             "snap": "0", "adc": "1"})

        write_corrected_layout(original, output,
                               antenna_ids=[2], fitted_y=[7.5])

        with open(output, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert float(rows[0]["y"]) == 5.0  # unchanged
        assert abs(float(rows[1]["y"]) - 7.5) < 1e-5  # updated
        assert float(rows[1]["x"]) == 20.0  # x unchanged


class TestSelectCrossPlankRefs:
    def test_maximizes_row_separation(self):
        """Reference should come from the furthest row."""
        antenna_ids = [1, 2, 3, 4]
        rows = {1: "N21", 2: "N21", 3: "C", 4: "C"}
        positions = {
            1: np.array([0.0, 0.0, 0.0]),
            2: np.array([1.0, 0.0, 0.0]),
            3: np.array([0.0, -10.0, 0.0]),
            4: np.array([1.0, -10.0, 0.0]),
        }
        ref_map = select_cross_plank_refs(antenna_ids, rows, positions)
        # N21 antennas should get C ref (furthest), and vice versa
        assert rows[ref_map[1]] == "C"
        assert rows[ref_map[2]] == "C"
        assert rows[ref_map[3]] == "N21"
        assert rows[ref_map[4]] == "N21"

    def test_picks_closest_to_x0(self):
        """Among candidates in the best row, pick closest to x=0."""
        antenna_ids = [1, 2, 3]
        rows = {1: "A", 2: "B", 3: "B"}
        positions = {
            1: np.array([5.0, 0.0, 0.0]),
            2: np.array([0.1, -10.0, 0.0]),
            3: np.array([3.0, -10.0, 0.0]),
        }
        ref_map = select_cross_plank_refs(antenna_ids, rows, positions)
        # Ant 1 (row A) should get ant 2 (row B, closer to x=0)
        assert ref_map[1] == 2

    def test_three_rows(self):
        """With three rows, each row gets ref from the furthest row."""
        antenna_ids = [1, 2, 3]
        rows = {1: "top", 2: "mid", 3: "bot"}
        positions = {
            1: np.array([0.0, 10.0, 0.0]),
            2: np.array([0.0, 0.0, 0.0]),
            3: np.array([0.0, -10.0, 0.0]),
        }
        ref_map = select_cross_plank_refs(antenna_ids, rows, positions)
        # top(y=10) → bot(y=-10), bot(y=-10) → top(y=10), mid → either top or bot
        assert ref_map[1] == 3  # top → bot (20m separation)
        assert ref_map[3] == 1  # bot → top (20m separation)
        assert ref_map[2] in [1, 3]  # mid → top or bot (10m either way)


class TestMetricsRegistry:
    def test_all_registered(self):
        assert "circvar" in POSITION_METRICS
        assert "stdev" in POSITION_METRICS
