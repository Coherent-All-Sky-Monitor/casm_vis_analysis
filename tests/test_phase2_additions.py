"""Tests for Phase 2 additive changes: source_flux, RFIMask, ported helpers,
dict-based fringe_stop wrapper."""

import numpy as np
import pandas as pd
import pytest

from casm_vis_analysis import (
    RFIMask,
    SOURCE_CATALOG,
    SUN_SPECTRAL_INDEX,
    auto_detect_sign,
    coherence_metric,
    fringe_stop,
    fringe_stop_array,
    fringe_stop_single_baseline,
    source_flux,
)


class TestSourceFlux:
    def test_known_sources(self):
        for name in ("cas_a", "cyg_a", "tau_a", "vir_a"):
            f = source_flux(name, np.array([400.0]))
            assert f.shape == (1,)
            assert f[0] == pytest.approx(SOURCE_CATALOG[name]["flux_400"])

    def test_sun_uses_locked_index(self):
        # Locked: -0.1 spectral index
        f = source_flux("sun", np.array([400.0, 800.0]))
        # At 800 MHz the flux should be 800/400 ** -0.1 = 2 ** -0.1
        ratio = f[1] / f[0]
        assert ratio == pytest.approx(2.0 ** SUN_SPECTRAL_INDEX, rel=1e-9)

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown source"):
            source_flux("not-a-source", np.array([400.0]))

    def test_case_insensitive_name(self):
        f1 = source_flux("CAS_A", np.array([400.0]))
        f2 = source_flux("cas_a", np.array([400.0]))
        assert np.allclose(f1, f2)

    def test_hyphen_aliasing(self):
        f = source_flux("cas-a", np.array([400.0]))
        assert f[0] == pytest.approx(SOURCE_CATALOG["cas_a"]["flux_400"])


class TestRFIMask:
    def test_no_default_mask(self):
        # The whole point of the rule: empty mask is the legitimate "no RFI" state.
        m = RFIMask(bad_ranges_mhz=[])
        freqs = np.array([100.0, 200.0, 400.0, 600.0])
        good = m(freqs)
        assert good.all()

    def test_flag_bins_inclusive(self):
        m = RFIMask(bad_ranges_mhz=[(395.0, 405.0)])
        freqs = np.array([390.0, 395.0, 400.0, 405.0, 410.0])
        bad = m.flag_bins(freqs)
        assert (bad == [False, True, True, True, False]).all()

    def test_call_returns_good_mask(self):
        m = RFIMask(bad_ranges_mhz=[(395.0, 405.0)])
        freqs = np.array([390.0, 400.0, 410.0])
        good = m(freqs)
        assert (good == [True, False, True]).all()

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="hi < lo"):
            RFIMask(bad_ranges_mhz=[(450.0, 400.0)])

    def test_multiple_ranges(self):
        m = RFIMask(bad_ranges_mhz=[(100.0, 110.0), (200.0, 210.0)])
        freqs = np.array([105.0, 150.0, 205.0])
        bad = m.flag_bins(freqs)
        assert (bad == [True, False, True]).all()


class TestPortedHelpers:
    def test_coherence_metric_perfect_alignment(self):
        # Constant phase across freq -> coherence = 1
        T, F = 4, 8
        vis = np.ones((T, F), dtype=complex) * np.exp(1j * 0.7)
        coh = coherence_metric(vis)
        assert coh.shape == (T,)
        assert np.allclose(coh, 1.0)

    def test_coherence_metric_random_phase(self):
        rng = np.random.RandomState(42)
        T, F = 4, 1000
        # Random phases across freq -> coherence near 0
        vis = np.exp(1j * rng.uniform(0, 2 * np.pi, (T, F)))
        coh = coherence_metric(vis)
        # Expected ~1/sqrt(F) for random
        assert coh.max() < 0.2

    def test_coherence_metric_freq_mask(self):
        T, F = 2, 10
        vis = np.ones((T, F), dtype=complex)
        mask = np.zeros(F, dtype=bool)
        mask[3:7] = True
        coh = coherence_metric(vis, freq_mask=mask)
        assert coh.shape == (T,)
        assert np.allclose(coh, 1.0)

    def test_fringe_stop_single_baseline_shape(self):
        T, F = 5, 16
        vis = np.ones((T, F), dtype=complex)
        freq_hz = np.linspace(60e6, 80e6, F)
        tau_s = np.linspace(0, 1e-9, T)
        fs = fringe_stop_single_baseline(vis, freq_hz, tau_s)
        assert fs.shape == (T, F)
        assert fs.dtype == complex

    def test_fringe_stop_single_baseline_zero_delay_identity(self):
        T, F = 3, 8
        vis = np.full((T, F), 1.0 + 2.0j)
        freq_hz = np.linspace(60e6, 80e6, F)
        fs = fringe_stop_single_baseline(vis, freq_hz, np.zeros(T))
        assert np.allclose(fs, vis)

    def test_auto_detect_sign_picks_correct(self):
        # Build vis that has phase = +1 * 2pi * f * tau, so sign=-1 should
        # remove it (yielding constant phase -> high coherence).
        T, F = 8, 64
        freq_mhz = np.linspace(60.0, 80.0, F)
        tau_s = np.linspace(0.1e-9, 0.5e-9, T)
        # Construct vis where phase = +2pi*f*tau
        phase = 2 * np.pi * tau_s[:, None] * (freq_mhz[None, :] * 1e6)
        vis = np.exp(1j * phase)
        sign = auto_detect_sign(vis, freq_mhz, tau_s)
        assert sign == -1


class TestFringeStopWrapper:
    """Compose-friendly fringe_stop(data, ant, ref_ant=, source=) wrapper."""

    @pytest.fixture
    def mini_ant(self, tmp_path):
        from casm_io.correlator.mapping import AntennaMapping
        df = pd.DataFrame({
            "antenna_id": [1, 2, 3, 4],
            "snap_id": [0, 0, 0, 0],
            "adc": [0, 1, 2, 3],
            "packet_index": [0, 1, 2, 3],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "functional": [1, 1, 1, 1],
        })
        path = tmp_path / "mini.csv"
        df.to_csv(path, index=False)
        return AntennaMapping.load(path)

    def test_wrapper_returns_fringe_stopped_data(self, mini_ant):
        T, F, n_targets = 4, 8, 3   # ref + 3 targets
        data = {
            "vis": np.ones((T, F, n_targets), dtype=complex),
            "freq_mhz": np.linspace(60.0, 80.0, F),
            "time_unix": np.linspace(1.7e9, 1.7e9 + 100, T),
        }
        out = fringe_stop(data, mini_ant, ref_ant=1, source="sun", sign=-1)
        # Required keys
        for k in (
            "vis", "vis_stopped", "vis_for_calibration",
            "geometric_phase", "tau_s", "freq_mhz", "time_unix",
            "source", "ref_ant", "sign", "target_aids", "target_labels",
        ):
            assert k in out, f"missing key {k}"
        assert out["source"] == "sun"
        assert out["ref_ant"] == 1
        assert out["target_aids"] == [2, 3, 4]
        assert out["vis_stopped"].shape == (T, F, n_targets)
        assert out["vis_for_calibration"] is out["vis_stopped"]

    def test_wrapper_rejects_invalid_ref_ant(self, mini_ant):
        T, F, n_targets = 4, 8, 3
        data = {
            "vis": np.ones((T, F, n_targets), dtype=complex),
            "freq_mhz": np.linspace(60.0, 80.0, F),
            "time_unix": np.linspace(1.7e9, 1.7e9 + 100, T),
        }
        with pytest.raises(ValueError, match="ref_ant=99"):
            fringe_stop(data, mini_ant, ref_ant=99, source="sun")
