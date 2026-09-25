from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from casm_io.correlator.mapping import AntennaMapping
from casm_vis_analysis.beam_power import beam_power_vs_time
from casm_vis_analysis.calibration_checks import exact_stationary_response, peak_metrics, stationary_check


def mapping():
    return AntennaMapping(pd.DataFrame(dict(antenna_id=[1, 2], snap_id=[0, 0], adc=[0, 1],
        packet_index=[0, 1], x_m=[0., 0.], y_m=[0., 0.], z_m=[0., 0.], functional=[1, 1])))


def test_dynamic_spectrum_matches_existing_mean_and_single_cal_application():
    freq = np.array([450., 440., 430.])
    vis = np.ones((7, 3, 3), dtype=complex)
    vis[:, :, 1] = 1j
    data = dict(vis=vis, freq_mhz=freq, time_unix=np.arange(7), freq_mask=np.array([False, True, False]))
    cal = SimpleNamespace(weights=np.array([[1, 1, 1], [1j, 1j, 1j]]),
        ant_ids=[1, 2], frequencies_hz=freq * 1e6)
    result = beam_power_vs_time(data, mapping(), [('fixed', 90, 0)], cal_weights=cal, return_spectrum=True)
    np.testing.assert_allclose(result['spectrum']['fixed'], 2)
    np.testing.assert_allclose(result['power']['fixed'], result['spectrum']['fixed'].mean(axis=1))
    np.testing.assert_array_equal(result['freq_mhz'], [450., 430.])
    assert 'spectrum' not in beam_power_vs_time(data, mapping(), [('fixed', 90, 0)], cal_weights=cal)


def test_compact_triangle_preserves_original_input_labels(monkeypatch):
    from casm_vis_analysis import calibration_checks as checks
    frame = mapping().dataframe.copy()
    frame['packet_index'] = [8, 14]
    original = AntennaMapping(frame)
    monkeypatch.setattr(checks.AntennaMapping, 'load', lambda path: original)
    def read(*args, **kwargs):
        assert kwargs['inputs'] == [8, 14]
        return SimpleNamespace(vis=np.ones((4, 5, 3), dtype=complex), freq_mhz=np.arange(5),
                               time_unix=np.arange(4.), metadata={})
    monkeypatch.setattr(checks, 'read_visibilities', read)
    data, compact, _ = checks.read_selected('unused', 'layout.csv', [1, 2], ['start', 'end'])
    assert compact.packet_index(1) == 0
    assert compact.packet_index(2) == 1
    assert 'input 8' in data['antenna_labels'][1]
    assert 'input 14' in data['antenna_labels'][2]


def test_peak_alignment_rejects_weak_and_edge_channels():
    rng = np.random.default_rng(18)
    minutes = np.linspace(-150, 150, 151)
    pulse = np.exp(-.5 * (minutes / 22)**2)
    spectrum = np.column_stack([pulse, np.roll(pulse, 10), pulse * 0, np.linspace(0, 4, len(pulse))])
    spectrum += rng.normal(0, .01, spectrum.shape)
    model = np.tile(pulse[:, None], (1, 4))
    result = peak_metrics(minutes, spectrum, model, abs(minutes) >= 90, minimum_snr=8)
    assert result['accepted'].tolist() == [True, True, False, False]
    assert abs(result['residual_minutes'][0]) < 5
    assert result['residual_minutes'][1] > 15
    assert np.isnan(result['residual_minutes'][2:]).all()


def test_missing_halfmax_and_offsource_are_not_fabricated():
    minutes = np.arange(20.)
    spec = np.tile(np.arange(20.)[:, None], (1, 3))
    output = peak_metrics(minutes, spec, spec, minutes < 5)
    assert output['summary']['fwhm_min'] is None
    with pytest.raises(ValueError, match='four off-source'):
        peak_metrics(minutes, spec, spec, minutes < 2)


def test_exact_model_unity_for_colocated_array():
    result = exact_stationary_response(mapping(), 'cyg-a', np.array([1.79e9, 1.79e9 + 100]), np.array([440., 450.]), 1.79e9)
    np.testing.assert_allclose(result, 1)


def test_stationary_plot_common_mask_and_artifacts(tmp_path):
    minutes = np.linspace(-150, 150, 61)
    freq = np.array([460., 450., 440., 430.])
    rng = np.random.default_rng(12)
    vis = np.ones((61, 4, 3), dtype=np.complex64)
    vis[:, :, 1] = 10 + np.exp(-.5 * (minutes[:, None] / 25)**2) + rng.normal(0, .001, (61, 4))
    first = SimpleNamespace(weights=np.ones((2, 4)), ant_ids=[1, 2], frequencies_hz=freq * 1e6, flags=np.array([1, 1, 1, 1]))
    second = SimpleNamespace(**{**first.__dict__, 'flags': np.array([1, 0, 1, 1])})
    output = stationary_check(dict(vis=vis, freq_mhz=freq, time_unix=1.79e9 + minutes * 60),
        mapping(), {'candidate': first, 'reference': second}, 'cyg-a', 1.79e9, tmp_path)
    assert output['common_channels'] == 3
    assert (tmp_path / 'transit-candidate.png').is_file()
    assert (tmp_path / 'chromaticity-candidate.png').is_file()
    assert (tmp_path / 'comparison.png').is_file()
    from matplotlib.image import imread
    for filename in ('transit-candidate.png', 'chromaticity-candidate.png', 'comparison.png'):
        np.testing.assert_allclose(imread(tmp_path / filename)[0, 0, :3], 1)
    candidate = np.load(tmp_path / 'transit-candidate.npz')
    reference = np.load(tmp_path / 'transit-reference.npz')
    np.testing.assert_allclose(candidate['spectrum'], reference['spectrum'])
    assert candidate['spectrum'].max() < 3
