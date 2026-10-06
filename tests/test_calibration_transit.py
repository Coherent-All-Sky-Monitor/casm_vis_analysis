import numpy as np
import pytest

from casm_vis_analysis.calibration_transit import (
    background_selection, pair_weight_sum, prepare_series, transit_figure,
    timing_figure, comparison_figure,
)


def fixture():
    minutes = np.linspace(-150, 150, 131)
    off = abs(minutes) >= 90
    freq = np.linspace(485, 390, 9)
    signal = np.exp(-.5*(minutes/20)**2) + .01*np.sin(minutes)
    return minutes*60, freq, signal[:, None]*np.ones((1, len(freq))), off


def test_weighted_mean_removes_arbitrary_scale_but_raw_sum_retained():
    times, freq, signal, off = fixture()
    schemes = [np.ones((3, 9)), np.array([1., .25, .0625])[:, None]*np.ones((1,9))]
    expected = signal-signal[off].mean(axis=0)
    for weights in schemes:
        explicit = 2*sum(weights[i]*weights[j] for i in range(3) for j in range(i+1,3))
        np.testing.assert_allclose(pair_weight_sum(weights), explicit)
        for voltage_scale in [1, 200]:
            denominator = pair_weight_sum(weights*voltage_scale)
            saved = signal*denominator
            mean = prepare_series(times, freq, saved, 0, off, pair_sum=denominator)
            raw = prepare_series(times, freq, saved, 0, off, scale='sum')
            np.testing.assert_allclose(mean['spectrum'], expected, atol=1e-12)
            np.testing.assert_allclose(raw['spectrum'], expected*denominator, atol=1e-8)
            np.testing.assert_allclose(mean['excess'], raw['excess'], atol=1e-10)


def test_sun_screening_requires_both_sides_and_solar_exception():
    times, _, _, outer = fixture()
    sun = np.full(len(times), -20.)
    sun[:15] = 5
    off, _, info = background_selection(times, 0, 'cyg-a', sun_alt=sun)
    assert info['valid'] and info['excluded_sun_up'] == 15
    np.testing.assert_array_equal(off, outer & (sun < 0))
    assert sum(i['samples'] for i in info['intervals']) == off.sum()
    sun[times < 0] = 10
    off, _, info = background_selection(times, 0, 'cyg-a', sun_alt=sun)
    assert not info['valid'] and info['before_samples'] == 0
    off, _, info = background_selection(times, 0, 'sun', sun_alt=sun)
    assert info['valid'] and info['warning'] and info['policy'] == 'outer_window'
    np.testing.assert_array_equal(off, outer)
    with pytest.raises(ValueError): background_selection(times, 0, 'sun', off_minutes=151, sun_alt=sun)


def test_recenter_removes_old_constant_template_and_requires_normalizer():
    times, freq, values, off = fixture()
    first = prepare_series(times, freq, values, 0, off, scale='sum')
    second = prepare_series(times, freq, values + np.arange(9)[None, :]*300, 0, off, scale='sum')
    np.testing.assert_allclose(first['spectrum'], second['spectrum'], atol=1e-11)
    with pytest.raises(ValueError, match='normalization'):
        prepare_series(times, freq, values, 0, off)
    invalid = off.copy()
    invalid[times < -149*60] = False
    invalid[(times < 0) & (times > -147*60)] = False
    with pytest.raises(ValueError, match='both sides'):
        prepare_series(times, freq, values, 0, invalid, scale='sum')


def test_percentile_contrast_preserves_outlier_and_single_timing_panel():
    times, freq, values, off = fixture()
    model = values.copy()
    values[60, 3] = 1e8
    series = prepare_series(times, freq, values, 0, off, scale='sum', model=model)
    figure = transit_figure(series, 'Synthetic transit')
    image = figure.axes[0].collections[0]
    assert image.get_clim()[1] < 100 and series['spectrum'][60,3] > 1e7
    np.testing.assert_allclose(figure.axes[1].lines[0].get_ydata(), series['spectrum'].mean(axis=1))
    assert len(timing_figure(series, 'Timing').axes) == 1
    assert len(comparison_figure({'Example':series}, 'Comparison').axes) == 2


def test_flagged_frequency_gap_stays_blank_without_changing_values():
    from casm_vis_analysis.calibration_transit import _frequency_display
    freq = np.array([480.,479.,470.,469.])
    values = np.arange(12.).reshape(3,4)
    full, display = _frequency_display(freq,values)
    np.testing.assert_allclose(full,np.arange(480.,468.,-1))
    np.testing.assert_array_equal(display[:,[0,1,10,11]],values)
    assert np.isnan(display[:,2:10]).all()
    full, display = _frequency_display(freq[::-1],values[:,::-1])
    assert np.all(np.diff(full)>0) and np.isnan(display[:,2:10]).all()
