"""Prepared-array rendering: frequency orientation, gaps and normalization."""
import numpy as np
import matplotlib.pyplot as plt
import pytest

from casm_vis_analysis.solar_waterfall import plot_dynamic_spectrum
from casm_vis_analysis import solar_waterfall


def test_array_renderer_preserves_gaps_and_zero_channels():
    samples = np.array([[2., 0., 4.], [4., 0., np.nan], [6., 0., 8.]])
    fig = plot_dynamic_spectrum(samples, [1000, 1002, 1010], [450, 440, 430],
                                integration_s=2, quantity="Correlated amplitude")
    mesh = fig.axes[0].collections[0]
    cells = mesh.get_array().reshape(3, 5)
    np.testing.assert_allclose(cells[0, ::2], [.5, 1, 1.5])
    assert np.ma.getmaskarray(cells)[1].all()
    assert np.ma.getmaskarray(cells)[:, 1::2].all()
    assert fig.axes[0].get_ylim()[0] < fig.axes[0].get_ylim()[1]
    assert fig.axes[1].get_ylabel() == "Correlated amplitude"
    assert np.isnan(fig.axes[2].lines[0].get_ydata()).any()
    plt.close(fig)


def test_ascending_frequency_and_save(tmp_path):
    out = tmp_path / "array.png"
    assert plot_dynamic_spectrum(np.ones((2, 3)), [1000, 1002], [430, 440, 450], out) == out
    assert out.stat().st_size > 0


@pytest.mark.parametrize("samples,times,freqs", [
    (np.ones((2, 3), complex), [1, 2], [3, 2, 1]),
    (-np.ones((2, 3)), [1, 2], [3, 2, 1]),
    (np.zeros((2, 3)), [1, 2], [3, 2, 1]),
    (np.ones((2, 3)), [2, 1], [3, 2, 1]),
    (np.ones((2, 3)), [1, 2], [3, 1, 2]),
])
def test_invalid_measurements_fail(samples, times, freqs):
    with pytest.raises(ValueError):
        plot_dynamic_spectrum(samples, times, freqs)


def test_filterbank_front_door_uses_shared_renderer(monkeypatch, tmp_path):
    monkeypatch.setattr(solar_waterfall, "downsample", lambda path, tfac: (
        np.array([[2., 3.], [4., 6.]]), np.array([.5, 1.5]),
        np.array([450., 440.]), {"tstart": 60000., "tsamp": .1}))
    output = tmp_path / "filterbank.png"
    assert solar_waterfall.plot_waterfall("unused.fil", output, "IB", tfac=10,
                                          chans=[0, 1]) == output
    assert output.stat().st_size > 0
