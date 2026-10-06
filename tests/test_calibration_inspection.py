from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from casm_io.correlator.mapping import AntennaMapping
from casm_vis_analysis import calibration_inspection as inspect


def test_weight_formulas_and_invalid_noise():
    noise = np.array([[1., 0., np.nan], [4., 9., 4.], [16., -1., 1.]])
    signal = np.ones_like(noise)*np.array([1., 3., 2.])[:, None]
    out = inspect.amplitude_profiles(noise, signal, [True, True, False])
    np.testing.assert_allclose(out['uniform'][:, 0], 1)
    np.testing.assert_allclose(out['inv_var'][:, 0], [1, .25, .0625])
    np.testing.assert_allclose(out['inv_std'][:, 0], [1, .5, .25])
    np.testing.assert_allclose(out['matched'][:, 0], [1, .75, .125])
    assert np.isnan(out['inv_var'][0, 1]) and np.isnan(out['inv_std'][2, 1])
    assert all(np.isnan(value[:, 2]).all() for value in out.values())


def test_source_voltage_response_from_crosses_not_autos():
    amplitude = np.array([[2., 3.], [4., 5.], [7., 8.]])
    pairs = [(0, 1), (0, 2), (1, 2)]
    cross = np.array([amplitude[i]*amplitude[j] for i,j in pairs])
    np.testing.assert_allclose(inspect.source_response(cross, pairs, 3), amplitude)
    assert np.isnan(inspect.source_response(cross[:1], pairs[:1], 2)).all()


def test_history_figure_uses_short_day_legend_and_separate_residuals(tmp_path):
    freq = np.linspace(480, 390, 128)
    days = ['2026-10-02', '2026-10-03', '2026-10-04']
    payload = {}
    for day in days:
        payload[f'{day}_measured'] = np.exp(1j*np.linspace(-9, 9, len(freq)))
        payload[f'{day}_candidate_residual'] = np.full(len(freq), .1)
        payload[f'{day}_reference_residual'] = np.full(len(freq), .2)
        payload[f'{day}_with subtraction_residual'] = np.full(len(freq), 2.)
    fig = inspect.history_figure(freq, payload, days, 'N1E1', 'N2E1',
        candidate_label='Without static subtraction', reference_weights_label='Current deployed weights')
    assert len(fig.axes) == 3
    assert [len(ax.lines) for ax in fig.axes] == [3, 4, 4]
    legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in legend.get_texts()] == days
    assert all(handle.get_markersize() >= 8 for handle in legend.legend_handles)
    assert fig.axes[1].get_title() == 'Without static subtraction'
    assert fig.axes[2].get_title() == 'Current deployed weights'
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), payload[f'{days[0]}_candidate_residual'])
    fig.savefig(tmp_path/'history.png')


def test_render_history_preserves_phase_arrays_and_common_mask(tmp_path, monkeypatch):
    from importlib import import_module
    fringe_stop = import_module('casm_vis_analysis.fringe_stop')
    freq = np.array([480., 450., 400.])
    cal = SimpleNamespace(ant_ids=[9, 10], frequencies_hz=freq*1e6, flags=[True, False, True],
                          weights=np.exp(1j*np.array([[.2, .2, .2], [.7, .7, .7]])))
    mapping = SimpleNamespace(format_antenna=lambda a: f'Ant {a}')
    def stopped(data, *args, **kwargs):
        assert data['freq_mask'].tolist() == [False, True, False]
        return dict(vis_stopped=np.ones((2, 3, 1), complex)*np.exp(.5j),
                    time_mask=np.array([True, True]), target_aids=[10])
    monkeypatch.setattr(fringe_stop, 'fringe_stop', stopped)
    result = inspect.render_history([('2026-10-04', {'freq_mhz':freq}, mapping)],
        {'candidate':cal, 'reference':cal}, 9, tmp_path,
        labels={'candidate_label':'Without static subtraction', 'reference_weights_label':'Current deployed weights'})
    assert result['baselines'] == 1
    with np.load(tmp_path/'history-9-10.npz') as data:
        np.testing.assert_allclose(data['2026-10-04_candidate_residual'][[0, 2]], 0, atol=1e-15)
        assert np.isnan(data['2026-10-04_candidate_residual'][1])


def test_preview_all_samples_contribute_and_complex_mean():
    values = np.arange(35.).reshape(5, 7).astype(complex)
    values[1, 2] = np.nan
    tb, fb = np.array_split(np.arange(5), 3), np.array_split(np.arange(7), 2)
    expected = np.array([[np.nanmean(values[t][:, f]) for f in fb] for t in tb])
    np.testing.assert_allclose(inspect._reduce(values, tb, fb), expected)
    z = np.exp(1j*np.array([[3., -3.]]))
    assert abs(np.angle(inspect._reduce(z, [np.array([0])], [np.array([0,1])])[0,0])) > 3


def test_aligned_weights_reverse_mask_and_missing_antenna():
    cal = SimpleNamespace(frequencies_hz=np.array([400., 450., 500.])*1e6,
        weights=np.array([[1., 2., 3.]]), flags=np.array([True, False, True]), ant_ids=[9])
    result = inspect.aligned_weights(cal, [9, 12], [500.,450.,400.])
    np.testing.assert_allclose(result[0,[0,2]], [3,1])
    assert np.isnan(result[0,1]) and np.isnan(result[1]).all()
    with pytest.raises(ValueError, match='frequency grids'):
        inspect.aligned_weights(cal, [9], [501.,450.,400.])


def test_saved_svd_correction_and_static_separation(tmp_path, monkeypatch):
    n, nf, nt = 3, 8, 5
    mapping = AntennaMapping(pd.DataFrame(dict(antenna_id=[9,10,12], snap_id=[0,0,1], adc=[0,1,0],
        packet_index=[2,0,1], x_m=[0.,0.,0.], y_m=[0.,0.,0.], z_m=[0.,2.,5.], functional=[1,1,1])))
    phase = np.array([.2, .8, -1.3])
    gains = np.exp(1j*phase)
    amplitude = np.array([2.,3.,4.])
    vis = np.empty((nt,nf,6),complex)
    for i in range(n):
        for j in range(i,n):
            a,b=sorted((mapping.packet_index([9,10,12][i]),mapping.packet_index([9,10,12][j])))
            k=inspect.triu_flat_index(n,a,b)
            freq=np.linspace(480,390,nf)
            tau=([0.,2.,5.][j]-[0.,2.,5.][i])/299792458.
            value=amplitude[i]*amplitude[j]*gains[i]*gains[j].conj()*np.exp(2j*np.pi*freq*1e6*tau)
            vis[:,:,k]=value if mapping.packet_index([9,10,12][i])<=mapping.packet_index([9,10,12][j]) else value.conj()
    freq=np.linspace(480,390,nf)
    cal=SimpleNamespace(weights=np.repeat(gains.conj()[:,None],nf,axis=1),frequencies_hz=freq*1e6,
                        flags=np.ones(nf,bool),ant_ids=[9,10,12],ref_ant_id=9)
    monkeypatch.setattr(inspect,'source_altaz',lambda *args:(np.full(nt,50.),np.zeros(nt)))
    monkeypatch.setattr(inspect,'source_enu',lambda *args:np.tile([0,0,1],(nt,1)))
    night=np.ones_like(vis)*.5
    data=dict(vis=vis+night,freq_mhz=freq,time_unix=np.arange(nt,dtype=float))
    result=inspect.save_inspection(data,mapping,cal,tmp_path,background={**data,'vis':night})
    assert result['schemes']==dict.fromkeys(inspect.SCHEMES,True)
    with np.load(tmp_path/'pair_9_10.npz') as pair:
        np.testing.assert_allclose(np.angle(pair['svd_mean']),0,atol=1e-7)
        np.testing.assert_allclose(pair['fringe_mean'],6*np.exp(-.6j),rtol=1e-6)
        np.testing.assert_allclose(pair['raw_mean']-pair['static_mean'],.5,atol=1e-6)
    with np.load(tmp_path/'antennas.npz') as antennas:
        np.testing.assert_allclose(antennas['source_response'],amplitude[:,None]*np.ones((n,nf)))
        np.testing.assert_allclose(antennas['night'],.5)
        np.testing.assert_allclose(antennas['residual_rms'],0,atol=1e-7)


def test_amplitude_transit_same_samples_mask_and_no_cal_mutation(tmp_path, monkeypatch):
    from casm_vis_analysis import sources, calibration_transit
    monkeypatch.setattr(calibration_transit,'source_altaz',lambda source,times:(np.full(len(times),50.),np.full(len(times),180.)))
    ants, nf = [9,10,12], 8
    mapping = AntennaMapping(pd.DataFrame(dict(antenna_id=ants,snap_id=[0,0,1],adc=[0,1,0],
        packet_index=[0,1,2],x_m=[0.,0.,0.],y_m=[0.,0.,0.],z_m=[0.,0.,0.],functional=[1,1,1])))
    freq = np.linspace(480.,390.,nf)
    signal = np.array([1.,2.,3.,20.,3.,2.,1.])
    vis = np.broadcast_to(signal[:,None,None],(7,nf,6)).astype(complex).copy()
    gains = np.exp(1j*np.array([.2,.7,-.5]))
    for i in range(3):
        for j in range(i,3):
            vis[:,:,inspect.triu_flat_index(3,i,j)] *= gains[i]*gains[j].conj()
    original = np.repeat(gains.conj()[:,None],nf,axis=1)
    cal = SimpleNamespace(weights=original.copy(),ant_ids=ants,flags=np.ones(nf,bool),frequencies_hz=freq*1e6)
    noise = np.broadcast_to(np.array([1.,4.,16.])[:,None],(3,nf)).copy()
    noise[0,-1] = np.nan
    profiles = inspect.amplitude_profiles(noise,np.ones_like(noise),np.ones(nf,bool))
    profile = tmp_path/'profiles.npz'
    np.savez(profile,antennas=ants,freq_mhz=freq,**{'amp_'+k:v for k,v in profiles.items()})
    monkeypatch.setattr(sources,'source_altaz',lambda *args:(np.array([50.]),np.array([180.])))
    data = dict(vis=vis,freq_mhz=freq,time_unix=60*np.array([-140.,-120.,-100.,0.,100.,120.,140.]))
    result = inspect.compare_amplitude_transit(data,mapping,cal,profile,'sun',0.,tmp_path)
    assert result['channels'] == nf-1
    assert set(result['schemes']) == set(inspect.SCHEMES)
    np.testing.assert_array_equal(cal.weights,original)
    for name, amplitude in profiles.items():
        scale = 2*sum(amplitude[i,0]*amplitude[j,0] for i in range(3) for j in range(i+1,3))
        with np.load(tmp_path/f'amplitude-{name}-transit.npz') as saved:
            np.testing.assert_allclose(saved['light_curve'],scale*(signal-2),atol=1e-5)
            np.testing.assert_array_equal(saved['freq_mhz'],freq[:-1])
            np.testing.assert_allclose(saved['pair_weight_sum'],scale)
    assert result['schema_version'] == 2 and result['background']['warning']
