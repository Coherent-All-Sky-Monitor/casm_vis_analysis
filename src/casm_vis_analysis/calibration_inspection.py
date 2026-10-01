"""Saved, read-only array diagnostics for a fixed SVD calibration.

No solving, calibration writing, or hardware access. Native-channel spectra and
bounded time/frequency previews are saved once; selecting a display reference
does not change the solve. Baseline files use V(first antenna, second antenna).
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np

from casm_io.correlator.baselines import triu_flat_index
from .fringe_stop import compute_baselines_enu, geometric_delay, fringe_stop_array
from .sources import source_enu, source_altaz

SCHEMES = ('uniform', 'inv_var', 'inv_std', 'matched')


def aligned_weights(cal, antennas, freq):
    """Return applied calibration weights (antenna, channel); reject mismatches."""
    cf = np.asarray(cal.frequencies_hz) / 1e6
    weights = np.asarray(cal.weights)
    bad = ~np.asarray(cal.flags, bool)  # calibration files: True = good
    if cf.shape != np.shape(freq):
        raise ValueError('Calibration and visibility channel counts differ')
    if np.allclose(cf, freq, rtol=0, atol=1e-5):
        pass
    elif np.allclose(cf[::-1], freq, rtol=0, atol=1e-5):
        weights, bad = weights[:, ::-1], bad[::-1]
    else:
        raise ValueError('Calibration and visibility frequency grids differ')
    rows = {int(a): i for i, a in enumerate(cal.ant_ids)}
    result = np.full((len(antennas), len(freq)), np.nan + 1j*np.nan)
    for i, antenna in enumerate(antennas):
        if antenna in rows:
            result[i] = weights[rows[antenna]]
    result[:, bad] = np.nan
    result[np.abs(result) == 0] = np.nan
    return result


def amplitude_profiles(noise_power, source_amplitude, good):
    """Voltage amplitudes: 1, 1/N, 1/sqrt(N), s/N; max=1 per channel.

    N is positive off-source raw auto-power (voltage variance), NOT variance of
    auto-power over time. s is coherent source voltage response. Invalid inputs
    are omitted (NaN), never interpreted as exceptionally quiet antennas.
    """
    noise = np.asarray(noise_power, float)
    source = np.asarray(source_amplitude, float)
    if noise.shape != source.shape or noise.ndim != 2 or len(good) != noise.shape[1]:
        raise ValueError('Expected matching antenna × frequency noise/source arrays')
    valid = np.isfinite(noise) & (noise > 0) & np.asarray(good)[None, :]
    result = {'uniform': np.where(np.asarray(good)[None, :], np.ones_like(noise), np.nan)}
    with np.errstate(divide='ignore', invalid='ignore'):
        values = {'inv_var': 1/noise, 'inv_std': 1/np.sqrt(noise), 'matched': source/noise}
    for name, amplitude in values.items():
        ok = valid & np.isfinite(amplitude) & (amplitude > 0)
        maximum = np.max(np.where(ok, amplitude, 0), axis=0)
        result[name] = np.divide(amplitude, maximum, out=np.full_like(noise, np.nan),
                                 where=ok & (maximum[None, :] > 0))
    return result


def source_response(cross_mean, pairs, n_ant):
    """Fit log|<V_ij,fs>| = log s_i + log s_j, excluding autos.

    This is a single-source amplitude-model diagnostic, not a second phase
    solution or an absolute flux calibration. Missing/zero pairs are excluded.
    At least three antennas and a full-rank design are required.
    """
    out = np.full((n_ant, cross_mean.shape[1]), np.nan)
    if n_ant < 3:
        return out
    design = np.zeros((len(pairs), n_ant))
    for row, (i, j) in enumerate(pairs):
        design[row, [i, j]] = 1
    power = np.abs(cross_mean)
    for channel in range(power.shape[1]):
        valid = np.isfinite(power[:, channel]) & (power[:, channel] > 0)
        if valid.sum() < n_ant:
            continue
        solution, _, rank, _ = np.linalg.lstsq(design[valid], np.log(power[valid, channel]), rcond=None)
        if rank == n_ant:
            out[:, channel] = np.exp(solution)
    return out


def _mean(values, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(values, axis=axis)


def _reduce(values, time_bins, frequency_bins):
    # Average complex values before phase; separately save mean |V|.
    ts, fs = [int(t[0]) for t in time_bins], [int(f[0]) for f in frequency_bins]
    finite = np.isfinite(values)
    total = np.add.reduceat(np.add.reduceat(np.where(finite, values, 0), ts, axis=0), fs, axis=1)
    count = np.add.reduceat(np.add.reduceat(finite.astype('int32'), ts, axis=0), fs, axis=1)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)


def save_inspection(data, mapping, cal, out_dir, *, source='sun', background=None,
                    reference_cal=None, subtract_static=True):
    """Save all unique baselines, autos, phase weights and amplitude experiments.

    Input is the compact triangle returned by calibration_checks.read_selected.
    Spectra keep every native channel. Previews average at most 180 time bins ×
    256 frequency bins; all samples contribute, no stride sampling. Phase-only
    correction never normalizes baseline amplitude. Noise is a temporal median
    of raw night autos; the static visibility is a separate complex mean.
    """
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    freq = np.asarray(data['freq_mhz'])
    times = np.asarray(data['time_unix'])
    vis = np.asarray(data['vis'])
    ants = sorted(mapping.active_antennas())
    n = len(ants)
    if vis.shape != (len(times), len(freq), n*(n+1)//2):
        raise ValueError('Inspection requires a compact full triangle')
    if background is not None and not np.array_equal(freq, background['freq_mhz']):
        raise ValueError('Source and night-window frequency grids differ')
    weights = aligned_weights(cal, ants, freq)
    good = np.isfinite(weights).all(axis=0)
    phase_weights = weights / np.abs(weights)
    static = _mean(background['vis'], axis=0) if background is not None else None
    use_static = static is not None and subtract_static
    time_mask = np.asarray(source_altaz(source, times)[0]) >= 10
    if time_mask.sum() < 3:
        raise ValueError('Need three source integrations above 10 degrees')
    tb = np.array_split(np.arange(len(times)), min(180, len(times)))
    fb = np.array_split(np.arange(len(freq)), min(256, len(freq)))
    packet = [mapping.packet_index(a) for a in ants]
    frame = mapping.dataframe.set_index('antenna_id')
    positions = frame.loc[ants, ['x_m', 'y_m', 'z_m']].to_numpy(float)
    direction = source_enu(source, times)
    auto = np.empty((n, len(freq)))
    night = np.full_like(auto, np.nan)
    static_auto = np.full_like(auto, np.nan)
    pair_ids, means = [], []
    residual_sq = np.zeros_like(auto)
    residual_n = np.zeros_like(auto)
    for i, antenna in enumerate(ants):
        diagonal = triu_flat_index(n, packet[i], packet[i])
        auto[i] = np.real(_mean(vis[:, :, diagonal], axis=0))
        if background is not None:
            night[i] = np.median(np.real(background['vis'][:, :, diagonal]), axis=0)
            static_auto[i] = np.real(static[:, diagonal])
        for j in range(i+1, n):
            a, b = sorted((packet[i], packet[j]))
            index = triu_flat_index(n, a, b)
            raw = vis[:, :, index]
            bg = static[:, index] if static is not None else None
            if packet[i] > packet[j]:
                raw = raw.conj()
                if bg is not None:
                    bg = bg.conj()
            working = raw - bg[None, :] if use_static else raw
            tau = geometric_delay(direction, compute_baselines_enu(positions, i, [j]))
            fs = fringe_stop_array(working[:, :, None], freq, tau, sign=-1)
            stopped = fs['vis_stopped'][:, :, 0]
            # geometric model is the phase present in raw data (opposite correction)
            model = np.exp(-1j * fs['geometric_phase'][:, :, 0])
            corrected = stopped * (phase_weights[i]*phase_weights[j].conj())[None, :]
            stages = {'raw': raw, 'geometric': model, 'fringe': stopped, 'svd': corrected}
            if bg is not None:
                stages['static'] = raw - bg[None, :]
            saved = {}
            for name, values in stages.items():
                values = np.where(good[None, :], values, np.nan)
                saved[name+'_mean'] = _mean(values[time_mask], axis=0).astype('complex64')
                saved[name+'_amp'] = _mean(np.abs(values[time_mask]), axis=0).astype('float32')
                saved[name+'_dynamic'] = _reduce(values, tb, fb).astype('complex64')
                saved[name+'_dynamic_amp'] = _reduce(np.abs(values), tb, fb).astype('float32')
            np.savez_compressed(dest / f'pair_{antenna}_{ants[j]}.npz', **saved)
            pair_ids.append((i, j))
            means.append(_mean(np.where(good[None, :], stopped[time_mask], np.nan), axis=0))
            residual = np.angle(saved['svd_mean'])
            valid = np.isfinite(saved['svd_mean']) & (np.abs(saved['svd_mean']) > 0)
            for row in (i, j):
                residual_sq[row] += np.where(valid, residual**2, 0)
                residual_n[row] += valid
        print(f'Inspection baselines: {i+1}/{n} antennas', flush=True)
    response = source_response(np.asarray(means), pair_ids, n) if use_static else np.full_like(auto, np.nan)
    profiles = amplitude_profiles(night, response, good)
    residual = np.sqrt(np.divide(residual_sq, residual_n, out=np.full_like(auto, np.nan), where=residual_n > 0))
    reference = aligned_weights(reference_cal, ants, freq) if reference_cal is not None else np.full_like(weights, np.nan)
    # Preserve all arrays for downloads; drawing endpoints only read these files.
    np.savez_compressed(dest / 'antennas.npz', freq_mhz=freq, antennas=ants,
        weights=weights, phase_weights=phase_weights, reference_weights=reference,
        auto=auto, night=night, static_auto=static_auto, source_response=response,
        residual_rms=residual, **{f'amp_{key}': value for key, value in profiles.items()})
    meta = dict(version=1, source=source, antennas=ants, ref_ant=int(cal.ref_ant_id),
        t0=float(times[0]), t1=float(times[-1]), samples=len(times),
        integration_s=float(np.median(np.diff(times))), static_used=use_static,
        freq_mhz=freq.tolist(), preview_freq_mhz=[float(np.mean(freq[f])) for f in fb],
        preview_times=[float(np.mean(times[t])) for t in tb],
        schemes={key: bool(np.isfinite(value).any()) for key, value in profiles.items()},
        stages=list(stages), curve='Phase of complex mean; magnitude is mean |V|; source above 10°',
        preview='All samples averaged into at most 180 time × 256 frequency cells',
        amplitude_method='N = time-median raw night auto-power; s = log-linear least-squares fit of static-subtracted, fringe-stopped cross amplitudes; amplitudes max-normalized per channel',
        warning='Matched assumes a dominant coherent source and independent antenna noise; validate on independent sky data. Int8 preview excludes beam steering and downstream detector quantization.')
    (dest / 'inspection.json').write_text(json.dumps(meta))
    return meta


def render_history(datasets, calibrations, reference, out_dir):
    """One full-sized figure per reference baseline; fixed cal across all days.

    datasets: (date label, compact data, mapping). Curves use a complex time
    mean of each day's fringe-stopped data, not an average of phase angles.
    """
    import matplotlib.pyplot as plt
    from .fringe_stop import fringe_stop
    dest = Path(out_dir)
    common = sorted(set.intersection(*(set(map(int, c.ant_ids)) for c in calibrations.values())))
    if reference not in common:
        raise ValueError('History reference must exist in every calibration')
    comparisons = []
    for day, data, mapping in datasets:
        weights = {name: aligned_weights(cal, common, data['freq_mhz']) for name, cal in calibrations.items()}
        good = np.logical_and.reduce([np.isfinite(w).all(axis=0) for w in weights.values()])
        fs = fringe_stop({**data, 'freq_mask': ~good}, mapping, ref_ant=reference, source='sun')
        observed = _mean(fs['vis_stopped'][fs['time_mask']], axis=0)
        observed[~good] = np.nan
        comparisons.append((day, observed, fs['target_aids'], weights, data['freq_mhz']))
    colors = ['#2459a6', '#cc7528', '#207c5b']
    for target in common:
        if target == reference:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, layout='constrained')
        payload = {}
        for day_index, (day, observed, targets, weights, freq) in enumerate(comparisons):
            values = observed[:, targets.index(target)]
            axes[0].plot(freq, np.angle(values), '.', ms=1.4, color=colors[day_index%3], label=day)
            payload[f'{day}_measured'] = values
            for name, w in weights.items():
                corr = w[common.index(reference)] * w[common.index(target)].conj()
                residual = np.angle(values*corr)
                rms = float(np.sqrt(np.nanmean(residual**2)))
                axes[1].plot(freq, residual, '.' if name=='candidate' else 'x', ms=1.7,
                    alpha=1 if name=='candidate' else .35, color=colors[day_index%3],
                    label=f'{day} · {name} · RMS {rms:.2f} rad')
                payload[f'{day}_{name}_residual'] = residual
                if day_index == 0:
                    axes[0].plot(freq, np.angle(corr.conj()), '.', ms=1., color='#202d3c' if name=='candidate' else '#a4acb5', label=f'{name} prediction')
        for ax in axes:
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi,-np.pi/2,0,np.pi/2,np.pi], ['−π','−π/2','0','π/2','π'])
            ax.grid(alpha=.18); ax.legend(fontsize=9, ncol=2)
            ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Fringe-stopped phase (rad)')
        axes[1].set(ylabel='SVD residual phase (rad)', xlabel='Frequency (MHz)')
        axes[1].axhline(0, color='#596775', lw=.8)
        label = datasets[0][2].format_antenna
        fig.suptitle(f'{label(reference)} × {label(target)} · cross-day validation\nSame candidate on every day · no static template reused', fontsize=13)
        fig.savefig(dest/f'history-{reference}-{target}.png', dpi=125, facecolor='white')
        plt.close(fig)
        np.savez_compressed(dest/f'history-{reference}-{target}.npz', freq_mhz=freq, **payload)
    return dict(reference=reference, baselines=len(common)-1, days=[d[0] for d in datasets], static_used=False)


def compare_amplitude_transit(data, mapping, cal, profile_file, source, center, out_dir,
                              band=(390.625, 484.375), off_minutes=90):
    """Fixed pointing, matched samples/mask, four saved amplitude experiments.

    Plot absolute cross-power and source excess divided by measured off-window
    RMS. This is not an absolute sensitivity/SEFD measurement. No weights files
    are written, and the calibration object is not mutated.
    """
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from .beam_power import beam_power_vs_time
    with np.load(profile_file, allow_pickle=False) as saved:
        profiles = dict(saved)
    freq = np.asarray(data['freq_mhz'])
    if not np.array_equal(profiles['freq_mhz'], freq):
        raise ValueError('Amplitude profiles and transit frequency grids differ')
    ants = sorted(mapping.active_antennas())
    rows = [list(profiles['antennas']).index(a) for a in ants]
    weights = aligned_weights(cal, ants, freq)
    weights = weights/np.abs(weights)
    available = {name: profiles['amp_'+name][rows] for name in SCHEMES
                 if np.isfinite(profiles['amp_'+name][rows]).any()}
    good = np.isfinite(weights).all(axis=0) & (freq >= band[0]) & (freq <= band[1])
    for profile in available.values():
        good &= np.isfinite(profile).all(axis=0)
    if good.sum() < 3:
        raise ValueError('Fewer than three channels survive all amplitude schemes')
    minutes = (np.asarray(data['time_unix'])-center)/60
    off = abs(minutes) >= off_minutes
    if (minutes < -off_minutes).sum() < 2 or (minutes > off_minutes).sum() < 2:
        raise ValueError('Amplitude comparison needs off-source integrations on both sides')
    background = np.median(data['vis'][off].real,axis=0)+1j*np.median(data['vis'][off].imag,axis=0)
    working = {**data, 'vis': data['vis']-background[None], 'freq_mask': ~good}
    from .sources import source_altaz
    alt, az = source_altaz(source, np.array([center]))
    point = [('fixed', float(alt[0]), float(az[0]))]
    dest = Path(out_dir)
    fig, axes = plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    output = {}
    colors = ['#2459a6','#cc7528','#207c5b','#9b3a40']
    for (name, amplitude), color in zip(available.items(),colors):
        trial = SimpleNamespace(weights=np.nan_to_num(weights*amplitude), ant_ids=ants,
                                frequencies_hz=freq*1e6)
        result = beam_power_vs_time(working,mapping,point,cal_weights=trial,return_spectrum=True)
        spectrum = result['spectrum']['fixed']
        light = spectrum.mean(axis=1)
        baseline = float(np.mean(light[off])); rms = float(np.std(light[off],ddof=1))
        normalized = (light-baseline)/rms if rms>0 else np.full_like(light,np.nan)
        axes[0].plot(minutes/60,light,color=color,label=name)
        axes[1].plot(minutes/60,normalized,color=color,label=name)
        output[name] = dict(off_rms=rms, peak_over_off_rms=float(np.nanmax(normalized)) if rms>0 else None)
        image, ax = plt.subplots(figsize=(13,4.5),layout='constrained')
        limit = max(float(np.max(np.abs(spectrum))),1e-12)
        mesh=ax.pcolormesh(minutes/60,result['freq_mhz'],spectrum.T,cmap='RdBu_r',vmin=-limit,vmax=limit,shading='auto',rasterized=True)
        image.colorbar(mesh,ax=ax,label='Cross-power (counts)')
        ax.set(xlabel='Hours from meridian transit',ylabel='Frequency (MHz)',title=f'{source} · {name} · fixed transit pointing')
        ax.axvline(0,color='#405369',ls=':',lw=1)
        image.savefig(dest/f'amplitude-{name}-transit.png',dpi=130,facecolor='white');plt.close(image)
        np.savez_compressed(dest/f'amplitude-{name}-transit.npz',time_unix=data['time_unix'],freq_mhz=result['freq_mhz'],spectrum=spectrum,light_curve=light,off_mask=off)
    for ax in axes:
        ax.axvline(0,color='#405369',ls=':',lw=1);ax.grid(alpha=.18);ax.legend(ncol=4)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set(ylabel='Cross-power (counts)',title=f'{source} · amplitude schemes · identical antennas and channels')
    axes[1].set(xlabel='Hours from meridian transit',ylabel='Source excess / off-window RMS')
    fig.savefig(dest/'amplitude-comparison.png',dpi=130,facecolor='white');plt.close(fig)
    return dict(schemes=output,antennas=ants,channels=int(good.sum()),
                note='Max-normalized voltage amplitudes; measured off-window RMS includes sky/RFI, not just thermal noise.')
