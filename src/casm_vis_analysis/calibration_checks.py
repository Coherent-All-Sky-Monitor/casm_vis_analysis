"""Reproducible, offline calibration review on selected visibility triangles.

No solver, weights writer, registry access or deployment lives here. Calibration
weights are applied once through beam_power_vs_time. Masks use True=bad on
input data and True=good in the delay fitter, matching their respective APIs.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from casm_io.constants import C_LIGHT_M_S
from casm_io.correlator import load_format, read_visibilities
from casm_io.correlator.mapping import AntennaMapping

from .beam_power import beam_power_vs_time
from .delay import apply_delay, fit_delay
from .fringe_stop import fringe_stop
from .sources import source_altaz, source_enu


def transit_center(source, day):
    """Altitude maximum, refined to one second, within a UTC date."""
    start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()
    times = start + np.arange(0, 86400, 60)
    peak = times[np.argmax(source_altaz(source, times)[0])]
    fine = np.arange(max(start, peak - 90), min(start + 86400, peak + 91))
    return float(fine[np.argmax(source_altaz(source, fine)[0])])


def read_selected(data_dir, layout_csv, antennas, window):
    """Read only selected inputs and remap the mapping to the compact triangle."""
    mapping = AntennaMapping.load(layout_csv)
    frame = mapping.dataframe
    wired = set(frame.loc[frame.functional == 1, 'antenna_id'].astype(int))
    if not 2 <= len(antennas) <= 32 or not set(antennas) <= wired:
        raise ValueError('Need 2..32 distinct, wired antennas in the selected layout')
    inputs = sorted(mapping.packet_index(antenna) for antenna in antennas)
    if len(inputs) != len(set(inputs)):
        raise ValueError('Duplicate packet indices in antenna selection')
    result = read_visibilities(*window, time_tz='UTC', data_dir=str(data_dir),
                               fmt=load_format('layout_64ant'), inputs=inputs, workers=1, verbose=True)
    times = np.asarray(result.time_unix)
    vis = np.asarray(result.vis)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError('Need at least three ordered visibility integrations')
    keep = np.isfinite(vis).all(axis=(1, 2)) & (np.max(np.abs(vis), axis=(1, 2)) > 0)
    if keep.sum() < 3:
        raise ValueError('Fewer than three nonzero, finite integrations')
    cadence = float(np.median(np.diff(times)))
    if np.max(np.diff(times[keep])) > 3 * cadence:
        raise ValueError('Visibility coverage has a gap exceeding three integrations; choose a complete interval')
    frame = frame[frame.antenna_id.isin(antennas)].copy()
    ranks = {packet: rank for rank, packet in enumerate(inputs)}
    frame['packet_index'] = [ranks[int(packet)] for packet in frame.packet_index]
    frame['functional'] = 1
    frame['include_in_beamforming'] = 1
    print(f'Calibration review selected vis shape: {vis[keep].shape}', flush=True)
    data = dict(vis=vis[keep], freq_mhz=np.asarray(result.freq_mhz), time_unix=times[keep],
                antenna_labels={antenna: mapping.format_antenna(antenna) for antenna in antennas})
    return data, AntennaMapping(frame), dict(requested_window=list(window),
        actual_window_unix=[float(times[keep][0]), float(times[keep][-1])],
        integrations=int(keep.sum()), rejected_integrations=int((~keep).sum()), inputs=inputs,
        metadata=getattr(result, 'metadata', {}))


def phase_stages(data, mapping, ref_ant, source, static=None):
    """Raw, fringe-stopped and fixed instrumental-delay residual on identical axes."""
    raw = fringe_stop(data, mapping, ref_ant=ref_ant, source=source)
    working = dict(data)
    if static is not None:
        working['vis'] = data['vis'] - static[None, :, :]
    stopped = fringe_stop(working, mapping, ref_ant=ref_ant, source=source)
    above = np.asarray(source_altaz(source, data['time_unix'])[0]) >= 10
    if above.sum() < 3:
        raise ValueError('Source must be above 10 degrees for at least three fit integrations')
    fitted = fit_delay(stopped['vis_stopped'], data['freq_mhz'], time_mask=above,
                       freq_mask=stopped['freq_mask'], model='linear')
    stages = {'Raw': raw['vis'], 'Fringe-stopped': stopped['vis_stopped'],
              'Delay-subtracted': apply_delay(stopped['vis_stopped'], data['freq_mhz'], fitted)}
    if static is not None:
        stages = {'Raw': raw['vis'], 'Static-subtracted': stopped['vis'], **dict(list(stages.items())[1:])}
    return stages, stopped, fitted


def render_phase(data, mapping, ref_ant, source, out_dir, static=None):
    """Both representations; curve is angle(complex mean), never mean(angle)."""
    import matplotlib.pyplot as plt
    stages, stopped, fitted = phase_stages(data, mapping, ref_ant, source, static)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    freq = data['freq_mhz']
    minutes = (data['time_unix'] - data['time_unix'][0]) / 60
    names = list(stages)
    outputs = []
    labels = data.get('antenna_labels', {})
    with plt.style.context(['default', {'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18}]):
        for index, target in enumerate(stopped['target_aids']):
            for view in ('curve', 'waterfall'):
                fig, axes = plt.subplots(len(names), 1, figsize=(12, 3 * len(names)), layout='constrained', squeeze=False)
                for row, name in enumerate(names):
                    axis = axes[row, 0]
                    values = stages[name][:, :, index].copy()
                    values[:, ~stopped['freq_mask']] = np.nan
                    if view == 'curve':
                        axis.plot(freq, np.angle(np.nanmean(values[stopped['time_mask']], axis=0)), '.', ms=1.8, color='#6941a5')
                        axis.set(xlabel='Frequency (MHz)', ylabel='Phase (rad)', ylim=(-np.pi, np.pi))
                    else:
                        mesh = axis.pcolormesh(minutes, freq, np.angle(values).T, cmap='twilight', vmin=-np.pi, vmax=np.pi, shading='auto', rasterized=True)
                        axis.set(xlabel='Minutes from first integration', ylabel='Frequency (MHz)')
                        fig.colorbar(mesh, ax=axis, label='Phase\n(rad)', ticks=[-np.pi, 0, np.pi], pad=.02)
                    axis.set_title(name, loc='left', fontsize=11, pad=12)
                fig.suptitle(f'{labels.get(ref_ant, mapping.format_antenna(ref_ant))} × {labels.get(target, mapping.format_antenna(target))} · {source}\nFixed delay fit over the displayed above-horizon interval', fontsize=12)
                filename = f'phase-{target}-{view}.png'
                fig.savefig(dest / filename, dpi=120, facecolor='white')
                plt.close(fig)
                outputs.append(filename)
    np.savez_compressed(dest / 'phase-data.npz', freq_mhz=freq, time_unix=data['time_unix'],
                        target_antennas=stopped['target_aids'], **{name.replace('-', '_'): values for name, values in stages.items()})
    return dict(figures=outputs, fit=fitted, baselines=stopped['target_labels'],
                curve='Phase of complex average over above-horizon fit interval',
                fit_window_unix=data['time_unix'][stopped['time_mask']][[0, -1]].tolist())


def exact_stationary_response(mapping, source, times, freq_mhz, center):
    """Exact equal-amplitude cross-only array factor, normalized to unity on-axis."""
    positions = mapping.dataframe[['x_m', 'y_m', 'z_m']].to_numpy(dtype=float)
    positions = positions - positions[0]
    delta = source_enu(source, times) - source_enu(source, np.array([center]))[0]
    delay = delta @ positions.T / C_LIGHT_M_S
    response = np.empty((len(times), len(freq_mhz)))
    for channel, frequency in enumerate(freq_mhz):
        voltage = np.exp(2j * np.pi * frequency * 1e6 * delay).sum(axis=1)
        response[:, channel] = (np.abs(voltage)**2 - len(positions)) / (len(positions) * (len(positions) - 1))
    return response


def peak_metrics(times_minutes, spectrum, model, off_mask, minimum_snr=5):
    """Channel peak timings with off-source S/N admission and boundary rejection.

    Three-sample boxcar peaks are descriptive, not precision timing fits. Weak
    channels remain NaN. Width is only measured when both half-max crossings
    bracket the broadband peak. No arbitrary calibration pass/fail threshold.
    """
    from scipy.ndimage import uniform_filter1d
    if np.count_nonzero(off_mask) < 4:
        raise ValueError('Need at least four off-source integrations, on both sides of transit')
    smooth = uniform_filter1d(spectrum, size=3, axis=0, mode='nearest')
    background = np.mean(spectrum[off_mask], axis=0)
    noise = np.std(spectrum[off_mask], axis=0, ddof=1)
    indices = np.argmax(smooth, axis=0)
    heights = smooth[indices, np.arange(spectrum.shape[1])] - background
    snr = np.divide(heights, noise, out=np.zeros_like(heights), where=noise > 0)
    accepted = (snr >= minimum_snr) & (indices > 1) & (indices < len(times_minutes) - 2) & np.isfinite(snr)
    measured = np.where(accepted, times_minutes[indices], np.nan)
    expected = times_minutes[np.argmax(model, axis=0)]
    residual = measured - expected
    light = np.mean(spectrum, axis=1)
    light -= np.mean(light[off_mask])
    peak = int(np.argmax(light))
    half = light[peak] / 2
    left = np.flatnonzero(light[:peak] < half)
    right = np.flatnonzero(light[peak + 1:] < half) + peak + 1
    width = None
    if light[peak] > 0 and left.size and right.size:
        lower, upper = left[-1], right[0]
        left_time = np.interp(half, light[lower:lower + 2], times_minutes[lower:lower + 2])
        right_time = np.interp(half, light[upper - 1:upper + 1][::-1], times_minutes[upper - 1:upper + 1][::-1])
        width = float(right_time - left_time)
    rms = np.std(light[off_mask], ddof=1)
    return dict(peak_minutes=measured, expected_minutes=expected, residual_minutes=residual,
                accepted=accepted, snr=snr, light_curve=light,
                summary=dict(accepted_channels=int(accepted.sum()), total_channels=len(accepted),
                median_timing_residual_min=float(np.nanmedian(residual)) if accepted.any() else None,
                timing_scatter_min=float(1.4826 * np.nanmedian(np.abs(residual - np.nanmedian(residual)))) if accepted.any() else None,
                broadband_peak_min=float(times_minutes[peak]), fwhm_min=width,
                peak_off_rms=float(light[peak] / rms) if rms > 0 else None,
                absolute_peak=float(light[peak]), minimum_channel_snr=minimum_snr))


def stationary_check(data, mapping, calibrations, source, center, out_dir,
                     band=(384, 512), off_minutes=90, minimum_snr=5):
    """Stationary-beam review with Sun-screened backgrounds and weighted means."""
    from .calibration_inspection import aligned_weights
    from .calibration_transit import (background_selection, pair_weight_sum, prepare_series,
                                      transit_figure, timing_figure, comparison_figure)
    off, sun_alt, background_info = background_selection(data['time_unix'], center, source, off_minutes)
    if not background_info['valid']:
        raise ValueError(background_info['reason'])
    freq = data['freq_mhz']
    good = (freq >= band[0]) & (freq <= band[1])
    for cal in calibrations.values():
        cal_freq = np.asarray(cal.frequencies_hz) / 1e6
        flags = np.asarray(cal.flags, dtype=bool)
        if cal_freq[0] < cal_freq[-1]:
            cal_freq, flags = cal_freq[::-1], flags[::-1]
        if cal_freq.shape != freq.shape or not np.allclose(cal_freq, freq, atol=1e-5, rtol=0):
            raise ValueError('Calibration and native visibility channel centers differ')
        good &= flags
    if good.sum() < 3:
        raise ValueError('Fewer than three channels survive the common calibration mask')
    background = np.median(data['vis'][off].real, axis=0) + 1j * np.median(data['vis'][off].imag, axis=0)
    working = {**data, 'vis': data['vis'] - background[None], 'freq_mask': ~good}
    altitude, azimuth = source_altaz(source, np.array([center]))
    pointing = [('stationary', float(altitude[0]), float(azimuth[0]))]
    model = exact_stationary_response(mapping, source, data['time_unix'], freq[good], center)
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results, curves = {}, {}
    for label, cal in calibrations.items():
        spectrum = beam_power_vs_time(working, mapping, pointing, cal_weights=cal, return_spectrum=True)['spectrum']['stationary']
        denominator = pair_weight_sum(aligned_weights(cal, sorted(mapping.active_antennas()), freq))[good]
        series = prepare_series(data['time_unix'], freq[good], spectrum, center, off,
                                pair_sum=denominator, model=model, minimum_snr=minimum_snr)
        metrics = series['metrics']
        results[label], curves[label] = metrics['summary'], series
        transit_figure(series, f'{source} · {label} · fixed transit pointing').savefig(dest / f'transit-{label}.png', dpi=130)
        timing_figure(series, label).savefig(dest / f'peak-time-{label}.png', dpi=130)
        np.savez_compressed(dest / f'transit-{label}.npz', time_unix=data['time_unix'], freq_mhz=freq[good],
                            spectrum=spectrum, exact_response=model, pair_weight_sum=denominator,
                            off_mask=off, sun_alt_deg=sun_alt, schema_version=2,
                            **{key: value for key, value in metrics.items() if key != 'summary'})
    comparison_figure(curves, f'{source} · calibration comparison').savefig(dest / 'comparison.png', dpi=130)
    return dict(schema_version=2, background=background_info, metrics=results, pointing_altaz=[float(altitude[0]), float(azimuth[0])],
                center_unix=center, common_channels=int(good.sum()), off_integrations=int(off.sum()),
                antennas=sorted(mapping.active_antennas()),
                warnings=['Cross-only signed power; not absolute sensitivity or SEFD.',
                          'Exact model uses ideal equal antenna amplitudes, without element beam or sky contaminants.',
                          'Channel timings use a three-integration boxcar and peak/background-std selection, not a precision fit.',
                          'A smooth transit or aligned peaks alone does not prove a calibration correct.'])


def render_raw_history(datasets, targets, ref_ant, out_dir):
    """Overlay complex-averaged raw phase for matched solar windows across days.

    Each dataset is (day label, selected data, compact mapping). The caller
    records the windows and supplies the same target/ref antenna IDs each day.
    """
    import matplotlib.pyplot as plt
    with plt.style.context(['default', {'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18}]):
        fig, axes = plt.subplots(len(targets), 1, figsize=(12, 3.5 * len(targets)), layout='constrained', squeeze=False)
        arrays = {}
        for day, data, mapping in datasets:
            labels = data.get('antenna_labels', {})
            raw = fringe_stop(data, mapping, ref_ant=ref_ant, source='sun')
            phase = np.angle(np.mean(raw['vis'], axis=0))
            arrays[day] = phase
            arrays[day + '_freq_mhz'] = data['freq_mhz']
            arrays[day + '_time_unix'] = data['time_unix']
            for index, target in enumerate(targets):
                axis = axes[index, 0]
                axis.plot(data['freq_mhz'], phase[:, raw['target_aids'].index(target)], '.', ms=1.4, alpha=.6, label=day)
                axis.set(xlabel='Frequency (MHz)', ylabel='Raw phase (rad)', ylim=(-np.pi, np.pi),
                         title=f'{labels.get(ref_ant, mapping.format_antenna(ref_ant))} × {labels.get(target, mapping.format_antenna(target))}')
                axis.legend(markerscale=3, fontsize=8)
        fig.suptitle('Raw sawtooth · average phase · matched solar windows\nNo static subtraction', fontsize=12)
        dest = Path(out_dir)
        fig.savefig(dest / 'raw-sawtooth-history.png', dpi=130, facecolor='white')
        plt.close(fig)
        np.savez_compressed(dest / 'raw-sawtooth-history.npz', **arrays)
