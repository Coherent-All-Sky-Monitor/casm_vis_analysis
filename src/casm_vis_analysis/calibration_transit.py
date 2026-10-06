"""Saved-array transit review: explicit backgrounds and comparable cross sums.

No visibility reads, calibration solves, weight writes or deployment operations.
All plotted powers are signed cross-only quantities, not calibrated flux.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .sources import source_altaz

SCHEME_LABELS = {'uniform': 'Uniform', 'inv_var': 'Inverse variance',
                 'inv_std': 'Inverse standard deviation', 'matched': 'Matched'}
COLORS = ['#2459a6', '#c16b21', '#207c5b', '#9b3a40']


def pair_weight_sum(weights):
    """2 sum(i<j) |w_i| |w_j| per channel; equal unit weights give N(N-1).

    Dividing the cross sum by this removes arbitrary voltage-weight scale. It
    does not equalize receiver gains or calibrate the result to Jy.
    """
    amplitudes = np.abs(np.asarray(weights, dtype=complex))
    if amplitudes.ndim != 2 or amplitudes.shape[0] < 2:
        raise ValueError('Need at least two antenna weight spectra')
    total = amplitudes.sum(axis=0)**2 - (amplitudes**2).sum(axis=0)
    return np.where(np.isfinite(total) & (total > 0), total, np.nan)


def background_selection(time_unix, center, source, off_minutes=90, sun_alt=None):
    """Prefer Sun <=-10 degrees for non-solar backgrounds, with warning fallback.

    Four samples in total are required; a one-sided background is labelled.
    If dark samples are insufficient, use Sun-down or all outer samples and
    explicitly warn about solar contamination. Beamforming remains available.
    Altitudes are geometric at saved sample timestamps (not a terrain horizon).
    """
    times = np.asarray(time_unix, dtype=float)
    if times.ndim != 1 or len(times) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Need ordered, finite integration timestamps')
    if not np.isfinite(center) or not np.isfinite(off_minutes) or not 0 < off_minutes < 150:
        raise ValueError('Background boundary must be between 0 and 150 minutes')
    if sun_alt is None:
        from astropy.utils import iers
        # Offline scheduling check; use the locally installed Earth-orientation
        # table, without network requests or changing persistent IERS settings.
        with iers.conf.set_temp('auto_download', False), iers.conf.set_temp('auto_max_age', None):
            sun_alt = source_altaz('sun', times)[0]
    sun_alt = np.asarray(sun_alt, dtype=float)
    if sun_alt.shape != times.shape or not np.isfinite(sun_alt).all():
        raise ValueError('Solar altitude is unavailable for the saved samples')
    minutes = (times - center) / 60
    outer = np.abs(minutes) >= off_minutes
    solar = source.lower() == 'sun'
    off = outer if solar else outer & (sun_alt <= -10)
    policy = 'outer_window' if solar else 'dark'
    warnings = ['Solar self-check: the outer windows may contain solar response.'] if solar else []
    if not solar and off.sum() < 4:
        down = outer & (sun_alt < 0)
        off = down if down.sum() >= 4 else outer
        policy = 'sun_down_fallback' if down.sum() >= 4 else 'solar_contaminated'
        warnings.append('Too few background samples with Sun ≤ −10°. Expect bright solar contamination; beamforming remains available.')
    before, after = int((off & (minutes < 0)).sum()), int((off & (minutes > 0)).sum())
    valid = int(off.sum()) >= 4
    if valid and (before == 0 or after == 0):
        warnings.append(('Post-transit-only' if before == 0 else 'Pre-transit-only') + ' background; before/after background drift cannot be checked.')
    if not solar and np.any(sun_alt > -10) and policy == 'dark':
        warnings.append('Part of this transit window has Sun above −10°; expect solar contamination there. Those samples are excluded from the background.')
    groups = np.split(np.flatnonzero(off), np.flatnonzero(np.diff(np.flatnonzero(off)) > 1)+1)
    intervals = []
    for indices in groups:
        if not len(indices):
            continue
        interval = dict(samples=len(indices), sun_alt_min_deg=float(sun_alt[indices].min()),
                        sun_alt_max_deg=float(sun_alt[indices].max()))
        for name, zone in [('utc', timezone.utc), ('local', ZoneInfo('America/Los_Angeles'))]:
            interval[name] = [datetime.fromtimestamp(float(times[i]), zone).isoformat() for i in indices[[0, -1]]]
        intervals.append(interval)
    info = dict(valid=valid, policy=policy, sun_limit_deg=-10, samples=int(off.sum()),
                before_samples=before, after_samples=after, outer_samples=int(outer.sum()),
                excluded_sun_up=int((outer & ~off & (sun_alt >= 0)).sum()),
                excluded_solar=int((outer & ~off).sum()), intervals=intervals,
                off_minutes=float(off_minutes),
                note='Sun check uses geometric altitude at each integration timestamp.',
                warning=' '.join(warnings) or None,
                reason=None if valid else 'Fewer than four outer-window samples. Choose a wider window or another background boundary.')
    return off, sun_alt, info


def prepare_series(time_unix, freq_mhz, spectrum, center, off, *, pair_sum=None,
                   scale='mean', model=None, minimum_snr=5):
    """Recenter saved fixed-pointing spectra and derive every displayed metric.

    Old products subtract a constant per baseline/channel before beamforming.
    For a fixed pointing this is a time-constant spectrum, so subtracting the
    newly selected off-window mean here cancels the old constant exactly (up
    to saved floating-point precision). No raw visibility reread is required.
    """
    from .calibration_checks import peak_metrics
    times, freq = np.asarray(time_unix), np.asarray(freq_mhz)
    values = np.asarray(spectrum, dtype=float)
    off = np.asarray(off, dtype=bool)
    if values.shape != (len(times), len(freq)) or off.shape != times.shape or not np.isfinite(values).all():
        raise ValueError('Invalid saved transit spectrum')
    if np.count_nonzero(off) < 4:
        raise ValueError('Need at least four background samples')
    values = values - values[off].mean(axis=0)
    if scale == 'mean':
        denominator = np.asarray(pair_sum, dtype=float)
        if denominator.shape != freq.shape or not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError('Saved baseline-weight normalization is unavailable; use Raw sum or rerun this check')
        values = values / denominator
    elif scale != 'sum':
        raise ValueError('Power scale must be mean or sum')
    light = values.mean(axis=1)
    rms = float(np.std(light[off], ddof=1))
    excess = (light - light[off].mean()) / rms if rms > 0 else np.full_like(light, np.nan)
    minutes = (times-center)/60
    metrics = None
    if model is not None:
        metrics = peak_metrics(minutes, values, np.asarray(model), off, minimum_snr)
    return dict(time_unix=times, freq_mhz=freq, minutes=minutes, spectrum=values,
                light_curve=light, excess=excess, off_mask=off, off_rms=rms,
                peak_over_off_rms=float(np.max(excess)) if rms > 0 else None,
                metrics=metrics, model=model, scale=scale)


def _axes(rows=1, height=4.3):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(12, height*rows), layout='constrained', facecolor='white')
    axes = np.atleast_1d(fig.subplots(rows, 1, sharex=True))
    for ax in axes:
        ax.set_facecolor('white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.16)
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(12)
        ax.yaxis.label.set_size(12)
    return fig, axes


def _windows(ax, series):
    hours = series['minutes']/60
    mask = series['off_mask']
    groups = np.split(np.flatnonzero(mask), np.flatnonzero(np.diff(np.flatnonzero(mask)) > 1)+1)
    half = np.median(np.diff(hours))/2
    for indices in groups:
        if len(indices):
            ax.axvspan(hours[indices[0]]-half, hours[indices[-1]]+half, color='#64748b', alpha=.12)
    ax.axvline(0, color='#526173', ls=':', lw=1)


def power_label(scale):
    return 'Mean weighted cross-power (counts)' if scale == 'mean' else 'Weighted cross-power sum (counts)'


def _frequency_display(freq, spectrum):
    """Keep missing channels blank instead of stretching neighbours across gaps."""
    freq = np.asarray(freq)
    if len(freq) < 2:
        return freq, spectrum
    differences = np.diff(freq)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError('Saved frequency centers must be ordered')
    step = np.min(np.abs(differences)) * np.sign(differences[0])
    indices = np.rint((freq-freq[0])/step).astype(int)
    if indices[-1] > 65535 or not np.allclose(freq, freq[0]+indices*step, rtol=0, atol=1e-5):
        raise ValueError('Saved frequency centers are not on a bounded regular channel grid')
    if indices[-1] == len(freq)-1:
        return freq, spectrum
    full = freq[0]+np.arange(indices[-1]+1)*step
    display = np.full((spectrum.shape[0], len(full)), np.nan)
    display[:,indices] = spectrum
    return full, display


def transit_figure(series, title):
    """99th-percentile display contrast; saturated tails remain in every statistic."""
    fig, axes = _axes(2, 3.4)
    spectrum = series['spectrum']
    limit = max(float(np.percentile(np.abs(spectrum), 99)), np.finfo(float).tiny)
    hours = series['minutes']/60
    freq, display = _frequency_display(series['freq_mhz'], spectrum)
    mesh = axes[0].pcolormesh(hours, freq, display.T, cmap='RdBu_r',
                              vmin=-limit, vmax=limit, shading='auto', rasterized=True)
    extend = 'both' if (spectrum < -limit).any() and (spectrum > limit).any() else 'min' if (spectrum < -limit).any() else 'max' if (spectrum > limit).any() else 'neither'
    fig.colorbar(mesh, ax=axes[0], label=power_label(series['scale']), extend=extend, pad=.02)
    axes[0].set(ylabel='Frequency (MHz)', title=title)
    axes[1].plot(hours, series['light_curve'], color=COLORS[0], lw=1.8)
    axes[1].set(xlabel='Hours from predicted meridian transit', ylabel=power_label(series['scale']))
    for ax in axes:
        _windows(ax, series)
    fig.supxlabel('Colour limits: 99th percentile of |power|; extremes saturate. All samples remain in the curve.\nShading marks the background samples.', fontsize=10)
    return fig


def comparison_figure(series, title):
    """Weighted power and measured background-standard-deviation comparison."""
    fig, axes = _axes(2, 3.3)
    for (label, item), color in zip(series.items(), COLORS):
        hours = item['minutes']/60
        axes[0].plot(hours, item['light_curve'], color=color, label=label, lw=1.8)
        axes[1].plot(hours, item['excess'], color=color, label=label, lw=1.8)
    first = next(iter(series.values()))
    axes[0].set(title=title, ylabel=power_label(first['scale']))
    axes[1].set(xlabel='Hours from predicted meridian transit', ylabel='Excess / background std')
    for ax in axes:
        _windows(ax, first)
    axes[0].legend(fontsize=11, ncol=2)
    fig.supxlabel('Excess = power − background mean. Background std includes sky variations and interference.', fontsize=10)
    return fig


def timing_figure(series, title):
    """Single optional peak-time panel; no duplicate residual subplot."""
    fig, (ax,) = _axes()
    metrics = series['metrics']
    if metrics is None:
        raise ValueError('No saved ideal response for channel timing')
    ax.plot(series['freq_mhz'], metrics['peak_minutes'], '.', ms=4, color=COLORS[0], label='Measured peak')
    ax.plot(series['freq_mhz'], metrics['expected_minutes'], color=COLORS[1], label='Ideal array prediction')
    ax.set(xlabel='Frequency (MHz)', ylabel='Minutes from predicted transit', title=f'{title} · peak time vs frequency')
    ax.legend(fontsize=11)
    fig.supxlabel('Strongest 3-integration-smoothed feature in the full window; this does not identify the source.\nWeak channels and window-edge peaks are omitted.', fontsize=10)
    return fig
