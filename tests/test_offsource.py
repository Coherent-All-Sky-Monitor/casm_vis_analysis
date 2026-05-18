"""Tests for the off-source vector-subtraction module.

The regression target is the central use case: a static cross-talk
pedestal injected into every integration must be removed by
``subtract_static_visibility`` when the estimate comes from
``average_visibility`` over a window that contains *only* the pedestal
+ thermal noise.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from casm_vis_analysis import (
    build_static_visibility,
    find_quiet_windows,
    average_visibility,
    subtract_static_visibility,
    plot_offsource_diagnostic,
    save_static_visibility,
    load_static_visibility,
)


# --------------------------------------------------------------------- #
# find_quiet_windows                                                     #
# --------------------------------------------------------------------- #

def test_find_quiet_windows_handles_two_separate_intervals():
    # 1-min cadence over 24 h starting at OVRO local midnight 2026-05-09.
    # Sun is below the horizon roughly 02:46 -> 12:48 UTC May 9 and again
    # after sunset; we ask for sun<0 and expect at least two windows
    # (pre-sunrise + post-sunset).
    import datetime, zoneinfo
    pt = zoneinfo.ZoneInfo("America/Los_Angeles")
    t0 = datetime.datetime(2026, 5, 9, 0, 0, tzinfo=pt).timestamp()
    t = np.arange(t0, t0 + 24 * 3600, 60.0)

    windows = find_quiet_windows(
        t,
        altitude_caps={"sun": 0.0},
        min_duration_s=30 * 60,
    )
    assert len(windows) >= 1
    for w in windows:
        assert w["duration_s"] >= 30 * 60
        assert w["altitudes"]["sun"][2] < 0.0   # max alt < 0


def test_find_quiet_windows_min_duration_filter():
    t = np.linspace(0, 100, 11)   # cadence 10 s; no real sky math here
    # Force all samples "quiet" by capping at +90° altitude.
    windows = find_quiet_windows(
        t, altitude_caps={"sun": 90.0}, min_duration_s=0,
    )
    assert len(windows) == 1
    assert windows[0]["duration_s"] == 100

    # Same data, but require >5 minutes — should drop the window.
    windows = find_quiet_windows(
        t, altitude_caps={"sun": 90.0}, min_duration_s=5 * 60,
    )
    assert windows == []


# --------------------------------------------------------------------- #
# average_visibility                                                     #
# --------------------------------------------------------------------- #

def test_average_visibility_mask_path():
    T, F, B = 20, 8, 3
    rng = np.random.default_rng(0)
    vis = rng.standard_normal((T, F, B)) + 1j * rng.standard_normal((T, F, B))
    data = {"vis": vis, "time_unix": np.arange(T, dtype=float)}

    mask = np.zeros(T, dtype=bool); mask[5:15] = True
    avg = average_visibility(data, time_mask=mask)
    np.testing.assert_allclose(avg, vis[5:15].mean(axis=0))


def test_average_visibility_time_range_path():
    T, F, B = 20, 8, 3
    vis = np.ones((T, F, B), dtype=np.complex64)
    times = np.linspace(1000.0, 1019.0, T)
    data = {"vis": vis, "time_unix": times}

    avg = average_visibility(data, time_range_unix=(1005.0, 1010.0))
    # Six samples in [1005, 1010] inclusive at unit spacing.
    assert avg.shape == (F, B)
    np.testing.assert_allclose(avg, np.ones((F, B)))


def test_average_visibility_applies_freq_mask_to_output():
    T, F, B = 10, 5, 2
    vis = np.full((T, F, B), 2.0 + 1.0j, dtype=np.complex64)
    freq_mask = np.array([False, True, False, True, False])  # True=flagged/bad
    data = {
        "vis": vis,
        "time_unix": np.arange(T, dtype=float),
        "freq_mask": freq_mask,
    }
    avg = average_visibility(data, time_range_unix=(0, T - 1))
    # Unflagged channels should equal the input value; flagged are NaN.
    assert np.isnan(avg[1, 0]) and np.isnan(avg[3, 1])
    np.testing.assert_allclose(avg[0], 2.0 + 1.0j)
    np.testing.assert_allclose(avg[2], 2.0 + 1.0j)
    np.testing.assert_allclose(avg[4], 2.0 + 1.0j)


def test_average_visibility_rejects_both_or_neither_window():
    data = {"vis": np.zeros((4, 2, 1), dtype=np.complex64),
            "time_unix": np.arange(4, dtype=float)}
    with pytest.raises(ValueError):
        average_visibility(data)
    with pytest.raises(ValueError):
        average_visibility(
            data,
            time_mask=np.ones(4, dtype=bool),
            time_range_unix=(0, 3),
        )


# --------------------------------------------------------------------- #
# subtract_static_visibility                                             #
# --------------------------------------------------------------------- #

def test_subtract_removes_injected_pedestal():
    """End-to-end: inject a complex cross-talk pedestal into every
    integration on top of a time-varying source-like signal, average
    a chunk that contains *only* the pedestal + thermal noise, and
    confirm the subtraction recovers the pedestal-free source signal.
    """
    rng = np.random.default_rng(42)
    T, F, B = 120, 16, 6

    # Source-like signal: deterministic phase ramp in time, present
    # only in the second half (samples 60..119).
    sig = np.zeros((T, F, B), dtype=np.complex64)
    for b in range(B):
        ph = np.linspace(0, 2 * np.pi * (b + 1), T)[:, None]
        sig[60:, :, b] = (np.exp(1j * ph) * np.ones((1, F)))[60:, :]

    # Static cross-talk pedestal: distinct complex value per (F, B).
    static = (rng.standard_normal((F, B)) + 1j *
              rng.standard_normal((F, B))).astype(np.complex64)

    # Thermal noise on every integration.
    noise = (0.01 *
             (rng.standard_normal((T, F, B)) +
              1j * rng.standard_normal((T, F, B)))).astype(np.complex64)

    vis = sig + static[None, :, :] + noise
    data = {"vis": vis, "time_unix": np.arange(T, dtype=float)}

    # First half (samples 0..59) is "off-source" — contains only
    # pedestal + noise. Estimate static from it.
    off = np.zeros(T, dtype=bool); off[:60] = True
    est = average_visibility(data, time_mask=off)

    # Estimate should match the injected pedestal to within ~noise/sqrt(N).
    np.testing.assert_allclose(est, static, atol=5 * 0.01 / np.sqrt(60))

    # Subtraction on the source-half samples should recover ~sig with noise.
    cleaned = subtract_static_visibility(data, est)
    on_residual = cleaned["vis"][60:] - sig[60:]
    # Residual scale ~ noise on the corrected samples + noise floor of est.
    assert np.std(on_residual) < 0.05
    # Original data is untouched.
    assert cleaned is not data
    assert np.may_share_memory(cleaned["vis"], data["vis"]) is False


def test_subtract_preserves_nan_channels():
    """Channels marked NaN in static_vis must pass through unchanged."""
    T, F, B = 5, 4, 2
    vis = np.full((T, F, B), 3.0 + 0.5j, dtype=np.complex64)
    static = np.full((F, B), 0.5 + 0.0j, dtype=np.complex64)
    static[2, :] = np.nan + 1j * np.nan   # one flagged channel

    data = {"vis": vis, "time_unix": np.arange(T, dtype=float)}
    cleaned = subtract_static_visibility(data, static)

    # Unflagged channels: cleaned = vis - static
    np.testing.assert_allclose(cleaned["vis"][:, 0, :], 3.0 + 0.5j - 0.5)
    # Flagged channel: cleaned == vis
    np.testing.assert_allclose(cleaned["vis"][:, 2, :], 3.0 + 0.5j)


def test_subtract_shape_check():
    data = {"vis": np.zeros((4, 8, 3), dtype=np.complex64),
            "time_unix": np.arange(4, dtype=float)}
    with pytest.raises(ValueError):
        subtract_static_visibility(data, np.zeros((8, 4), dtype=np.complex64))


# --------------------------------------------------------------------- #
# plot_offsource_diagnostic                                              #
# --------------------------------------------------------------------- #

def test_save_load_static_visibility_roundtrip(tmp_path):
    """Persisting then reloading must preserve the array and the metadata
    we promised to keep with it."""
    F, B = 64, 12
    rng = np.random.default_rng(7)
    static = (rng.standard_normal((F, B)) +
              1j * rng.standard_normal((F, B))).astype(np.complex64)
    freq_mhz = np.linspace(390.0, 484.0, F)
    altitudes = {
        "sun":   (-13.9, -12.7, -11.6),
        "cas-a": (6.2,    6.2,   6.2),
        "cyg-a": (-0.7,   0.0,   0.9),
        "tau-a": (15.5,  16.8,  18.2),
    }
    window = (1_700_000_000.0, 1_700_003_600.0)
    path = tmp_path / "static_2026-05-09.npz"

    save_static_visibility(
        path, static,
        freq_mhz=freq_mhz, window_unix=window,
        altitudes=altitudes, notes="layout 2026-05-09; RFI v2",
    )
    out = load_static_visibility(path)

    np.testing.assert_allclose(out["static_vis"], static)
    np.testing.assert_allclose(out["freq_mhz"], freq_mhz)
    assert out["window_unix"] == window
    assert out["altitudes"] == altitudes
    assert "layout 2026-05-09" in out["notes"]


def test_save_static_visibility_validates_shapes(tmp_path):
    path = tmp_path / "bad.npz"
    with pytest.raises(ValueError):
        save_static_visibility(
            path, np.zeros((10, 4, 2), dtype=np.complex64),
            freq_mhz=np.arange(10),
        )
    with pytest.raises(ValueError):
        save_static_visibility(
            path, np.zeros((10, 4), dtype=np.complex64),
            freq_mhz=np.arange(11),       # mismatched length
        )


def test_loaded_static_drops_straight_into_subtract(tmp_path):
    """Roundtrip should produce something subtract_static_visibility can use."""
    T, F, B = 8, 6, 4
    vis = np.full((T, F, B), 5.0 + 2.0j, dtype=np.complex64)
    static = np.full((F, B), 1.0 + 0.5j, dtype=np.complex64)
    data = {"vis": vis, "time_unix": np.arange(T, dtype=float),
            "freq_mhz": np.linspace(400, 430, F)}

    path = tmp_path / "static.npz"
    save_static_visibility(path, static, freq_mhz=data["freq_mhz"])
    loaded = load_static_visibility(path)
    cleaned = subtract_static_visibility(data, loaded["static_vis"])
    np.testing.assert_allclose(
        cleaned["vis"], vis - static[None, :, :]
    )


# --------------------------------------------------------------------- #
# build_static_visibility                                                #
# --------------------------------------------------------------------- #

def test_build_static_visibility_picks_a_window_and_calls_reader():
    """End-to-end with an injected fake reader: the function must pick
    a real quiet window on the given date, hand the chosen time range
    to the reader, average the returned visibilities, and package the
    result into the expected dict shape."""
    F, B = 32, 7
    captured = {}

    def fake_read(*, time_start, time_end, time_tz, data_root, fmt):
        # Record what the orchestrator asked for.
        captured["time_start"] = time_start
        captured["time_end"] = time_end
        captured["time_tz"] = time_tz
        captured["data_root"] = data_root
        captured["fmt"] = fmt
        # Synthesise a small visibility result for the requested window.
        T = 16
        rng = np.random.default_rng(123)
        vis = (rng.standard_normal((T, F, B)) +
               1j * rng.standard_normal((T, F, B))).astype(np.complex64)
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(time_tz)
        t0 = datetime.strptime(time_start,
                               "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        t1 = datetime.strptime(time_end,
                               "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        return {
            "vis": vis,
            "freq_mhz": np.linspace(390.0, 484.0, F),
            "time_unix": np.linspace(t0, t1, T),
        }

    result = build_static_visibility(
        date="2026-05-09", fmt="<format>", data_root="/mnt",
        verbose=False, _read_fn=fake_read,
    )

    # Shape and key contract.
    assert set(result.keys()) >= {
        "date", "data", "static_vis", "quiet_mask",
        "window_unix", "altitudes", "freq_mhz",
    }
    assert result["static_vis"].shape == (F, B)
    assert result["quiet_mask"].shape == (16,)
    assert result["quiet_mask"].all()
    assert result["date"] == "2026-05-09"

    # The reader must have been called with a window that lands inside
    # the requested calendar date (after tz conversion).
    assert captured["time_start"].startswith("2026-05-09")
    assert captured["time_tz"] == "America/Los_Angeles"
    assert captured["data_root"] == "/mnt"

    # Per-source altitudes report against the default caps.
    for name in ("sun", "tau-a", "cyg-a", "cas-a"):
        assert name in result["altitudes"]


def test_build_static_visibility_caps_long_windows():
    """A 'quiet window' that exceeds max_duration_s must be trimmed
    before the reader sees it."""
    F, B = 8, 3

    captured = {}

    def fake_read(*, time_start, time_end, time_tz, data_root, fmt):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(time_tz)
        t0 = datetime.strptime(time_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        t1 = datetime.strptime(time_end,   "%Y-%m-%d %H:%M:%S").replace(tzinfo=tz).timestamp()
        captured["duration_s"] = t1 - t0
        return {
            "vis": np.zeros((4, F, B), dtype=np.complex64),
            "freq_mhz": np.linspace(390.0, 484.0, F),
            "time_unix": np.linspace(t0, t1, 4),
        }

    build_static_visibility(
        date="2026-05-09", fmt="x", data_root="/mnt", verbose=False,
        max_duration_s=10 * 60, _read_fn=fake_read,
    )
    assert captured["duration_s"] <= 10 * 60 + 1   # allow rounding


def test_build_static_visibility_raises_when_no_window():
    """If altitude caps are impossible to satisfy, raise something useful."""
    with pytest.raises(RuntimeError, match="No quiet window"):
        build_static_visibility(
            date="2026-05-09", fmt="x",
            altitude_caps={"sun": -90.0},          # Sun never below -90°
            verbose=False, _read_fn=lambda **kw: None,
        )


def test_plot_offsource_diagnostic_runs_and_shades_window():
    """Smoke test: function returns a Figure and shades the quiet window."""
    T, F, B = 60, 32, 5
    rng = np.random.default_rng(1)
    vis = (rng.standard_normal((T, F, B)) +
           1j * rng.standard_normal((T, F, B))).astype(np.complex64)
    time_unix = np.linspace(0, 3600, T) + 1_700_000_000.0
    freq_mhz = np.linspace(400.0, 430.0, F)
    data = {"vis": vis, "time_unix": time_unix, "freq_mhz": freq_mhz}

    quiet = np.zeros(T, dtype=bool); quiet[20:35] = True
    static = average_visibility(data, time_mask=quiet)

    fig = plot_offsource_diagnostic(data, static, quiet, n_baselines=3)
    ax_t, ax_f = fig.axes[:2]
    # At least one shaded patch on the time axis (matplotlib stores
    # axvspan() patches on the axes' patches list).
    assert len(ax_t.patches) >= 1
    # Both panels populated with line plots.
    assert len(ax_t.lines) == 3
    assert len(ax_f.lines) == 3
    # Frequency axis spans the data band.
    xlim_f = ax_f.get_xlim()
    assert xlim_f[0] <= freq_mhz.min() and xlim_f[1] >= freq_mhz.max()
