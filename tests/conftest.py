"""Synthetic fixtures for casm_vis_analysis tests."""

import numpy as np
import pytest


@pytest.fixture
def n_ant():
    """Number of antennas in synthetic data."""
    return 4


@pytest.fixture
def n_time():
    return 20


@pytest.fixture
def n_freq():
    return 64


@pytest.fixture
def tiny_freq(n_freq):
    """Descending frequency axis in MHz (native order)."""
    return np.linspace(468.75, 375.0, n_freq)


@pytest.fixture
def tiny_time(n_time):
    """Unix timestamps — 1 hour around a transit-like event."""
    t0 = 1738000000.0  # arbitrary epoch
    return t0 + np.arange(n_time) * 137.44


@pytest.fixture
def tiny_vis(n_time, n_freq, n_ant):
    """Synthetic cross-correlation vis for ref vs n_ant-1 targets.

    Shape: (T, F, n_targets) complex64 with unit amplitude and random phase.
    """
    n_targets = n_ant - 1
    rng = np.random.default_rng(42)
    phase = rng.uniform(-np.pi, np.pi, (n_time, n_freq, n_targets))
    return np.exp(1j * phase).astype(np.complex64)


@pytest.fixture
def tiny_autocorr(n_time, n_freq, n_ant):
    """Synthetic autocorrelation vis (real, positive).

    Shape: (T, F, n_ant) float32.
    """
    rng = np.random.default_rng(123)
    return (100 + 10 * rng.standard_normal((n_time, n_freq, n_ant))).astype(np.float32)


@pytest.fixture
def tiny_positions(n_ant):
    """ENU antenna positions in meters, shape (n_ant, 3)."""
    rng = np.random.default_rng(7)
    pos = np.zeros((n_ant, 3))
    pos[:, 0] = rng.uniform(-50, 50, n_ant)  # East
    pos[:, 1] = rng.uniform(-5, 5, n_ant)    # North (small — E-W array)
    pos[:, 2] = 0.0                           # Up
    return pos


@pytest.fixture
def tiny_baseline_enu(tiny_positions):
    """Baseline vectors from ant 0 to ants 1..N, shape (n_targets, 3)."""
    return tiny_positions[1:] - tiny_positions[0]
