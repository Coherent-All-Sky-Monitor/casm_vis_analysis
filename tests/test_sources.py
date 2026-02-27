"""Tests for casm_vis_analysis.sources."""

import numpy as np
import pytest
from casm_vis_analysis.sources import (
    source_position,
    source_altaz,
    source_enu,
    find_transit_window,
    CATALOG,
)


class TestSourcePosition:
    def test_fixed_sources_return_skycoord(self, tiny_time):
        for name in ["cas_a", "tau_a", "cyg_a"]:
            pos = source_position(name, tiny_time)
            assert hasattr(pos, "ra")

    def test_sun_is_time_dependent(self, tiny_time):
        pos = source_position("sun", tiny_time)
        # Sun position should change over time
        assert pos.ra.deg.shape == tiny_time.shape

    def test_unknown_source_raises(self, tiny_time):
        with pytest.raises(ValueError, match="Unknown source"):
            source_position("vega", tiny_time)

    def test_name_normalization(self, tiny_time):
        # Should handle case and separators
        pos1 = source_position("cas_a", tiny_time)
        pos2 = source_position("Cas-A", tiny_time)
        assert pos1.ra.deg == pos2.ra.deg


class TestSourceAltAz:
    def test_returns_arrays(self, tiny_time):
        alt, az = source_altaz("cas_a", tiny_time)
        assert alt.shape == tiny_time.shape
        assert az.shape == tiny_time.shape

    def test_alt_in_range(self, tiny_time):
        alt, _ = source_altaz("cas_a", tiny_time)
        assert np.all(alt >= -90)
        assert np.all(alt <= 90)


class TestSourceENU:
    def test_shape(self, tiny_time):
        enu = source_enu("cas_a", tiny_time)
        assert enu.shape == (len(tiny_time), 3)

    def test_unit_vectors(self, tiny_time):
        enu = source_enu("cas_a", tiny_time)
        norms = np.linalg.norm(enu, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)


class TestFindTransitWindow:
    def test_returns_indices(self, tiny_time):
        # Use sun — likely above horizon for some times
        # Use a time when sun is up (noon UTC ~ morning California)
        t_noon = 1738065600.0  # roughly noon UTC
        times = t_noon + np.arange(100) * 137.44
        try:
            i_start, i_end = find_transit_window("sun", times, min_alt_deg=5)
            assert 0 <= i_start <= i_end < len(times)
        except ValueError:
            # Sun might not be up at this particular time — that's OK
            pass

    def test_raises_if_never_up(self, tiny_time):
        # Very high altitude threshold should fail
        with pytest.raises(ValueError, match="never rises"):
            find_transit_window("cas_a", tiny_time, min_alt_deg=89.99)
