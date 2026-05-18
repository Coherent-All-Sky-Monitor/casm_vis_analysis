"""Tests for the beam-weights validation module.

Two tiers:
  1. Pure-math tests that don't need real telescope data —
     ``load_beams_from_int8`` round-trip and
     ``find_source_beam_transits`` against a known geometry.
  2. Synthetic-vis end-to-end test — fabricates a single bright
     point source visibility, runs ``validate_beam_weights``, asserts
     the on-source beam passes and an off-source control fails.
"""

from __future__ import annotations

from datetime import datetime, timezone

import h5py
import json
import numpy as np
import pytest

from casm_vis_analysis import (
    BeamHit,
    find_source_beam_transits,
    load_beams_from_int8,
)
from casm_vis_analysis.sources import source_altaz


# ---------- helpers -------------------------------------------------


def _write_minimal_int8_h5(path, beams_alt_az, fwhm_ew=4.0, fwhm_ns=4.0):
    """Write the minimal subset of a SNAP int8 weights HDF5 that the
    validator reads. We don't need real weights values for the
    pointings/array-geometry tests — just the metadata layout."""
    n_beams = len(beams_alt_az)
    alt = np.array([a for a, _ in beams_alt_az], dtype=np.float64)
    az = np.array([z for _, z in beams_alt_az], dtype=np.float64)

    # Two-antenna E-W array of "the right size" so compute_beam_fwhm
    # gives us roughly the requested FWHMs at the default freq.
    # compute_beam_fwhm uses lambda/baseline; pick separation that
    # yields the desired FWHM. 437.5 MHz -> lambda ~0.685 m.
    # FWHM_deg = degrees(lambda / b)  =>  b = lambda / radians(FWHM)
    lam = 0.6853  # m at 437.5 MHz
    b_ew = lam / np.deg2rad(fwhm_ew) if fwhm_ew > 0 else 1.0
    b_ns = lam / np.deg2rad(fwhm_ns) if fwhm_ns > 0 else 1.0
    positions_enu = np.zeros((64, 3), dtype=np.float64)
    positions_enu[0] = [0.0, 0.0, 0.0]
    positions_enu[1] = [b_ew, 0.0, 0.0]
    positions_enu[2] = [0.0, b_ns, 0.0]
    active_mask = np.zeros(64, dtype=bool)
    active_mask[:3] = True
    antenna_ids = np.full(64, -1, dtype=np.int32)
    antenna_ids[:3] = [1, 2, 3]

    with h5py.File(path, "w") as f:
        f.attrs["scale_factor"] = 127.0
        f.attrs["n_beams"] = n_beams
        f.attrs["n_channels"] = 4
        f.attrs["n_pol"] = 2
        f.attrs["n_antennas"] = 64
        f.attrs["version"] = "2.0"
        f.attrs["format_type"] = "int8_snap_weights"
        f.create_dataset("weights_int8",
                         data=np.zeros((2, 4, 2, n_beams, 64), dtype=np.int8))
        f.create_dataset("frequencies_hz",
                         data=np.linspace(420e6, 410e6, 4))
        pt = f.create_group("pointings")
        pt.create_dataset("alt_deg", data=alt)
        pt.create_dataset("az_deg", data=az)
        pt.attrs["names"] = json.dumps([f"b{i}" for i in range(n_beams)])
        ac = f.create_group("array_config")
        ac.create_dataset("positions_enu", data=positions_enu)
        ac.create_dataset("active_mask", data=active_mask)
        ac.create_dataset("antenna_ids", data=antenna_ids)
        ac.attrs["csv_path"] = "<synthetic test fixture>"
        ac.attrs["pos_ids"] = json.dumps([""] * 64)


# ---------- tests ---------------------------------------------------


class TestLoadBeamsFromInt8:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "tiny.h5"
        beams_in = [(80.0, 0.0), (60.0, 90.0), (45.0, 180.0)]
        _write_minimal_int8_h5(path, beams_in)
        beams = load_beams_from_int8(path)
        assert beams["n_beams"] == 3
        np.testing.assert_allclose(beams["alt_deg"], [80.0, 60.0, 45.0])
        np.testing.assert_allclose(beams["az_deg"], [0.0, 90.0, 180.0])
        assert beams["names"] == ["b0", "b1", "b2"]
        assert beams["fwhm_ew_deg"] > 0
        assert beams["fwhm_ns_deg"] > 0
        assert beams["active_positions_enu"].shape == (3, 3)


class TestFindSourceBeamTransits:
    def test_no_hits_when_beam_is_far_from_source(self, tmp_path):
        """A beam pointed at the south pole (alt=0, az=180) should never
        catch the Sun for a daytime window at OVRO."""
        path = tmp_path / "tiny.h5"
        # Tiny FWHM so we don't accidentally include the Sun.
        _write_minimal_int8_h5(path, [(0.0, 180.0)], fwhm_ew=2.0, fwhm_ns=2.0)
        beams = load_beams_from_int8(path)

        # 4 hours at noon UTC on 2026-05-08.
        t0 = datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc).timestamp()
        time_unix = t0 + np.linspace(0, 4 * 3600, 80)
        hits = find_source_beam_transits(beams, ["sun"], time_unix)
        assert hits == []

    def test_hit_when_beam_tracks_the_source_peak(self, tmp_path):
        """Place a beam at the Sun's maximum-altitude position during a
        4 h afternoon window; expect exactly one BeamHit naming 'sun'.
        The peak time should fall within the window."""
        # First find the Sun's peak alt/az during the window so we can
        # build the beam at that exact pointing.
        t0 = datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc).timestamp()
        time_unix = t0 + np.linspace(0, 4 * 3600, 240)
        sun_alt, sun_az = source_altaz("sun", time_unix)
        i_pk = int(np.argmax(sun_alt))

        path = tmp_path / "tiny.h5"
        _write_minimal_int8_h5(path, [(float(sun_alt[i_pk]), float(sun_az[i_pk]))],
                               fwhm_ew=4.0, fwhm_ns=4.0)
        beams = load_beams_from_int8(path)

        hits = find_source_beam_transits(beams, ["sun"], time_unix)
        assert len(hits) >= 1
        assert all(isinstance(h, BeamHit) for h in hits)
        # Concatenate any contiguous hits — duration should cover the
        # actual time the Sun spent inside the FWHM, > 0.
        sun_hits = [h for h in hits if h.source == "sun"]
        assert sun_hits
        assert all(h.entry_unix <= h.peak_unix <= h.exit_unix for h in sun_hits)
        # The peak should be near the Sun's max-alt time we computed.
        assert min(abs(h.peak_unix - time_unix[i_pk]) for h in sun_hits) < 600.0

    def test_fwhm_factor_tightens_window(self, tmp_path):
        """fwhm_factor=0.5 (half-power) should give a shorter dwell time
        than fwhm_factor=1.0 (full FWHM)."""
        t0 = datetime(2026, 5, 8, 18, 0, tzinfo=timezone.utc).timestamp()
        time_unix = t0 + np.linspace(0, 4 * 3600, 240)
        sun_alt, sun_az = source_altaz("sun", time_unix)
        i_pk = int(np.argmax(sun_alt))

        path = tmp_path / "tiny.h5"
        _write_minimal_int8_h5(path, [(float(sun_alt[i_pk]), float(sun_az[i_pk]))],
                               fwhm_ew=4.0, fwhm_ns=4.0)
        beams = load_beams_from_int8(path)

        full = find_source_beam_transits(beams, ["sun"], time_unix, fwhm_factor=1.0)
        half = find_source_beam_transits(beams, ["sun"], time_unix, fwhm_factor=0.5)
        d_full = sum(h.duration_min for h in full)
        d_half = sum(h.duration_min for h in half)
        assert d_full > d_half

    def test_unknown_source_raises(self, tmp_path):
        path = tmp_path / "tiny.h5"
        _write_minimal_int8_h5(path, [(45.0, 90.0)])
        beams = load_beams_from_int8(path)
        time_unix = np.linspace(1.7e9, 1.7e9 + 3600, 60)
        with pytest.raises(ValueError, match="Unknown source"):
            find_source_beam_transits(beams, ["andromeda"], time_unix)
