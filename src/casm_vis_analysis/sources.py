"""Source catalog and coordinate utilities for CASM visibility analysis.

Provides source positions (J2000), AltAz transforms, ENU direction vectors,
and transit window detection for the OVRO site.
"""

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

from casm_io.constants import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ELEV_M

# Fixed J2000 source catalog
CATALOG = {
    "cas_a": SkyCoord("23h23m24s", "+58d48m54s", frame="icrs"),
    "tau_a": SkyCoord("05h34m31.94s", "+22d00m52.2s", frame="icrs"),
    "cyg_a": SkyCoord("19h59m28.36s", "+40d44m02.1s", frame="icrs"),
}

OVRO_LOCATION = EarthLocation(
    lat=OVRO_LAT_DEG * u.deg,
    lon=OVRO_LON_DEG * u.deg,
    height=OVRO_ELEV_M * u.m,
)


def source_position(name, time_unix):
    """Get source SkyCoord at given times.

    Parameters
    ----------
    name : str
        Source name: 'sun', 'cas_a', 'tau_a', 'cyg_a'.
    time_unix : float or array-like
        Unix timestamps.

    Returns
    -------
    SkyCoord
        ICRS position(s). For Sun, this is time-dependent.
    """
    t = Time(np.atleast_1d(time_unix), format="unix")
    if name.lower() == "sun":
        return get_sun(t)
    key = name.lower().replace("-", "_").replace(" ", "_")
    if key not in CATALOG:
        raise ValueError(f"Unknown source '{name}'. Available: sun, {', '.join(CATALOG)}")
    return CATALOG[key]


def source_altaz(name, time_unix):
    """Compute source altitude and azimuth at OVRO.

    Parameters
    ----------
    name : str
        Source name.
    time_unix : array-like
        Unix timestamps.

    Returns
    -------
    alt_deg : ndarray
        Altitude in degrees.
    az_deg : ndarray
        Azimuth in degrees (N=0, E=90).
    """
    time_unix = np.atleast_1d(time_unix)
    t = Time(time_unix, format="unix")
    altaz_frame = AltAz(obstime=t, location=OVRO_LOCATION)
    pos = source_position(name, time_unix)
    aa = pos.transform_to(altaz_frame)
    return aa.alt.deg, aa.az.deg


def source_enu(name, time_unix):
    """Compute ENU unit direction vector toward source.

    Parameters
    ----------
    name : str
        Source name.
    time_unix : array-like
        Unix timestamps.

    Returns
    -------
    enu : ndarray, shape (T, 3)
        East, North, Up unit vectors.
    """
    alt_deg, az_deg = source_altaz(name, time_unix)
    alt_rad = np.deg2rad(alt_deg)
    az_rad = np.deg2rad(az_deg)
    e = np.sin(az_rad) * np.cos(alt_rad)
    n = np.cos(az_rad) * np.cos(alt_rad)
    up = np.sin(alt_rad)
    return np.column_stack([e, n, up])


def find_transit_window(name, time_unix, min_alt_deg=10.0):
    """Find time indices where source is above minimum altitude.

    Parameters
    ----------
    name : str
        Source name.
    time_unix : array-like
        Unix timestamps.
    min_alt_deg : float
        Minimum altitude threshold in degrees.

    Returns
    -------
    i_start, i_end : int
        Start and end indices (inclusive) of the transit window.
        Returns (0, len-1) if source is always above threshold.

    Raises
    ------
    ValueError
        If source never rises above min_alt_deg.
    """
    time_unix = np.atleast_1d(time_unix)
    alt_deg, _ = source_altaz(name, time_unix)
    above = alt_deg >= min_alt_deg
    if not np.any(above):
        raise ValueError(
            f"Source '{name}' never rises above {min_alt_deg} deg "
            f"(max alt: {alt_deg.max():.1f} deg)"
        )
    indices = np.where(above)[0]
    return int(indices[0]), int(indices[-1])
