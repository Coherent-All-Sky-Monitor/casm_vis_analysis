"""WGS84 ENU conversion for antenna positions."""

import numpy as np

_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = _A / np.sqrt(1.0 - _E2 * np.sin(lat) ** 2)
    x = (n + alt_m) * np.cos(lat) * np.cos(lon)
    y = (n + alt_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _E2) + alt_m) * np.sin(lat)
    return x, y, z


def geodetic_to_enu(lat_deg, lon_deg, alt_m, lat0, lon0, alt0):
    """Convert lat/lon/alt to ENU meters relative to (lat0, lon0, alt0)."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    x0, y0, z0 = geodetic_to_ecef(lat0, lon0, alt0)
    dx, dy, dz = x - x0, y - y0, z - z0
    lat0_r = np.radians(lat0)
    lon0_r = np.radians(lon0)
    sl, cl = np.sin(lat0_r), np.cos(lat0_r)
    so, co = np.sin(lon0_r), np.cos(lon0_r)
    east = -so * dx + co * dy
    north = -sl * co * dx - sl * so * dy + cl * dz
    up = cl * co * dx + cl * so * dy + sl * dz
    return east, north, up
