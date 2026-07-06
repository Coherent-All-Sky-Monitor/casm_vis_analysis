# Sources and Transit Windows

`casm_vis_analysis.sources` provides source positions, AltAz transforms, ENU direction vectors, and transit window detection for the OVRO site.

## Catalog

Fixed J2000 entries. The Sun is computed from ephemeris at each call.

| Key | Source | RA | Dec |
|---|---|---|---|
| `sun` | Sun | (ephemeris) | (ephemeris) |
| `cas_a` | Cas A | 23h23m24s | +58d48m54s |
| `cyg_a` | Cyg A | 19h59m28.36s | +40d44m02.1s |
| `tau_a` | Tau A | 05h34m31.94s | +22d00m52.2s |
| `vir_a` | Vir A | 12h30m49.42s | +12d23m28.0s |
| `b0329_54` | B0329+54 | 03h32m59.41s | +54d34m43.33s |

Name resolution is case-insensitive and normalises hyphens, spaces, and `+` to underscores, so `"cas-a"`, `"cas_a"`, and `"CAS A"` all resolve correctly.

## API

```python
from casm_vis_analysis.sources import (
    source_position,
    source_altaz,
    source_enu,
    find_transit_window,
    source_flux,
)
```

### `source_position(name, time_unix)`

Returns an Astropy `SkyCoord`. For the Sun, position is time-dependent; for catalog sources, returns the fixed J2000 coordinate.

```
name       : str — source name
time_unix  : float or array-like — Unix timestamps
Returns    : SkyCoord
```

### `source_altaz(name, time_unix)`

```
Returns    : (alt_deg, az_deg) — both ndarray, degrees
             Azimuth convention: N=0, E=90
```

### `source_enu(name, time_unix)`

ENU unit direction vector toward the source at each time sample.

```
Returns    : ndarray, shape (T, 3)
             Columns: East, North, Up
```

This is the input to `geometric_delay`. Call it once per source and reuse across baselines.

### `find_transit_window(name, time_unix, min_alt_deg=10.0)`

Returns the first and last index where the source is at or above `min_alt_deg`.

```
Returns    : (i_start, i_end) — inclusive integer indices
Raises     : ValueError if source never rises above the threshold
```

`fringe_stop()` calls this internally to populate `fs["time_mask"]`. You rarely need to call it directly unless you want to trim data before passing it in.

### `source_flux(name, freqs_mhz, sun_flux_400=100_000.0)`

Power-law flux model: `S(f) = S_400 * (f / 400 MHz)^alpha`.

| Source | S_400 (Jy) | alpha |
|---|---|---|
| sun | 100,000 (adjustable) | -0.1 (Cane et al., locked) |
| cas_a | 3400 | -0.77 |
| cyg_a | 4800 | -0.80 |
| tau_a | 1200 | -0.27 |
| vir_a | 280 | -0.86 |

The Sun's spectral index is locked at -0.1 (Cane et al. quiet-sun). Adjust `sun_flux_400` for active-Sun estimates.

## Site constants

OVRO site coordinates used by all coordinate transforms:

```python
from casm_io.constants import OVRO_LAT_DEG, OVRO_LON_DEG, OVRO_ELEV_M
# 37.2339 deg, -118.2821 deg, 1222.0 m
```

## Notes

- Cas A is circumpolar at OVRO (minimum altitude approximately 6°). It never fully sets, so `find_transit_window` always returns a range when called with `min_alt_deg <= 6`. This is relevant for `find_quiet_windows` in `offsource.py`, which uses an altitude cap rather than a below-horizon test.
- Call `source_altaz` on the full `data["time_unix"]` array rather than resampling; the function is vectorised and the cost is negligible compared to a disk read.
