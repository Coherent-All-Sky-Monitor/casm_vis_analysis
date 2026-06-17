"""Plotting utilities for CASM visibility analysis."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def format_time_range(time_unix, time_tz="UTC"):
    """Format first/last Unix timestamps as a local time range string.

    Parameters
    ----------
    time_unix : ndarray, shape (T,)
        Unix timestamps.
    time_tz : str
        IANA timezone name (e.g. ``"America/Los_Angeles"``) or ``"UTC"``.
        DST-aware; the label uses the actual abbreviation in effect
        (e.g. ``PDT`` in summer, ``PST`` in winter, ``UTC``).

    Returns
    -------
    str
        Single-line local time range, prefixed with the timezone
        abbreviation, e.g. ``"PDT: 2026-06-17 11:00:00 – 2026-06-17 11:20:00"``.
    """
    tz = ZoneInfo(time_tz) if time_tz != "UTC" else timezone.utc

    t0 = datetime.fromtimestamp(time_unix[0], tz=tz)
    t1 = datetime.fromtimestamp(time_unix[-1], tz=tz)

    fmt = "%Y-%m-%d %H:%M:%S"
    label = t0.strftime("%Z") or "UTC"  # %Z can be empty for bare offsets
    return f"{label}: {t0.strftime(fmt)} – {t1.strftime(fmt)}"
