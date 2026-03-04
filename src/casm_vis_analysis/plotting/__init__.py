"""Plotting utilities for CASM visibility analysis."""

from datetime import datetime, timezone, timedelta


def format_time_range(time_unix):
    """Format first/last Unix timestamps as a Pacific time range string.

    Parameters
    ----------
    time_unix : ndarray, shape (T,)
        Unix timestamps.

    Returns
    -------
    str
        Single-line Pacific time range.
    """
    tz_pt = timezone(timedelta(hours=-8), name="PST")

    t0 = datetime.fromtimestamp(time_unix[0], tz=tz_pt)
    t1 = datetime.fromtimestamp(time_unix[-1], tz=tz_pt)

    fmt = "%Y-%m-%d %H:%M:%S"
    return f"PT: {t0.strftime(fmt)} – {t1.strftime(fmt)}"
