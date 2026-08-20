"""Three-panel solar-transit waterfall plots from beamformer .fil files.

Produces, for one SIGPROC filterbank written by ``casm-beamdump-to-fil``, a
single PNG with a channel-mean-normalized dynamic spectrum, the raw bandpass,
and a few selected-channel light curves on a local-time axis. This is the
standard look for the solar campaign transit checks (on-Sun beams, nulls,
incoherent beam).

nbeams bug workaround
---------------------
``casm-beamdump-to-fil`` historically stamps SIGPROC ``nbeams`` with the
number of beams in the *dump block* rather than 1, even though each file
holds exactly one beam. Any ``nsamples`` derived from that header is wrong by
the same factor and would mis-stride the payload. We therefore never trust
``nbeams``/``nsamples`` and recompute the sample count from the file size:
``nsamp = (filesize - header_size) // (nchans * nbits // 8)``.

Memory model
------------
These files run to ~160 GB, so nothing is ever read whole. The payload is
memory-mapped as float32 ``(nsamp, nchans)`` and block-averaged in time in
chunks of roughly 2 GB; only the downsampled ``(nt, nchans)`` array (a few
MB at the default ~1 s binning) is held in memory.
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_TFAC = 954          # ~1.0 s bins at the 1.0486 ms sample time
DEFAULT_CHANS = (505, 1200, 1500, 2000)
DEFAULT_TZ = "America/Los_Angeles"
DEFAULT_CMAP = "inferno"
MJD_UNIX_EPOCH = 40587.0    # MJD of 1970-01-01
READ_BLOCK_BYTES = 2e9

# SIGPROC header keyword types; anything not listed is skipped as unknown.
_HDR_DOUBLE = {"fch1", "foff", "tsamp", "tstart", "az_start", "za_start",
               "src_raj", "src_dej", "period", "refdm", "fchannel"}
_HDR_INT = {"machine_id", "telescope_id", "data_type", "nchans", "nbits",
            "nifs", "nbeams", "ibeam", "nsamples", "barycentric",
            "pulsarcentric", "signed", "nsamp"}
_HDR_STR = {"rawdatafile", "source_name"}


def _read_string(f):
    """Read a SIGPROC length-prefixed keyword/value string."""
    raw = f.read(4)
    if len(raw) < 4:
        raise ValueError("truncated SIGPROC header")
    n = struct.unpack("<i", raw)[0]
    if not 0 <= n < 128:
        raise ValueError(f"implausible SIGPROC string length {n}")
    return f.read(n).decode("ascii", "replace")


def read_filterbank_header(path):
    """Parse a SIGPROC header inline. Returns dict plus '_header_size'."""
    header = {}
    with open(path, "rb") as f:
        if _read_string(f) != "HEADER_START":
            raise ValueError(f"{path}: not a SIGPROC filterbank")
        while True:
            key = _read_string(f)
            if key == "HEADER_END":
                break
            if key in _HDR_DOUBLE:
                header[key] = struct.unpack("<d", f.read(8))[0]
            elif key in _HDR_INT:
                header[key] = struct.unpack("<i", f.read(4))[0]
            elif key in _HDR_STR:
                header[key] = _read_string(f)
            else:
                raise ValueError(f"{path}: unknown header keyword {key!r}")
        header["_header_size"] = f.tell()
    return header


def open_fil(path):
    """Return (memmap[nsamp, nchans], header) with the true sample count."""
    header = read_filterbank_header(path)
    nchans = int(header["nchans"])
    nbits = int(header.get("nbits", 32))
    if nbits != 32:
        raise ValueError(f"{path}: expected nbits=32, got {nbits}")
    hdr_size = int(header["_header_size"])
    # nbeams/nsamples in the header are unreliable (see module docstring).
    nsamp = (os.path.getsize(path) - hdr_size) // (nchans * nbits // 8)
    data = np.memmap(path, dtype=np.float32, mode="r", offset=hdr_size,
                     shape=(nsamp, nchans))
    header["_nsamp"] = nsamp
    return data, header


def freq_axis(header):
    """Channel centre frequencies in MHz."""
    return (float(header["fch1"])
            + np.arange(int(header["nchans"])) * float(header["foff"]))


def downsample(path, tfac):
    """Block-average in time. Returns (spec[nt, nchans], tsec, freqs, header)."""
    data, header = open_fil(path)
    nsamp, nchans = data.shape
    nt = nsamp // tfac
    if nt < 1:
        raise ValueError(f"{path}: only {nsamp} samples, too few for --tfac {tfac}")
    out = np.empty((nt, nchans), dtype=np.float64)
    blk = max(1, int(READ_BLOCK_BYTES / (nchans * 4 * tfac)))
    for i0 in range(0, nt, blk):
        i1 = min(nt, i0 + blk)
        chunk = np.asarray(data[i0 * tfac:i1 * tfac], dtype=np.float32)
        out[i0:i1] = chunk.reshape(i1 - i0, tfac, nchans).mean(axis=1)
    tsamp = float(header["tsamp"])
    tsec = (np.arange(nt) + 0.5) * tfac * tsamp
    del data
    return out, tsec, freq_axis(header), header


def local_times(header, tsec, tz):
    """Sample times as timezone-aware datetimes in `tz`."""
    t0 = datetime.fromtimestamp((float(header["tstart"]) - MJD_UNIX_EPOCH) * 86400.0,
                                tz=timezone.utc)
    return [(t0 + timedelta(seconds=float(s))).astimezone(tz) for s in tsec]


def infer_beam(path):
    """Beam name from the filename: 'IB', or the number in _bNNN / _nNNN."""
    stem = os.path.basename(path)
    if stem.endswith(".fil"):
        stem = stem[:-4]
    if "IB" in stem:
        return "IB"
    # 'n' names the null beams from the 2026-08-17 run; 'b' the rest.
    m = re.search(r"_[bn](\d+)", stem)
    if m:
        return m.group(1)
    raise ValueError(
        f"cannot infer beam from filename {os.path.basename(path)!r}; "
        "pass --beam IB or --beam NNN"
    )


def beam_description(beam, role=None):
    """'Incoherent Beam' / 'Coherent Beam 336', with ', role' appended."""
    if str(beam).upper() == "IB":
        desc = "Incoherent Beam"
    else:
        desc = f"Coherent Beam {int(beam)}"
    if role:
        desc = f"{desc}, {role}"
    return desc


def plot_waterfall(path, out_path, beam, role=None, tfac=DEFAULT_TFAC,
                   chans=DEFAULT_CHANS, tz=DEFAULT_TZ, cmap=DEFAULT_CMAP):
    """Write the 3-panel waterfall PNG for one filterbank. Returns out_path."""
    zone = ZoneInfo(tz)
    spec, tsec, freqs, header = downsample(path, tfac)
    bandpass = spec.mean(axis=0)
    data_norm = spec / bandpass[None, :]
    times_local = local_times(header, tsec, zone)
    times_num = mdates.date2num(times_local)

    bad = [i for i in chans if not 0 <= i < freqs.size]
    if bad:
        raise ValueError(f"channel indices {bad} outside 0..{freqs.size - 1}")

    date_label = times_local[0].strftime("%B %-d %Y")
    title = (f"CASM Dynamic Spectrum: {beam_description(beam, role)} "
             f"({date_label})")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False,
                             gridspec_kw={"height_ratios": [3, 1, 1.8]})
    ax = axes[0]
    ax.imshow(data_norm.T, aspect="auto", origin="upper",
              extent=[times_num[0], times_num[-1], freqs[-1], freqs[0]],
              cmap=cmap,
              vmin=np.percentile(data_norm, 5),
              vmax=np.percentile(data_norm, 95))
    ax.set_ylabel("Frequency (MHz)")
    ax.set_title(title, fontweight="bold")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=zone))

    ax = axes[1]
    ax.plot(freqs, bandpass)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power")
    ax.set_yscale("log")

    ax = axes[2]
    for i in chans:
        ax.plot(times_local, data_norm[:, i], lw=0.8,
                label=f"ch {i} ({freqs[i]:.1f} MHz)")
    ax.plot(times_local, data_norm.mean(axis=1), color="k", lw=2, label="Mean")
    ax.set_ylabel("Normalized Power")
    ax.set_xlabel("Local Time (OVRO)")
    # Legend above the panel so it never covers the light curves.
    ax.legend(ncol=5, fontsize=9, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=zone))

    plt.setp(axes[0].get_xticklabels(), rotation=30, ha="right")
    plt.setp(axes[2].get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _parse_chans(arg):
    """'505,1200' -> [505, 1200]."""
    try:
        return [int(p) for p in arg.split(",") if p.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"--chans expects comma-separated ints (got {arg!r})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Three-panel solar-transit waterfall from a beamformer .fil")
    parser.add_argument("fil", help="Input SIGPROC filterbank (one beam)")
    parser.add_argument("--out-dir", default=".",
                        help="Directory for the output PNG (default: .)")
    parser.add_argument("--beam", default=None,
                        help="'IB' or beam number. Default: infer from filename")
    parser.add_argument("--role", default=None,
                        help="Role text appended to the title, e.g. 'Null'")
    parser.add_argument("--tfac", type=int, default=DEFAULT_TFAC,
                        help=f"Time samples averaged per bin (default: {DEFAULT_TFAC}, ~1 s)")
    parser.add_argument("--chans", type=_parse_chans, default=list(DEFAULT_CHANS),
                        metavar="I,J,K,L",
                        help="Channel indices for the light-curve panel "
                             f"(default: {','.join(str(c) for c in DEFAULT_CHANS)})")
    parser.add_argument("--tz", default=DEFAULT_TZ,
                        help=f"Timezone for the time axis (default: {DEFAULT_TZ})")
    parser.add_argument("--cmap", default=DEFAULT_CMAP,
                        help=f"Matplotlib colormap for the waterfall panel "
                             f"(default: {DEFAULT_CMAP}; try viridis, magma, cividis)")
    parser.add_argument("--out-name", default=None,
                        help="Output PNG basename (default: <stem>_waterfall_localtime.png)")
    args = parser.parse_args(argv)

    stem = os.path.basename(args.fil)
    if stem.endswith(".fil"):
        stem = stem[:-4]
    beam = args.beam if args.beam is not None else infer_beam(args.fil)
    out_name = args.out_name or f"{stem}_waterfall_localtime.png"
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, out_name)

    plot_waterfall(args.fil, out_path, beam, role=args.role, tfac=args.tfac,
                   chans=args.chans, tz=args.tz, cmap=args.cmap)
    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
