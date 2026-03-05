"""CLI entry points for CASM visibility analysis.

Four commands:
- casm-data-span: survey data directory and list observations
- casm-autocorr: autocorrelation power spectra
- casm-waterfall: upper-triangle waterfall matrix
- casm-fringe-stop: fringe-stop + optional delay correction + diagnostics
"""

import argparse
import sys

import numpy as np


def _common_parser(description):
    """Build parser with shared arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", required=True, help="Directory with .dat files")
    parser.add_argument("--obs", required=True, help="Observation ID (UTC timestamp)")
    parser.add_argument("--format", required=True,
                        help="Format name or path to JSON")
    parser.add_argument("--layout", required=True,
                        help="Path to antenna layout CSV")
    parser.add_argument("--output-dir", default="./output",
                        help="Base output directory (default: ./output)")
    parser.add_argument("--freq-order", default="descending",
                        choices=["ascending", "descending"],
                        help="Frequency axis order (default: descending)")
    parser.add_argument("--time-start", default=None,
                        help="Start time for data selection")
    parser.add_argument("--time-end", default=None,
                        help="End time for data selection")
    parser.add_argument("--time-tz", default="UTC",
                        help="Timezone for --time-start/--time-end (default: UTC)")
    parser.add_argument("--nfiles", type=int, default=None,
                        help="Number of files to read")
    parser.add_argument("--skip-nfiles", type=int, default=0,
                        help="Number of files to skip before reading (requires --nfiles)")
    parser.add_argument("--show", action="store_true",
                        help="Show plots inline (e.g. Jupyter) instead of saving to disk")
    return parser


def autocorr_main(argv=None):
    """Autocorrelation power spectrum plots."""
    from casm_io.correlator import load_format, VisibilityReader, AntennaMapping
    from casm_io.correlator.baselines import triu_flat_index
    from casm_vis_analysis.plotting.autocorr import plot_autocorr
    from casm_vis_analysis.output import make_output_dir

    parser = _common_parser("Plot autocorrelation power spectra")
    parser.add_argument("--ncols", type=int, default=4,
                        help="Columns in plot grid (default: 4)")
    parser.add_argument("--scale", default="dB", choices=["dB", "linear"],
                        help="Power axis scale (default: dB)")
    args = parser.parse_args(argv)

    fmt = load_format(args.format)
    ant = AntennaMapping.load(args.layout)
    reader = VisibilityReader(args.data_dir, args.obs, fmt)
    data = reader.read(
        freq_order=args.freq_order,
        time_start=args.time_start, time_end=args.time_end, time_tz=args.time_tz,
        nfiles=args.nfiles, skip_nfiles=args.skip_nfiles,
    )

    vis = data["vis"]
    freq_mhz = data["freq_mhz"]
    time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}, freq: {freq_mhz.shape}")

    # Extract autocorrelations
    active = ant.active_antennas()
    auto_indices = []
    labels = []
    for aid in active:
        pidx = ant.packet_index(aid)
        auto_indices.append(triu_flat_index(fmt.nsig, pidx, pidx))
        labels.append(ant.format_antenna(aid))

    auto_vis = vis[:, :, auto_indices]  # (T, F, n_ant)
    print(f"Autocorr shape: {auto_vis.shape}")

    if not args.show:
        dirs = make_output_dir(args.output_dir, args.obs)

    # Group by SNAP and plot
    snap_groups = {}
    for i, aid in enumerate(active):
        snap_id, _ = ant.snap_adc(aid)
        snap_groups.setdefault(snap_id, ([], []))
        snap_groups[snap_id][0].append(i)
        snap_groups[snap_id][1].append(labels[i])

    for snap_id, (indices, snap_labels) in sorted(snap_groups.items()):
        snap_vis = auto_vis[:, :, indices]
        path = None if args.show else dirs["autocorr"] / f"autocorr_snap{snap_id}.png"
        plot_autocorr(snap_vis, freq_mhz, snap_labels,
                      output_path=path, ncols=args.ncols,
                      time_unix=time_unix, snap_label=f"SNAP {snap_id}",
                      scale=args.scale)
        if not args.show:
            print(f"Saved: {path}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


def waterfall_main(argv=None):
    """Waterfall matrix plot."""
    from casm_io.correlator import load_format, VisibilityReader, AntennaMapping
    from casm_vis_analysis.plotting.waterfall import plot_waterfall
    from casm_vis_analysis.output import make_output_dir

    parser = _common_parser("Plot upper-triangle waterfall matrix")
    parser.add_argument("--split-max", type=int, default=16,
                        help="Max antennas per figure (default: 16)")
    parser.add_argument("--diag-spectra", action="store_true",
                        help="Show 1D power spectra on diagonal instead of 2D autocorrelation")
    parser.add_argument("--pub", action="store_true",
                        help="Publication quality output (300 DPI, PDF)")
    args = parser.parse_args(argv)

    fmt = load_format(args.format)
    ant = AntennaMapping.load(args.layout)
    reader = VisibilityReader(args.data_dir, args.obs, fmt)
    data = reader.read(
        freq_order=args.freq_order,
        time_start=args.time_start, time_end=args.time_end, time_tz=args.time_tz,
        nfiles=args.nfiles, skip_nfiles=args.skip_nfiles,
    )

    vis = data["vis"]
    freq_mhz = data["freq_mhz"]
    time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}")

    # Build per-antenna arrays for active antennas only
    active = ant.active_antennas()
    packet_indices = [ant.packet_index(aid) for aid in active]
    antenna_labels = [ant.format_antenna(aid) for aid in active]
    snap_adc_labels = [f"S{s}A{a}" for s, a in (ant.snap_adc(aid) for aid in active)]

    if not args.show:
        dirs = make_output_dir(args.output_dir, args.obs)

    plot_waterfall(vis, freq_mhz, time_unix, fmt.nsig,
                   packet_indices=packet_indices,
                   antenna_labels=antenna_labels,
                   snap_adc_labels=snap_adc_labels,
                   split_max=args.split_max,
                   output_dir=None if args.show else dirs["waterfall"],
                   diag_spectra=args.diag_spectra,
                   pub=args.pub)

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"Saved waterfalls to: {dirs['waterfall']}")


def fringe_stop_main(argv=None):
    """Fringe-stop, optional delay correction, and diagnostic plots."""
    from casm_io.correlator import load_format, VisibilityReader, AntennaMapping
    from casm_vis_analysis.sources import source_enu, find_transit_window
    from casm_vis_analysis.fringe_stop import (
        compute_baselines_enu, geometric_delay, fringe_stop,
    )
    from casm_vis_analysis.delay import fit_delay, apply_delay
    from casm_vis_analysis.plotting.fringe_diag import plot_fringe_diagnostic
    from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq
    from casm_vis_analysis.output import make_output_dir, save_results

    parser = _common_parser("Fringe-stop visibilities with diagnostics")
    parser.add_argument("--ref-ant", type=int, required=True,
                        help="Reference antenna ID")
    parser.add_argument("--source", default="sun",
                        help="Source name (default: sun)")
    parser.add_argument("--sign", type=int, default=-1,
                        help="Fringe-stop sign convention (default: -1)")
    parser.add_argument("--min-alt", type=float, default=10.0,
                        help="Minimum source altitude in degrees (default: 10)")
    parser.add_argument("--save-npz", action="store_true",
                        help="Save results to NPZ")
    parser.add_argument("--rfi-mask", default=None,
                        help="Path to RFI mask NPZ (bool array, key='mask')")
    parser.add_argument("--delay-model", nargs="*", default=None,
                        help="Delay models to apply (e.g., linear per_freq_phasor)")
    args = parser.parse_args(argv)

    fmt = load_format(args.format)
    ant = AntennaMapping.load(args.layout)

    # Determine ref and target packet indices
    ref_pidx = ant.packet_index(args.ref_ant)
    active = ant.active_antennas()
    target_aids = [a for a in active if a != args.ref_ant]
    target_pidxs = [ant.packet_index(a) for a in target_aids]

    reader = VisibilityReader(args.data_dir, args.obs, fmt)
    data = reader.read(
        ref=ref_pidx, targets=target_pidxs,
        time_start=args.time_start, time_end=args.time_end, time_tz=args.time_tz,
        nfiles=args.nfiles, skip_nfiles=args.skip_nfiles,
        freq_order=args.freq_order,
    )

    vis = data["vis"]
    freq_mhz = data["freq_mhz"]
    time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}, freq: {freq_mhz.shape}, time: {time_unix.shape}")

    # Load RFI mask
    freq_mask = None
    if args.rfi_mask:
        freq_mask = np.load(args.rfi_mask)["mask"]
        print(f"RFI mask: {np.sum(freq_mask)} / {len(freq_mask)} channels masked")

    # Transit window
    try:
        i_start, i_end = find_transit_window(
            args.source, time_unix, min_alt_deg=args.min_alt
        )
        print(f"Transit window: indices {i_start}–{i_end} "
              f"({i_end - i_start + 1} integrations)")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Using full time range.")
        i_start, i_end = 0, len(time_unix) - 1

    time_mask = np.zeros(len(time_unix), dtype=bool)
    time_mask[i_start:i_end + 1] = True

    # Source ENU
    s_enu = source_enu(args.source, time_unix)
    print(f"Source ENU shape: {s_enu.shape}")

    # Baseline vectors
    positions = ant.get_positions()
    ref_local = 0  # ref is first in positions for active antennas
    # Map antenna IDs to position indices
    active_sorted = sorted(active)
    ref_pos_idx = active_sorted.index(args.ref_ant)
    target_pos_idxs = [active_sorted.index(a) for a in target_aids]
    bl_enu = compute_baselines_enu(positions, ref_pos_idx, target_pos_idxs)
    print(f"Baseline ENU shape: {bl_enu.shape}")

    # Geometric delay and fringe-stop
    tau_s = geometric_delay(s_enu, bl_enu)
    print(f"Geometric delay shape: {tau_s.shape}")

    fs_result = fringe_stop(vis, freq_mhz, tau_s, sign=args.sign)
    vis_fs = fs_result["vis_stopped"]
    print(f"Fringe-stopped vis shape: {vis_fs.shape}")

    # Build diagnostic panels with cross-pair labels
    ref_snap_id, ref_adc = ant.snap_adc(args.ref_ant)
    ref_sac = f"S{ref_snap_id}A{ref_adc}"
    target_labels = [f"{ref_sac} x S{s}A{a}"
                     for s, a in (ant.snap_adc(aid) for aid in target_aids)]
    ref_snap = ref_snap_id
    target_snaps = [ant.snap_adc(a)[0] for a in target_aids]

    panels = [
        ("Raw phase", vis),
        ("Geometric", fs_result["geometric_phase"]),
        ("Fringe-stopped", vis_fs),
    ]

    # Optional delay correction (each model applied independently to vis_fs)
    delay_results = {}
    vis_last_corrected = vis_fs
    if args.delay_model:
        label_map = {
            "linear": "Post-delay cal (linear)",
            "per_freq_phasor": "Post-delay cal (per-freq phasor)",
        }
        for model_name in args.delay_model:
            print(f"Fitting delay model: {model_name}")
            fit_params = fit_delay(vis_fs, freq_mhz, time_mask=time_mask,
                                   freq_mask=freq_mask, model=model_name)
            vis_corrected = apply_delay(vis_fs, freq_mhz, fit_params,
                                        model=model_name)
            delay_results[model_name] = fit_params
            panels.append((label_map.get(model_name, f"Post-delay cal ({model_name})"),
                           vis_corrected))
            vis_last_corrected = vis_corrected
            print(f"  delay_ns: {fit_params.get('delay_ns', 'N/A')}")

    # Output
    dirs = None
    if not args.show or args.save_npz:
        dirs = make_output_dir(args.output_dir, args.obs)

    # Fringe diagnostic waterfalls
    fringe_out = None if args.show else dirs["fringe_stop"]
    plot_fringe_diagnostic(
        panels, time_unix, freq_mhz, target_labels,
        target_snaps, ref_snap, output_dir=fringe_out,
    )
    if not args.show:
        print(f"Saved fringe diagnostics to: {dirs['fringe_stop']}")

    # Phase vs freq
    phase_panels = [(label, data) for label, data in panels
                    if label != "Geometric"]
    phase_out = None if args.show else dirs["fringe_stop"] / "phase_vs_freq.png"
    plot_phase_vs_freq(
        phase_panels, freq_mhz, baseline_labels=target_labels,
        output_path=phase_out,
        time_unix=time_unix,
        time_mask=time_mask,
    )
    if not args.show:
        print(f"Saved phase vs freq to: {dirs['fringe_stop']}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()

    # Optional NPZ save
    if args.save_npz:
        save_dict = {
            "vis_raw": vis,
            "vis_stopped": vis_fs,
            "vis_for_calibration": vis_fs,
            "geometric_phase": fs_result["geometric_phase"],
            "tau_s": tau_s,
            "freq_mhz": freq_mhz,
            "time_unix": time_unix,
            "sign": np.array(args.sign),
        }
        if freq_mask is not None:
            save_dict["freq_mask"] = freq_mask
        for model_name, params in delay_results.items():
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    save_dict[f"delay_{model_name}_{k}"] = v
        if vis_last_corrected is not vis_fs:
            save_dict["vis_delay_corrected"] = vis_last_corrected

        npz_path = dirs["fringe_stop"] / "fringe_stop_results.npz"
        save_results(npz_path, **save_dict)
        print(f"Saved NPZ: {npz_path}")


def data_span_main(argv=None):
    """Survey a data directory and list all observations with time spans."""
    import os
    import re
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    from casm_io.correlator import load_format, VisibilityReader

    parser = argparse.ArgumentParser(
        description="List observations in a data directory with time spans"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with .dat files")
    parser.add_argument("--format", required=True,
                        help="Format name or path to JSON")
    args = parser.parse_args(argv)

    fmt = load_format(args.format)

    # Discover unique obs_ids from filenames like YYYY-MM-DD-HH:MM:SS.dat.N
    obs_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})\.dat\.\d+")
    obs_ids = set()
    for fname in os.listdir(args.data_dir):
        m = obs_pattern.match(fname)
        if m:
            obs_ids.add(m.group(1))

    if not obs_ids:
        print(f"No observations found in {args.data_dir}")
        return

    # Gather info for each obs_id
    pacific = ZoneInfo("US/Pacific")
    rows = []
    for obs_id in sorted(obs_ids):
        try:
            reader = VisibilityReader(args.data_dir, obs_id, fmt)
        except RuntimeError:
            continue
        n_files = reader.n_files
        start_unix, end_unix = reader.time_span
        duration_s = end_unix - start_unix
        duration_hr = duration_s / 3600.0

        utc_start = datetime.fromtimestamp(start_unix, tz=timezone.utc)
        utc_end = datetime.fromtimestamp(end_unix, tz=timezone.utc)
        pac_start = utc_start.astimezone(pacific)
        pac_end = utc_end.astimezone(pacific)

        dt_fmt = "%m-%d %H:%M:%S"
        rows.append((
            obs_id, n_files, duration_hr,
            utc_start.strftime(dt_fmt),
            utc_end.strftime(dt_fmt),
            pac_start.strftime(dt_fmt),
            pac_end.strftime(dt_fmt),
        ))

    # Sort by obs_id (already chronological)
    rows.sort(key=lambda r: r[0])

    # Print table
    hdr = f"{'Obs ID':<22s} {'Files':>5s} {'Duration':>10s} {'UTC Time Span':<31s} {'Pacific Time Span':<31s}"
    print(f"\nData directory: {args.data_dir}")
    print(f"Format: {args.format} (dt = {fmt.dt_raw_s:.3f} s)")
    print(f"Observations: {len(rows)}\n")
    print(hdr)
    print("-" * len(hdr))
    for obs_id, n_files, dur_hr, utc_s, utc_e, pac_s, pac_e in rows:
        print(f"{obs_id:<22s} {n_files:>5d} {dur_hr:>8.1f} h "
              f"{utc_s} – {utc_e}  {pac_s} – {pac_e}")
