"""CLI entry points for CASM visibility analysis.

Five commands:
- casm-data-span: survey data directory and list observations
- casm-autocorr: autocorrelation power spectra
- casm-waterfall: upper-triangle waterfall matrix
- casm-fringe-stop: fringe-stop + optional delay correction + diagnostics
- casm-fit-positions: antenna x-position fitting via solar fringe-stopping
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
    parser.add_argument("--layout", default=None,
                        help="Path to antenna layout CSV (optional for autocorr/waterfall)")
    parser.add_argument("--snaps", nargs="*", type=int, default=[0, 2, 4],
                        help="SNAP boards to plot when --layout is not given (default: 0 2 4)")
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


def _grid_label(df, antenna_id):
    """Return '(row-col)' grid label if row/col columns exist, else ''."""
    if "row" not in df.columns or "col" not in df.columns:
        return ""
    row_data = df.loc[df["antenna_id"] == antenna_id]
    if row_data.empty:
        return ""
    return f"({row_data.iloc[0]['row']}-{row_data.iloc[0]['col']})"


def autocorr_main(argv=None):
    """Autocorrelation power spectrum plots."""
    from casm_io.correlator import load_format, VisibilityReader
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

    if not args.show:
        dirs = make_output_dir(args.output_dir, args.obs)

    n_adc = 12

    if args.layout is not None:
        from casm_io.correlator import AntennaMapping
        ant = AntennaMapping.load(args.layout)
        df = ant.dataframe
        active = ant.active_antennas()

        # Build autocorr indices and labels, grouped by SNAP
        snap_groups = {}
        for aid in active:
            snap_id, adc = ant.snap_adc(aid)
            pidx = ant.packet_index(aid)
            auto_idx = triu_flat_index(fmt.nsig, pidx, pidx)
            grid = _grid_label(df, aid)
            label = f"S{snap_id}A{adc} {grid}".strip()
            snap_groups.setdefault(snap_id, ([], []))
            snap_groups[snap_id][0].append(auto_idx)
            snap_groups[snap_id][1].append(label)
    else:
        # No layout — plot all 12 inputs per SNAP
        snap_groups = {}
        for snap_id in args.snaps:
            auto_indices = []
            labels = []
            for adc in range(n_adc):
                pidx = snap_id * n_adc + adc
                auto_indices.append(triu_flat_index(fmt.nsig, pidx, pidx))
                labels.append(f"S{snap_id}A{adc}")
            snap_groups[snap_id] = (auto_indices, labels)

    for snap_id, (auto_indices, snap_labels) in sorted(snap_groups.items()):
        snap_vis = vis[:, :, auto_indices]
        print(f"SNAP {snap_id}: {len(auto_indices)} inputs, shape {snap_vis.shape}")
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
    from casm_io.correlator import load_format, VisibilityReader
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

    n_adc = 12

    if args.layout is not None:
        from casm_io.correlator import AntennaMapping
        ant = AntennaMapping.load(args.layout)
        df = ant.dataframe
        active = ant.active_antennas()
        packet_indices = [ant.packet_index(aid) for aid in active]
        snap_adc_labels = [f"S{s}A{a}" for s, a in (ant.snap_adc(aid) for aid in active)]
        antenna_labels = []
        for aid in active:
            snap_id, adc = ant.snap_adc(aid)
            grid = _grid_label(df, aid)
            antenna_labels.append(f"S{snap_id}A{adc} {grid}".strip())
    else:
        # No layout — all 12 inputs per SNAP
        packet_indices = []
        antenna_labels = []
        snap_adc_labels = []
        for snap_id in sorted(args.snaps):
            for adc in range(n_adc):
                pidx = snap_id * n_adc + adc
                packet_indices.append(pidx)
                sac = f"S{snap_id}A{adc}"
                antenna_labels.append(sac)
                snap_adc_labels.append(sac)

    print(f"Plotting {len(packet_indices)} inputs")

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


def _print_geometry_table(target_labels, bl_enu, tau_s):
    """Print baseline geometry summary table."""
    bl_len = np.linalg.norm(bl_enu, axis=1)
    tau_min_ns = tau_s.min(axis=0) * 1e9
    tau_max_ns = tau_s.max(axis=0) * 1e9

    print(f"\n{'Baseline':<22s} {'|BL| (m)':>10s} {'τ_geo min (ns)':>16s} {'τ_geo max (ns)':>16s}")
    print("-" * 66)
    for i, label in enumerate(target_labels):
        print(f"  {label:<20s} {bl_len[i]:>10.2f} {tau_min_ns[i]:>+16.2f} {tau_max_ns[i]:>+16.2f}")
    print()


def _print_delay_table(target_labels, fit_params, model_name):
    """Print delay fit summary table, dispatching on model name."""
    if model_name == "linear":
        delay_ns = np.atleast_1d(fit_params["delay_ns"])
        r_sq = np.atleast_1d(fit_params["r_squared"])
        print(f"\nDelay fit ({model_name}):")
        print(f"  {'Baseline':<22s} {'delay (ns)':>12s} {'R²':>8s}")
        print("  " + "-" * 44)
        for i, label in enumerate(target_labels):
            print(f"  {label:<22s} {delay_ns[i]:>+12.3f} {r_sq[i]:>8.3f}")
    elif model_name == "per_freq_phasor":
        phase = fit_params["phasor_phase"]  # (F,) or (F, n_bl)
        if phase.ndim == 1:
            phase = phase[:, np.newaxis]
        mean_phi = np.mean(phase, axis=0)
        std_phi = np.std(phase, axis=0)
        print(f"\nDelay fit ({model_name}):")
        print(f"  {'Baseline':<22s} {'mean φ (rad)':>14s} {'std φ (rad)':>13s}")
        print("  " + "-" * 51)
        for i, label in enumerate(target_labels):
            print(f"  {label:<22s} {mean_phi[i]:>+14.3f} {std_phi[i]:>13.3f}")
    print()


def _print_antenna_delays(ant, active_sorted, ref_ant, baseline_delay_ns, target_aids):
    """Print per-antenna delay decomposition table."""
    from casm_vis_analysis.delay import build_delay_design_matrix, solve_antenna_delays

    n_ant = len(active_sorted)
    ref_idx = active_sorted.index(ref_ant)
    baseline_pairs = [(ref_idx, active_sorted.index(a)) for a in target_aids]

    A = build_delay_design_matrix(n_ant, baseline_pairs)
    ant_delays = solve_antenna_delays(
        np.atleast_1d(baseline_delay_ns), A, ref_ant_idx=ref_idx,
    )

    ref_label = ant.format_antenna(ref_ant)
    print(f"Per-antenna delays (ref: {ref_label} = 0 ns):")
    print(f"  {'Antenna':<14s} {'delay (ns)':>12s}")
    print("  " + "-" * 28)
    for i, aid in enumerate(active_sorted):
        label = ant.format_antenna(aid)
        print(f"  {label:<14s} {ant_delays[i]:>+12.3f}")
    print()


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
                        help="Path to RFI mask NPZ (bool array, key='mask', True=good channel)")
    parser.add_argument("--delay-model", nargs="*", default=None,
                        help="Delay models to apply (e.g., linear per_freq_phasor)")
    parser.add_argument("--antenna-delays", action="store_true",
                        help="Decompose baseline delays into per-antenna delays (requires linear in --delay-model)")
    args = parser.parse_args(argv)

    if args.layout is None:
        parser.error("--layout is required for casm-fringe-stop")

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
        print(f"Freq mask: {np.sum(freq_mask)} / {len(freq_mask)} channels selected (True=good)")

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

    # Baseline vectors — build positions for active antennas in sorted order
    # (get_positions() returns ALL CSV rows, which may not match active-only indexing)
    active_sorted = sorted(active)
    df = ant.dataframe
    positions = np.array([
        df.loc[df["antenna_id"] == a, ["x_m", "y_m", "z_m"]].values[0]
        for a in active_sorted
    ])
    ref_pos_idx = active_sorted.index(args.ref_ant)
    target_pos_idxs = [active_sorted.index(a) for a in target_aids]
    bl_enu = compute_baselines_enu(positions, ref_pos_idx, target_pos_idxs)
    print(f"Baseline ENU shape: {bl_enu.shape}")

    # Build cross-pair labels
    ref_snap_id, ref_adc = ant.snap_adc(args.ref_ant)
    ref_sac = f"S{ref_snap_id}A{ref_adc}"
    target_labels = [f"{ref_sac} x S{s}A{a}"
                     for s, a in (ant.snap_adc(aid) for aid in target_aids)]
    ref_snap = ref_snap_id
    target_snaps = [ant.snap_adc(a)[0] for a in target_aids]

    # Geometric delay and fringe-stop
    tau_s = geometric_delay(s_enu, bl_enu)
    print(f"Geometric delay shape: {tau_s.shape}")

    _print_geometry_table(target_labels, bl_enu, tau_s)

    fs_result = fringe_stop(vis, freq_mhz, tau_s, sign=args.sign)
    vis_fs = fs_result["vis_stopped"]
    print(f"Fringe-stopped vis shape: {vis_fs.shape}")

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
            _print_delay_table(target_labels, fit_params, model_name)

    # Per-antenna delay decomposition
    if args.antenna_delays:
        if "linear" in delay_results:
            _print_antenna_delays(
                ant, active_sorted, args.ref_ant,
                delay_results["linear"]["delay_ns"], target_aids,
            )
        else:
            print("Warning: --antenna-delays requires 'linear' in --delay-model, skipping.")

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


def _print_position_results_table(results, ref_label, axis="x"):
    """Print position fit results table."""
    a = axis  # "x" or "y"
    print(f"\nPosition fit results (ref: {ref_label}, axis: {a}):")
    print(f"  {'Antenna':<22s} {f'current_{a}':>10s} {f'fitted_{a}':>10s} "
          f"{f'sigma_{a}':>8s} {f'd{a}':>8s} {'score':>8s} {'delay_ns':>10s}")
    print("  " + "-" * 72)
    for r in results:
        d = r[f"best_{a}"] - r[f"current_{a}"]
        print(f"  {r['label']:<22s} {r[f'current_{a}']:>10.3f} {r[f'best_{a}']:>10.3f} "
              f"{r[f'sigma_{a}']:>8.3f} {d:>+8.3f} {r['best_score']:>8.4f} "
              f"{r['best_delay_ns']:>+10.3f}")
    print()


def fit_positions_main(argv=None):
    """Antenna position fitting via solar fringe-stopping."""
    from casm_io.correlator import load_format, VisibilityReader, AntennaMapping
    from casm_vis_analysis.sources import source_enu, find_transit_window
    from casm_vis_analysis.fringe_stop import compute_baselines_enu
    from casm_vis_analysis.position_fit import (
        fit_all_antennas, auto_detect_sign, choose_time_windows,
        write_corrected_layout, select_cross_plank_refs,
    )
    from casm_vis_analysis.plotting.position_fit import (
        plot_score_curves, plot_position_summary, plot_multiday_comparison,
    )
    from casm_vis_analysis.output import make_output_dir

    parser = _common_parser("Fit antenna positions via fringe-stopping")
    # Replace --obs with nargs="+" version for multi-day support
    for action in parser._actions:
        if hasattr(action, 'option_strings') and '--obs' in action.option_strings:
            parser._actions.remove(action)
            # Also remove from the optional actions group
            for group in parser._action_groups:
                if action in group._actions:
                    group._actions.remove(action)
            break
    parser._option_string_actions.pop('--obs', None)
    parser.add_argument("--obs", nargs="+", required=True,
                        help="Observation ID(s) (UTC timestamp), one or more")
    parser.add_argument("--ref-ant", type=int, default=None,
                        help="Reference antenna ID (required unless --cross-plank)")
    parser.add_argument("--cross-plank", action="store_true",
                        help="Auto-select cross-plank reference per target "
                             "(maximizes row separation to reduce cross-talk). "
                             "--ref-ant is used as position anchor if provided.")
    parser.add_argument("--source", default="sun",
                        help="Source name (default: sun)")
    parser.add_argument("--sign", default="-1",
                        help="Fringe-stop sign: -1, +1, or 'auto' (default: -1)")
    parser.add_argument("--min-alt", type=float, default=10.0,
                        help="Minimum source altitude in degrees (default: 10)")
    parser.add_argument("--rfi-mask", default=None,
                        help="Path to RFI mask NPZ (bool array, key='mask', True=good)")
    parser.add_argument("--x-range", default="-4,4",
                        help="X scan range relative to current position (default: -4,4)")
    parser.add_argument("--x-step", type=float, default=0.05,
                        help="X scan step size in meters (default: 0.05)")
    parser.add_argument("--output-layout", default=None,
                        help="Path to write corrected antenna layout CSV")
    parser.add_argument("--metric", default="circvar",
                        choices=["circvar", "stdev"],
                        help="Position metric (default: circvar)")
    parser.add_argument("--axis", default="x", choices=["x", "y"],
                        help="Position axis to fit: x (East) or y (North) "
                             "(default: x)")
    args = parser.parse_args(argv)

    if args.layout is None:
        parser.error("--layout is required for casm-fit-positions")
    if args.ref_ant is None and not args.cross_plank:
        parser.error("--ref-ant is required unless --cross-plank is set")

    # Parse axis and range
    axis = 0 if args.axis == "x" else 1
    axis_name = args.axis

    x_lo, x_hi = [float(v) for v in args.x_range.split(",")]
    x_range = (x_lo, x_hi)

    # Parse sign
    if args.sign == "auto":
        sign_value = None  # auto-detect per obs
    else:
        sign_value = int(args.sign)

    fmt = load_format(args.format)
    ant = AntennaMapping.load(args.layout)
    active = ant.active_antennas()
    active_sorted = sorted(active)
    df = ant.dataframe

    # Build positions dict
    pos_dict = {}
    for a in active_sorted:
        pos_dict[a] = df.loc[df["antenna_id"] == a,
                             ["x_m", "y_m", "z_m"]].values[0]

    def _make_label(aid):
        s, a = ant.snap_adc(aid)
        return f"S{s}A{a}"

    # Load RFI mask
    freq_mask = None
    if args.rfi_mask:
        freq_mask = np.load(args.rfi_mask)["mask"]
        print(f"Freq mask: {np.sum(freq_mask)} / {len(freq_mask)} channels (True=good)")

    # --------------- Cross-plank reference selection ---------------
    if args.cross_plank:
        # Read row info from layout
        row_col = "row"
        if row_col not in df.columns:
            parser.error("--cross-plank requires 'row' column in layout CSV")
        rows_map = {int(r["antenna_id"]): r[row_col]
                    for _, r in df.iterrows()
                    if int(r["antenna_id"]) in active}
        ref_map = select_cross_plank_refs(active_sorted, rows_map, pos_dict)

        # Determine anchor: use --ref-ant if provided, else pick the antenna
        # that is its own ref's ref (i.e., the ref for the most common row)
        if args.ref_ant is not None:
            anchor_id = args.ref_ant
        else:
            # Pick the antenna closest to x=0 overall
            anchor_id = min(active_sorted, key=lambda a: abs(pos_dict[a][0]))
        anchor_label = _make_label(anchor_id)
        print(f"Cross-plank mode: anchor = {anchor_label} (ant {anchor_id})")

        # Group targets by their assigned ref
        ref_groups = {}
        for aid in active_sorted:
            rid = ref_map[aid]
            if aid == rid:
                continue  # skip self-ref (ref is target in another group)
            ref_groups.setdefault(rid, []).append(aid)

        # Ensure anchor appears as a target in some group
        anchor_is_target = any(anchor_id in tgts for tgts in ref_groups.values())
        if not anchor_is_target and anchor_id in ref_map:
            # anchor is used as ref but not as target — it won't be fitted
            # That's fine: its position is assumed exact
            pass

        # Print ref assignments
        print(f"\nReference assignments (row separation):")
        for rid, tgts in sorted(ref_groups.items()):
            rlabel = _make_label(rid)
            rrow = rows_map[rid]
            tgt_strs = [f"{_make_label(t)}({rows_map[t]})" for t in tgts]
            print(f"  ref {rlabel} ({rrow}): {', '.join(tgt_strs)}")
    else:
        if args.ref_ant is None:
            parser.error("--ref-ant is required unless --cross-plank is set")
        anchor_id = args.ref_ant
        anchor_label = _make_label(anchor_id)
        # Single-ref mode: one group with all non-ref antennas
        ref_groups = {anchor_id: [a for a in active_sorted if a != anchor_id]}

    # Process each observation
    all_day_results = []

    for obs_id in args.obs:
        print(f"\n{'='*60}")
        print(f"Processing observation: {obs_id}")
        print(f"{'='*60}")

        # We need time/freq info before loading per-group data.
        # Load a small probe to get time_unix and freq_mhz.
        probe_ref_id = list(ref_groups.keys())[0]
        probe_tgt_id = ref_groups[probe_ref_id][0]
        probe_reader = VisibilityReader(args.data_dir, obs_id, fmt)
        probe_data = probe_reader.read(
            ref=ant.packet_index(probe_ref_id),
            targets=[ant.packet_index(probe_tgt_id)],
            time_start=args.time_start, time_end=args.time_end,
            time_tz=args.time_tz,
            nfiles=args.nfiles, skip_nfiles=args.skip_nfiles,
            freq_order=args.freq_order,
        )
        freq_mhz = probe_data["freq_mhz"]
        time_unix = probe_data["time_unix"]

        # Transit window and time masks
        if axis == 1:  # y-fitting
            from casm_vis_analysis.sources import source_altaz
            alt, _ = source_altaz(args.source, time_unix)
            above = alt >= args.min_alt
            n_above = int(np.sum(above))
            if n_above < 3:
                print(f"Warning: only {n_above} samples above {args.min_alt}°")
                print("Using full time range.")
                time_mask_fit = np.ones(len(time_unix), dtype=bool)
                time_mask_score = np.ones(len(time_unix), dtype=bool)
            else:
                time_mask_score = above
                idxs = np.where(above)[0]
                n_fit = max(3, int(len(idxs) * 0.8))
                margin = (len(idxs) - n_fit) // 2
                time_mask_fit = np.zeros(len(time_unix), dtype=bool)
                time_mask_fit[idxs[margin:margin + n_fit]] = True
                print(f"Y-fit: {n_above} samples above {args.min_alt}° "
                      f"(alt range: {alt[above].min():.1f}°–{alt[above].max():.1f}°)")
                print(f"Fit window: {np.sum(time_mask_fit)} samples, "
                      f"Score window: {np.sum(time_mask_score)} samples")
        else:  # x-fitting
            try:
                time_mask_fit, time_mask_score, transit_info = choose_time_windows(
                    time_unix, args.source, min_alt_deg=args.min_alt,
                )
                print(f"Transit: {transit_info['n_transit']} integrations, "
                      f"indices {transit_info['i_start']}–{transit_info['i_end']}")
                print(f"Fit window: {np.sum(time_mask_fit)} samples, "
                      f"Score window: {np.sum(time_mask_score)} samples")
            except ValueError as e:
                print(f"Warning: {e}")
                print("Using full time range.")
                time_mask_fit = np.ones(len(time_unix), dtype=bool)
                time_mask_score = np.ones(len(time_unix), dtype=bool)

        # Source ENU
        s_enu = source_enu(args.source, time_unix)
        print(f"Source ENU shape: {s_enu.shape}")

        # Fit each ref group
        all_group_results = {}  # aid → result dict

        for ref_id, target_ids in sorted(ref_groups.items()):
            ref_label_g = _make_label(ref_id)
            ref_pos = pos_dict[ref_id]
            tgt_pidxs = [ant.packet_index(a) for a in target_ids]
            tgt_labels = [_make_label(a) for a in target_ids]
            tgt_positions = np.array([pos_dict[a] for a in target_ids])

            print(f"\nRef group: {ref_label_g} → "
                  f"{len(target_ids)} targets")

            reader = VisibilityReader(args.data_dir, obs_id, fmt)
            data = reader.read(
                ref=ant.packet_index(ref_id), targets=tgt_pidxs,
                time_start=args.time_start, time_end=args.time_end,
                time_tz=args.time_tz,
                nfiles=args.nfiles, skip_nfiles=args.skip_nfiles,
                freq_order=args.freq_order,
            )
            vis = data["vis"]
            print(f"Loaded vis: {vis.shape}")

            # Auto-detect sign if needed
            if sign_value is None:
                bl_enu_0 = tgt_positions[0] - ref_pos
                sign_obs = auto_detect_sign(
                    vis[:, :, 0], freq_mhz, s_enu, bl_enu_0,
                    time_mask=time_mask_score, freq_mask=freq_mask,
                )
                print(f"Auto-detected sign: {sign_obs}")
            else:
                sign_obs = sign_value

            print(f"Scanning {axis_name}-positions "
                  f"(metric={args.metric}, sign={sign_obs}):")
            group_results = fit_all_antennas(
                vis, freq_mhz, s_enu, ref_pos, tgt_positions,
                tgt_labels,
                time_mask_fit=time_mask_fit,
                time_mask_score=time_mask_score,
                freq_mask=freq_mask,
                x_range=x_range,
                x_step=args.x_step,
                sign=sign_obs,
                metric=args.metric,
                axis=axis,
            )

            for aid, r in zip(target_ids, group_results):
                r["ref_label"] = ref_label_g
                all_group_results[aid] = r

        # Adjust all fitted positions to the anchor frame
        if args.cross_plank:
            # First pass: compute corrected positions for all fitted antennas
            # Start from anchor: its position is fixed at layout value
            corrected_pos = {anchor_id: pos_dict[anchor_id][axis]}

            # Iteratively resolve: if a ref's position is known, correct its targets
            max_iter = len(ref_groups) + 1
            for _ in range(max_iter):
                for ref_id, target_ids in ref_groups.items():
                    if ref_id not in corrected_pos:
                        # Check if ref was fitted as a target
                        if ref_id in all_group_results:
                            # ref was fitted relative to some other ref
                            # We need that other ref's corrected pos
                            other_ref = ref_map.get(ref_id, None)
                            if other_ref and other_ref in corrected_pos:
                                # offset for the group that fitted ref_id
                                r = all_group_results[ref_id]
                                ref_of_ref_pos = corrected_pos[other_ref]
                                # fitted pos assumed ref_of_ref at layout pos
                                layout_ref_of_ref = pos_dict[other_ref][axis]
                                adj = ref_of_ref_pos - layout_ref_of_ref
                                corrected_pos[ref_id] = r[f"best_{axis_name}"] + adj
                        else:
                            # ref was never fitted (it was only a ref) —
                            # use layout position
                            corrected_pos[ref_id] = pos_dict[ref_id][axis]
                    # Now correct targets in this group
                    if ref_id in corrected_pos:
                        ref_layout = pos_dict[ref_id][axis]
                        adj = corrected_pos[ref_id] - ref_layout
                        for aid in target_ids:
                            if aid in all_group_results and aid not in corrected_pos:
                                r = all_group_results[aid]
                                corrected_pos[aid] = r[f"best_{axis_name}"] + adj

            # Apply corrections
            for aid, r in all_group_results.items():
                if aid in corrected_pos:
                    r[f"best_{axis_name}"] = corrected_pos[aid]

        # Build unified results list in antenna order for display/plots
        results_ordered = []
        target_aids_ordered = [a for a in active_sorted if a in all_group_results]
        target_labels_ordered = [_make_label(a) for a in target_aids_ordered]
        for aid in target_aids_ordered:
            r = all_group_results[aid]
            r["label"] = _make_label(aid)
            r[f"current_{axis_name}"] = float(pos_dict[aid][axis])
            results_ordered.append(r)

        ref_label = anchor_label
        _print_position_results_table(results_ordered, ref_label, axis=axis_name)
        all_day_results.append((obs_id, results_ordered))
        target_aids = target_aids_ordered
        target_labels = target_labels_ordered

        # Plots for this observation
        if not args.show:
            dirs = make_output_dir(args.output_dir, obs_id)
            out_dir = dirs["position_fit"]
        else:
            out_dir = None

        plot_score_curves(results_ordered, ref_label, output_dir=out_dir)

        summary_path = None if args.show else dirs["position_fit"] / "position_summary.png"
        plot_position_summary(results_ordered, ref_label, output_path=summary_path)
        if summary_path:
            print(f"Saved: {summary_path}")

    # Multi-day summary
    if len(all_day_results) > 1:
        print(f"\n{'='*60}")
        print("Multi-day comparison")
        print(f"{'='*60}")

        # Print cross-day table
        print(f"\n  {'Antenna':<22s}", end="")
        for obs_id, _ in all_day_results:
            print(f" {obs_id[:16]:>18s}", end="")
        print(f" {'median':>10s} {'std':>8s}")
        print("  " + "-" * (22 + 18 * len(all_day_results) + 20))

        for ant_idx, label in enumerate(target_labels):
            print(f"  {label:<22s}", end="")
            vals = []
            for _, results in all_day_results:
                v = results[ant_idx][f"best_{axis_name}"]
                vals.append(v)
                print(f" {v:>18.3f}", end="")
            vals = np.array(vals)
            print(f" {np.median(vals):>10.3f} {np.std(vals):>8.3f}")
        print()

        # Multi-day comparison plot
        if not args.show:
            multiday_path = (
                make_output_dir(args.output_dir, "multiday")["position_fit"]
                / "multiday_comparison.png"
            )
        else:
            multiday_path = None

        plot_multiday_comparison(all_day_results, target_labels,
                                output_path=multiday_path,
                                axis=axis_name)
        if multiday_path:
            print(f"Saved: {multiday_path}")

    # Write corrected layout
    if args.output_layout:
        if len(all_day_results) == 1:
            fitted_vals = [r[f"best_{axis_name}"]
                           for r in all_day_results[0][1]]
        else:
            # Use median across days
            fitted_vals = []
            for ant_idx in range(len(target_aids)):
                vals = [results[ant_idx][f"best_{axis_name}"]
                        for _, results in all_day_results]
                fitted_vals.append(float(np.median(vals)))

        kw = {f"fitted_{axis_name}": fitted_vals}
        write_corrected_layout(args.layout, args.output_layout,
                               antenna_ids=target_aids, **kw)
        print(f"Wrote corrected layout: {args.output_layout}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
