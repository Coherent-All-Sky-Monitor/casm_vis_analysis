"""CLI entry points for CASM visibility analysis.

Three commands:
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
    args = parser.parse_args(argv)

    fmt = load_format(args.format)
    ant = AntennaMapping.load(args.layout)
    reader = VisibilityReader(args.data_dir, args.obs, fmt)
    data = reader.read(freq_order=args.freq_order)

    vis = data["vis"]
    freq_mhz = data["freq_mhz"]
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
        path = dirs["autocorr"] / f"autocorr_snap{snap_id}.png"
        plot_autocorr(snap_vis, freq_mhz, snap_labels,
                      output_path=path, ncols=args.ncols)
        print(f"Saved: {path}")


def waterfall_main(argv=None):
    """Waterfall matrix plot."""
    from casm_io.correlator import load_format, VisibilityReader, AntennaMapping
    from casm_vis_analysis.plotting.waterfall import plot_waterfall
    from casm_vis_analysis.output import make_output_dir

    parser = _common_parser("Plot upper-triangle waterfall matrix")
    parser.add_argument("--split-max", type=int, default=16,
                        help="Max antennas per figure (default: 16)")
    args = parser.parse_args(argv)

    fmt = load_format(args.format)
    ant = AntennaMapping.load(args.layout)
    reader = VisibilityReader(args.data_dir, args.obs, fmt)
    data = reader.read(freq_order=args.freq_order)

    vis = data["vis"]
    freq_mhz = data["freq_mhz"]
    time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}")

    labels = [ant.format_antenna(aid) for aid in ant.active_antennas()]
    dirs = make_output_dir(args.output_dir, args.obs)

    plot_waterfall(vis, freq_mhz, time_unix, fmt.nsig,
                   antenna_labels=labels, split_max=args.split_max,
                   output_dir=dirs["waterfall"])
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
    parser.add_argument("--time-start", default=None,
                        help="Start time for data loading")
    parser.add_argument("--time-end", default=None,
                        help="End time for data loading")
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
        time_start=args.time_start, time_end=args.time_end,
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

    # Build diagnostic panels
    target_labels = [ant.format_antenna(a) for a in target_aids]
    ref_snap, _ = ant.snap_adc(args.ref_ant)
    target_snaps = [ant.snap_adc(a)[0] for a in target_aids]

    panels = [
        ("Raw phase", vis),
        ("Geometric", fs_result["geometric_phase"]),
        ("Fringe-stopped", vis_fs),
    ]

    # Optional delay correction
    delay_results = {}
    vis_current = vis_fs
    if args.delay_model:
        for model_name in args.delay_model:
            print(f"Fitting delay model: {model_name}")
            fit_params = fit_delay(vis_current, freq_mhz, time_mask=time_mask,
                                   model=model_name)
            vis_corrected = apply_delay(vis_current, freq_mhz, fit_params,
                                         model=model_name)
            delay_results[model_name] = fit_params

            label_map = {
                "linear": "Post-linear",
                "per_freq_phasor": "Post-phasor",
            }
            panels.append((label_map.get(model_name, f"Post-{model_name}"),
                           vis_corrected))
            vis_current = vis_corrected
            print(f"  delay_ns: {fit_params.get('delay_ns', 'N/A')}")

    # Output
    dirs = make_output_dir(args.output_dir, args.obs)

    # Fringe diagnostic waterfalls
    plot_fringe_diagnostic(
        panels, time_unix, freq_mhz, target_labels,
        target_snaps, ref_snap, output_dir=dirs["fringe_stop"],
    )
    print(f"Saved fringe diagnostics to: {dirs['fringe_stop']}")

    # Phase vs freq
    phase_panels = [(label, data) for label, data in panels
                    if label != "Geometric"]
    plot_phase_vs_freq(
        phase_panels, freq_mhz, baseline_labels=target_labels,
        output_path=dirs["fringe_stop"] / "phase_vs_freq.png",
    )
    print(f"Saved phase vs freq to: {dirs['fringe_stop']}")

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
        if vis_current is not vis_fs:
            save_dict["vis_delay_corrected"] = vis_current

        npz_path = dirs["fringe_stop"] / "fringe_stop_results.npz"
        save_results(npz_path, **save_dict)
        print(f"Saved NPZ: {npz_path}")
