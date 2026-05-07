"""Programmatic entry points for the three CLI flows.

These are the same orchestration routines that `casm-autocorr`,
`casm-waterfall`, and `casm-fringe-stop` use; the CLIs in `cli.py` parse
argv and forward to the matching `run_*` here. Call directly from
notebooks or scripts to skip argv.

All three accept the same set of common parameters and return a dict of
the loaded arrays, computed fits, and figure objects. When `show=True`,
plots are not saved to disk; instead `plt.show()` is invoked at the end
so the inline backend (Jupyter) renders them.
"""

from __future__ import annotations

import numpy as np


def _load_mapping(layout):
    from casm_io.correlator import AntennaMapping
    return AntennaMapping.load(layout)


def _layout_groups(ant, fmt, include_inactive=False):
    """Per-SNAP autocorr indices and antenna labels (with grid annotation)."""
    from casm_io.correlator.baselines import triu_flat_index

    df = ant.dataframe
    aids = (df["antenna_id"].tolist() if include_inactive
            else ant.active_antennas())
    snap_groups = {}
    for aid in aids:
        snap_id, adc = ant.snap_adc(aid)
        pidx = ant.packet_index(aid)
        auto_idx = triu_flat_index(fmt.nsig, pidx, pidx)
        grid = ""
        if "row" in df.columns and "col" in df.columns:
            r = df.loc[df.antenna_id == aid].iloc[0]
            grid = f"({r.row}-{r.col})" if r.row else ""
        label = f"S{snap_id}A{adc} {grid}".strip()
        snap_groups.setdefault(snap_id, ([], []))
        snap_groups[snap_id][0].append(auto_idx)
        snap_groups[snap_id][1].append(label)
    return snap_groups


def _snap_only_groups(snaps, fmt):
    """Fallback for missing layout: all 12 ADCs of each requested SNAP."""
    from casm_io.correlator.baselines import triu_flat_index
    snap_groups = {}
    for s in snaps:
        idxs, labels = [], []
        for adc in range(12):
            pidx = s * 12 + adc
            idxs.append(triu_flat_index(fmt.nsig, pidx, pidx))
            labels.append(f"S{s}A{adc}")
        snap_groups[s] = (idxs, labels)
    return snap_groups


def _read_vis(data_dir, obs, fmt, *, ref=None, targets=None,
              freq_order="descending", time_start=None, time_end=None,
              time_tz="UTC", nfiles=None, skip_nfiles=0, data_root=None,
              freq_range_mhz=None):
    """Dispatch between obs-mode and time-range mode.

    - If ``obs`` is given: open that observation with ``VisibilityReader``
      and optionally trim to ``time_start``/``time_end``/``nfiles``.
    - If ``obs`` is None: ``time_start`` and ``time_end`` are required
      and auto-discovery via ``read_visibilities`` finds and stitches
      every overlapping observation under ``data_dir`` (or ``data_root``
      if ``data_dir`` is None).

    ``freq_range_mhz=(lo, hi)`` triggers ``np.memmap`` + on-disk channel
    skip in casm_io's reader (real bytes-not-read).
    """
    if obs is None:
        from casm_io.correlator import read_visibilities
        if time_start is None or time_end is None:
            raise ValueError(
                "When obs is None, both time_start and time_end are required "
                "(auto-discovery via read_visibilities)."
            )
        return read_visibilities(
            time_start=time_start, time_end=time_end, time_tz=time_tz,
            data_root=data_root or "/mnt", data_dir=data_dir,
            fmt=fmt, ref=ref, targets=targets, freq_order=freq_order,
            freq_range_mhz=freq_range_mhz,
        )

    from casm_io.correlator import VisibilityReader
    reader = VisibilityReader(data_dir, obs, fmt)
    return reader.read(
        ref=ref, targets=targets,
        freq_order=freq_order,
        freq_range_mhz=freq_range_mhz,
        time_start=time_start, time_end=time_end, time_tz=time_tz,
        nfiles=nfiles, skip_nfiles=skip_nfiles,
    )


def _maybe_make_dirs(output_dir, obs_label, *, needed):
    if not needed:
        return None
    from casm_vis_analysis.output import make_output_dir
    return make_output_dir(output_dir, obs_label)


def _obs_label(obs, time_start, time_end):
    """Label for the output subdirectory in either obs-mode or time-mode."""
    if obs is not None:
        return obs
    s = str(time_start).replace(' ', '_').replace(':', '-')
    e = str(time_end).replace(' ', '_').replace(':', '-')
    return f"range_{s}_to_{e}"


def run_autocorr(*, data_dir=None, obs=None, format, layout=None,
                 snaps=(0, 2, 4),
                 output_dir="./output", freq_order="descending",
                 time_start=None, time_end=None, time_tz="UTC",
                 nfiles=None, skip_nfiles=0, data_root=None,
                 freq_range_mhz=None, show=False,
                 ncols=4, scale="dB", include_inactive=False):
    """Autocorrelation power spectra per SNAP. Mirrors `casm-autocorr`.

    Returns
    -------
    dict
        Keys: ``vis``, ``freq_mhz``, ``time_unix``, ``figures``
        (list of matplotlib Figures, one per SNAP).
    """
    from casm_io.correlator import load_format
    from casm_vis_analysis.plotting.autocorr import plot_autocorr

    fmt = load_format(format)
    data = _read_vis(data_dir, obs, fmt,
                     freq_order=freq_order,
                     time_start=time_start, time_end=time_end, time_tz=time_tz,
                     nfiles=nfiles, skip_nfiles=skip_nfiles, data_root=data_root,
                     freq_range_mhz=freq_range_mhz)
    vis = data["vis"]; freq_mhz = data["freq_mhz"]; time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}, freq: {freq_mhz.shape}")

    if layout is not None:
        snap_groups = _layout_groups(_load_mapping(layout), fmt,
                                     include_inactive=include_inactive)
    else:
        snap_groups = _snap_only_groups(snaps, fmt)

    dirs = _maybe_make_dirs(output_dir, _obs_label(obs, time_start, time_end),
                            needed=not show)
    figures = []
    for snap_id, (idxs, labels) in sorted(snap_groups.items()):
        snap_vis = vis[:, :, idxs]
        print(f"SNAP {snap_id}: {len(idxs)} inputs, shape {snap_vis.shape}")
        path = None if show else dirs["autocorr"] / f"autocorr_snap{snap_id}.png"
        fig = plot_autocorr(snap_vis, freq_mhz, labels,
                            output_path=path, ncols=ncols,
                            time_unix=time_unix, snap_label=f"SNAP {snap_id}",
                            scale=scale)
        figures.append(fig)
        if not show:
            print(f"Saved: {path}")

    if show:
        import matplotlib.pyplot as plt
        plt.show()

    return {"vis": vis, "freq_mhz": freq_mhz, "time_unix": time_unix,
            "figures": figures}


def run_waterfall(*, data_dir=None, obs=None, format, layout=None,
                  snaps=(0, 2, 4),
                  output_dir="./output", freq_order="descending",
                  time_start=None, time_end=None, time_tz="UTC",
                  nfiles=None, skip_nfiles=0, data_root=None,
                  freq_range_mhz=None, show=False,
                  split_max=16, diag_spectra=False, pub=False,
                  include_inactive=False):
    """Upper-triangle waterfall matrix. Mirrors `casm-waterfall`.

    Returns
    -------
    dict
        Keys: ``vis``, ``freq_mhz``, ``time_unix``, ``figures``.
    """
    from casm_io.correlator import load_format
    from casm_vis_analysis.plotting.waterfall import plot_waterfall

    fmt = load_format(format)
    data = _read_vis(data_dir, obs, fmt,
                     freq_order=freq_order,
                     time_start=time_start, time_end=time_end, time_tz=time_tz,
                     nfiles=nfiles, skip_nfiles=skip_nfiles, data_root=data_root,
                     freq_range_mhz=freq_range_mhz)
    vis = data["vis"]; freq_mhz = data["freq_mhz"]; time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}")

    if layout is not None:
        ant = _load_mapping(layout)
        df = ant.dataframe
        aids = (df["antenna_id"].tolist() if include_inactive
                else ant.active_antennas())
        packet_indices = [ant.packet_index(aid) for aid in aids]
        snap_adc_labels = [f"S{s}A{a}"
                           for s, a in (ant.snap_adc(aid) for aid in aids)]
        antenna_labels = []
        for aid in aids:
            s, a = ant.snap_adc(aid)
            grid = ""
            if "row" in df.columns and "col" in df.columns:
                r = df.loc[df.antenna_id == aid].iloc[0]
                grid = f"({r.row}-{r.col})" if r.row else ""
            antenna_labels.append(f"S{s}A{a} {grid}".strip())
    else:
        packet_indices, antenna_labels, snap_adc_labels = [], [], []
        for s in sorted(snaps):
            for adc in range(12):
                packet_indices.append(s * 12 + adc)
                sac = f"S{s}A{adc}"
                antenna_labels.append(sac)
                snap_adc_labels.append(sac)

    print(f"Plotting {len(packet_indices)} inputs")
    dirs = _maybe_make_dirs(output_dir, _obs_label(obs, time_start, time_end),
                            needed=not show)

    figures = plot_waterfall(
        vis, freq_mhz, time_unix, fmt.nsig,
        packet_indices=packet_indices,
        antenna_labels=antenna_labels,
        snap_adc_labels=snap_adc_labels,
        split_max=split_max,
        output_dir=None if show else dirs["waterfall"],
        diag_spectra=diag_spectra,
        pub=pub,
    )
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    else:
        print(f"Saved waterfalls to: {dirs['waterfall']}")

    return {"vis": vis, "freq_mhz": freq_mhz, "time_unix": time_unix,
            "figures": figures}


def run_fringe_stop(*, data_dir=None, obs=None, format, layout, ref_ant,
                    source="sun", sign=-1, min_alt=10.0,
                    output_dir="./output", freq_order="descending",
                    time_start=None, time_end=None, time_tz="UTC",
                    nfiles=None, skip_nfiles=0, data_root=None,
                    freq_range_mhz=None, show=False,
                    rfi_mask=None, delay_model=None,
                    antenna_delays=False, save_npz=False):
    """Fringe-stop + optional delay correction + diagnostics.

    Mirrors `casm-fringe-stop`.

    Parameters
    ----------
    delay_model : list[str] or None
        E.g. ``["linear"]`` or ``["linear", "per_freq_phasor"]``.

    Returns
    -------
    dict
        Keys: ``vis``, ``vis_stopped``, ``vis_for_calibration``,
        ``geometric_phase``, ``tau_s``, ``freq_mhz``, ``time_unix``,
        ``time_mask``, ``target_aids``, ``target_labels``,
        ``delay_fits`` (per-model), ``figures``.
    """
    from casm_io.correlator import load_format
    from casm_vis_analysis.sources import source_enu, find_transit_window
    from casm_vis_analysis.fringe_stop import (
        compute_baselines_enu, geometric_delay, fringe_stop_array,
    )
    from casm_vis_analysis.delay import fit_delay, apply_delay
    from casm_vis_analysis.plotting.fringe_diag import plot_fringe_diagnostic
    from casm_vis_analysis.plotting.phase_freq import plot_phase_vs_freq
    from casm_vis_analysis.output import save_results
    from casm_vis_analysis.cli import (
        _print_geometry_table, _print_delay_table, _print_antenna_delays,
    )

    fmt = load_format(format)
    ant = _load_mapping(layout)

    ref_pidx = ant.packet_index(ref_ant)
    active = ant.active_antennas()
    target_aids = [a for a in active if a != ref_ant]
    target_pidxs = [ant.packet_index(a) for a in target_aids]

    data = _read_vis(data_dir, obs, fmt,
                     ref=ref_pidx, targets=target_pidxs,
                     freq_order=freq_order,
                     time_start=time_start, time_end=time_end, time_tz=time_tz,
                     nfiles=nfiles, skip_nfiles=skip_nfiles, data_root=data_root,
                     freq_range_mhz=freq_range_mhz)
    vis = data["vis"]; freq_mhz = data["freq_mhz"]; time_unix = data["time_unix"]
    print(f"Loaded vis: {vis.shape}, freq: {freq_mhz.shape}, "
          f"time: {time_unix.shape}")

    freq_mask = None
    if rfi_mask:
        mask_data = np.load(rfi_mask)
        freq_mask = mask_data["mask"]
        print(f"Freq mask: {np.sum(freq_mask)} / {len(freq_mask)} channels "
              f"selected (True=good)")
        if "freqs_mhz" in mask_data:
            mf = mask_data["freqs_mhz"]
            if len(mf) != len(freq_mhz) or abs(mf[0] - freq_mhz[0]) > 0.1:
                print(f"WARNING: RFI mask was built for "
                      f"{mf[0]:.1f}-{mf[-1]:.1f} MHz "
                      f"but data is {freq_mhz[0]:.1f}-{freq_mhz[-1]:.1f} MHz.")

    try:
        i0, i1 = find_transit_window(source, time_unix, min_alt_deg=min_alt)
        print(f"Transit window: indices {i0}–{i1} "
              f"({i1 - i0 + 1} integrations)")
    except ValueError as e:
        print(f"Warning: {e}\nUsing full time range.")
        i0, i1 = 0, len(time_unix) - 1
    time_mask = np.zeros(len(time_unix), dtype=bool)
    time_mask[i0:i1 + 1] = True

    s_enu = source_enu(source, time_unix)
    print(f"Source ENU shape: {s_enu.shape}")

    active_sorted = sorted(active)
    df = ant.dataframe
    positions = np.array([
        df.loc[df["antenna_id"] == a, ["x_m", "y_m", "z_m"]].values[0]
        for a in active_sorted
    ])
    ref_pos_idx = active_sorted.index(ref_ant)
    target_pos_idxs = [active_sorted.index(a) for a in target_aids]
    bl_enu = compute_baselines_enu(positions, ref_pos_idx, target_pos_idxs)
    print(f"Baseline ENU shape: {bl_enu.shape}")

    ref_snap_id, ref_adc = ant.snap_adc(ref_ant)
    ref_sac = f"S{ref_snap_id}A{ref_adc}"
    target_labels = [f"{ref_sac} x S{s}A{a}"
                     for s, a in (ant.snap_adc(aid) for aid in target_aids)]
    target_snaps = [ant.snap_adc(a)[0] for a in target_aids]

    tau_s = geometric_delay(s_enu, bl_enu)
    print(f"Geometric delay shape: {tau_s.shape}")
    _print_geometry_table(target_labels, bl_enu, tau_s)

    fs = fringe_stop_array(vis, freq_mhz, tau_s, sign=sign)
    vis_fs = fs["vis_stopped"]
    print(f"Fringe-stopped vis shape: {vis_fs.shape}")

    panels = [
        ("Raw phase", vis),
        ("Geometric", fs["geometric_phase"]),
        ("Fringe-stopped", vis_fs),
    ]

    delay_fits = {}
    vis_last_corrected = vis_fs
    if delay_model:
        label_map = {"linear": "Post-delay cal (linear)",
                     "per_freq_phasor": "Post-delay cal (per-freq phasor)"}
        for model_name in delay_model:
            print(f"Fitting delay model: {model_name}")
            params = fit_delay(vis_fs, freq_mhz, time_mask=time_mask,
                               freq_mask=freq_mask, model=model_name)
            corrected = apply_delay(vis_fs, freq_mhz, params, model=model_name)
            delay_fits[model_name] = params
            panels.append((label_map.get(model_name,
                                         f"Post-delay cal ({model_name})"),
                           corrected))
            vis_last_corrected = corrected
            _print_delay_table(target_labels, params, model_name)

    if antenna_delays:
        if "linear" in delay_fits:
            _print_antenna_delays(ant, active_sorted, ref_ant,
                                  delay_fits["linear"]["delay_ns"], target_aids)
        else:
            print("Warning: antenna_delays=True requires 'linear' in "
                  "delay_model, skipping.")

    dirs = _maybe_make_dirs(output_dir, _obs_label(obs, time_start, time_end),
                            needed=(not show or save_npz))

    fringe_out = None if show else dirs["fringe_stop"]
    diag_figs = plot_fringe_diagnostic(
        panels, time_unix, freq_mhz, target_labels,
        target_snaps, ref_snap_id, output_dir=fringe_out,
    )
    if not show:
        print(f"Saved fringe diagnostics to: {dirs['fringe_stop']}")

    phase_panels = [(lbl, d) for lbl, d in panels if lbl != "Geometric"]
    phase_out = None if show else dirs["fringe_stop"] / "phase_vs_freq.png"
    phase_fig = plot_phase_vs_freq(
        phase_panels, freq_mhz, baseline_labels=target_labels,
        output_path=phase_out, time_unix=time_unix, time_mask=time_mask,
    )
    if not show:
        print(f"Saved phase vs freq to: {dirs['fringe_stop']}")

    if show:
        import matplotlib.pyplot as plt
        plt.show()

    if save_npz:
        save_dict = {
            "vis_raw": vis, "vis_stopped": vis_fs,
            "vis_for_calibration": vis_fs,
            "geometric_phase": fs["geometric_phase"],
            "tau_s": tau_s, "freq_mhz": freq_mhz, "time_unix": time_unix,
            "sign": np.array(sign),
        }
        if freq_mask is not None:
            save_dict["freq_mask"] = freq_mask
        for model_name, params in delay_fits.items():
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    save_dict[f"delay_{model_name}_{k}"] = v
        if vis_last_corrected is not vis_fs:
            save_dict["vis_delay_corrected"] = vis_last_corrected
        npz_path = dirs["fringe_stop"] / "fringe_stop_results.npz"
        save_results(npz_path, **save_dict)
        print(f"Saved NPZ: {npz_path}")

    return {
        "vis": vis, "vis_stopped": vis_fs, "vis_for_calibration": vis_fs,
        "geometric_phase": fs["geometric_phase"], "tau_s": tau_s,
        "freq_mhz": freq_mhz, "time_unix": time_unix, "time_mask": time_mask,
        "target_aids": target_aids, "target_labels": target_labels,
        "delay_fits": delay_fits, "figures": diag_figs + [phase_fig],
    }
