"""Build the AntennaMapping-compatible layout CSV.

Reads two source-of-truth CSVs and writes the consumer file:

* positions  : `antenna_layout_april_ovro.csv` (lat/lon/alt per plank/element)
* wiring     : `casm_wiring.csv` (snap_ip, slot, feng_id, adc, plank, element, …)
* output     : `casm_antenna_layout_YYYY-MM-DD.csv` when ``dated=True``
               (default; updates ``current`` symlink atomically).
               Otherwise writes the legacy ``casm_antenna_layout_may2026.csv``
               for backward compatibility.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from casm_vis_analysis.layout.enu import geodetic_to_enu
from casm_vis_analysis.layout._grid import parse_grid_code

DEFAULT_POSITIONS_CSV = Path("/home/casm/software/dev/antenna_layouts/antenna_layout_april_ovro.csv")
DEFAULT_WIRING_CSV    = Path("/home/casm/software/dev/antenna_layouts/casm_wiring.csv")
DEFAULT_OUTPUT_CSV    = Path("/home/casm/software/dev/antenna_layouts/casm_antenna_layout_may2026.csv")
DEFAULT_LAYOUT_DIR    = Path("/home/casm/software/dev/antenna_layouts")


def _resolve_output_path(output_csv, dated: bool) -> Path:
    """Pick the output path. dated=True wins; output_csv overrides; legacy default last."""
    if output_csv is not None:
        return Path(output_csv)
    if dated:
        layout_dir = Path(os.environ.get("CASM_LAYOUT_DIR", DEFAULT_LAYOUT_DIR))
        date = _dt.date.today().isoformat()
        return layout_dir / f"casm_antenna_layout_{date}.csv"
    return DEFAULT_OUTPUT_CSV


def _update_current_symlink(target: Path) -> None:
    """Atomically point `<dir>/current` at the dated CSV."""
    link = target.parent / "current"
    tmp = target.parent / f".current.{os.getpid()}"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target.name)   # relative target so move-the-dir survives
    os.replace(tmp, link)
    print(f"updated symlink: {link} -> {target.name}")

ENU_ORIGIN_PLANK   = "N21"
ENU_ORIGIN_ELEMENT = "E1"

# Wiring CSV may use shorthand plank labels; positions CSV uses canonical names.
PLANK_ALIAS = {"N1": "N01", "OUT": "OUTRIGGER"}


def _annotate_with_casman(out: pd.DataFrame) -> pd.DataFrame:
    """Best-effort: fill antenna_part_num from CAsMan if available."""
    try:
        from casman.database.antenna_positions import get_all_antenna_positions
        from casman.antenna.chain import get_snap_ports_for_antenna
    except Exception as e:
        print(f"CAsMan not importable ({e}); leaving antenna_part_num blank.")
        out["antenna_part_num"] = ""
        return out

    casman_map = {}
    for p in get_all_antenna_positions():
        ports = get_snap_ports_for_antenna(p["antenna_number"])
        p1 = ports.get("p1") or {}
        if all(k in p1 for k in ("chassis", "slot", "port")):
            casman_map[(p1["chassis"], p1["slot"], p1["port"])] = p["antenna_number"]

    out["antenna_part_num"] = [
        casman_map.get((1, slot, int(adc)), "") if slot else ""
        for slot, adc in zip(out["slot"], out["adc"])
    ]
    return out


def _check_casman_diff(out: pd.DataFrame) -> dict:
    """Diff the active rows of `out` against CAsMan; print and return summary."""
    try:
        from casman.database.antenna_positions import get_all_antenna_positions
        from casman.antenna.chain import get_snap_ports_for_antenna
    except Exception as e:
        print(f"CAsMan not importable ({e}); skipping check.")
        return {"agree": 0, "mismatch": 0, "csv_only": 0, "casman_only": 0}

    casman = {}
    for p in get_all_antenna_positions():
        ports = get_snap_ports_for_antenna(p["antenna_number"])
        p1 = ports.get("p1") or {}
        if p1.get("chassis") != 1:
            continue
        decoded = parse_grid_code(p["grid_code"])
        if not decoded:
            continue
        plank, col = decoded
        casman[(p1["slot"], p1["port"])] = (plank, col, p["antenna_number"])

    df = out[(out.functional == 1) & (out.slot != "")].copy()
    csv_map = {(r.slot, int(r.adc)): (r.row, r.col) for _, r in df.iterrows()}

    print("\n=== CAsMan diff (chassis 1) ===")
    print(f"{'slot':>4} {'adc':>3}    {'CSV':>14}    {'CAsMan':>22}    status")
    print("-" * 70)
    agree = mismatch = csv_only = casman_only = 0
    for slot, adc in sorted(set(casman) | set(csv_map)):
        csv = csv_map.get((slot, adc))
        cm = casman.get((slot, adc))
        if csv and cm:
            ok = csv[0] == cm[0] and csv[1] == cm[1]
            if ok:
                agree += 1
                tag = "OK"
            else:
                mismatch += 1
                tag = "DIFF"
            print(f"{slot:>4s} {adc:>3d}    {csv[0]+' '+csv[1]:>14s}    "
                  f"{cm[0]+' '+cm[1]+' '+cm[2]:>22s}    {tag}")
        elif csv:
            csv_only += 1
            print(f"{slot:>4s} {adc:>3d}    {csv[0]+' '+csv[1]:>14s}    "
                  f"{'-':>22s}    CSV-only")
        else:
            casman_only += 1
            print(f"{slot:>4s} {adc:>3d}    {'-':>14s}    "
                  f"{cm[0]+' '+cm[1]+' '+cm[2]:>22s}    CAsMan-only")
    print(f"\nagree={agree}  mismatch={mismatch}  csv_only={csv_only}  casman_only={casman_only}")
    return {"agree": agree, "mismatch": mismatch,
            "csv_only": csv_only, "casman_only": casman_only}


def run_build_layout(*, positions_csv: Path | str | None = None,
                     wiring_csv: Path | str | None = None,
                     output_csv: Path | str | None = None,
                     dated: bool = True,
                     update_symlink: bool = True,
                     check_casman: bool = False) -> dict:
    """Build the AntennaMapping-compatible layout CSV.

    Parameters
    ----------
    positions_csv, wiring_csv : path-like
        Source CSVs. Defaults to canonical paths in
        ``/home/casm/software/dev/antenna_layouts``.
    output_csv : path-like, optional
        If given, writes here regardless of ``dated``.
    dated : bool
        When True (default), output filename is
        ``casm_antenna_layout_YYYY-MM-DD.csv`` under ``$CASM_LAYOUT_DIR``.
        When False, writes to the legacy ``casm_antenna_layout_may2026.csv``.
    update_symlink : bool
        When True (default) and ``dated`` (or output is in
        ``$CASM_LAYOUT_DIR``), atomically update the ``current`` symlink
        to point at the new file.

    Returns
    -------
    dict
        ``{'output_csv': Path, 'n_total': int, 'n_active': int,
        'casman_diff': dict | None, 'dataframe': pd.DataFrame,
        'symlink_updated': bool}``
    """
    positions_csv = Path(positions_csv) if positions_csv else DEFAULT_POSITIONS_CSV
    wiring_csv    = Path(wiring_csv)    if wiring_csv    else DEFAULT_WIRING_CSV
    output_csv    = _resolve_output_path(output_csv, dated)

    pos = pd.read_csv(positions_csv)
    wir = pd.read_csv(wiring_csv)
    print(f"positions: {pos.shape} ({positions_csv.name})")
    print(f"wiring:    {wir.shape} ({wiring_csv.name})")
    wir["plank"] = wir["plank"].replace(PLANK_ALIAS)

    merged = wir.merge(pos, on=["plank", "element"], how="left",
                       suffixes=("", "_pos"))
    unmatched = merged[merged["latitude_deg"].isna()]
    if len(unmatched):
        print("WARNING: unmatched plank/element rows:")
        print(unmatched[["snap_ip", "slot", "adc", "plank", "element"]].to_string(index=False))
    merged = merged.dropna(subset=["latitude_deg"]).reset_index(drop=True)
    print(f"after join: {merged.shape}")

    origin = pos[(pos["plank"] == ENU_ORIGIN_PLANK) &
                 (pos["element"] == ENU_ORIGIN_ELEMENT)]
    if len(origin) != 1:
        raise RuntimeError(f"ENU origin {ENU_ORIGIN_PLANK} {ENU_ORIGIN_ELEMENT} "
                           f"not unique in positions: {len(origin)}")
    lat0 = float(origin["latitude_deg"].iloc[0])
    lon0 = float(origin["longitude_deg"].iloc[0])
    alt0 = float(origin["altitude_m"].iloc[0])
    print(f"ENU origin {ENU_ORIGIN_PLANK} {ENU_ORIGIN_ELEMENT}: "
          f"lat={lat0:.7f} lon={lon0:.7f} alt={alt0:.4f}")

    e, n, u = geodetic_to_enu(
        merged["latitude_deg"].values,
        merged["longitude_deg"].values,
        merged["altitude_m"].values,
        lat0, lon0, alt0,
    )
    merged["x"] = np.round(e, 5)
    merged["y"] = np.round(n, 5)
    merged["z"] = np.round(u, 5)
    merged["packet_idx"] = merged["feng_id"].astype(int) * 12 + merged["adc"].astype(int)
    merged = merged.rename(columns={
        "feng_id": "snap",
        "plank": "row", "element": "col",
        "latitude_deg": "lat", "longitude_deg": "lon", "altitude_m": "alt",
        "source": "position_source",
    })

    out = merged[[
        "x", "y", "z", "snap", "adc", "packet_idx", "functional",
        "row", "col",
        "lat", "lon", "alt",
        "snap_ip", "slot",
        "position_source", "comments",
    ]].copy()
    out = _annotate_with_casman(out)

    # Pad with non-functional rows so every (snap, adc) is represented.
    have = set(zip(out["snap"], out["adc"]))
    pad = []
    for s in sorted(out["snap"].unique()):
        for a in range(12):
            if (s, a) in have:
                continue
            pad.append({
                "x": 0.0, "y": 0.0, "z": 0.0,
                "snap": s, "adc": a, "packet_idx": int(s) * 12 + a,
                "functional": 0, "row": "", "col": "",
                "lat": np.nan, "lon": np.nan, "alt": np.nan,
                "snap_ip": "", "slot": "",
                "position_source": "",
                "antenna_part_num": "",
                "comments": "unconnected",
            })
    if pad:
        out = pd.concat([out, pd.DataFrame(pad)], ignore_index=True)
    out = out.sort_values(["snap", "adc"]).reset_index(drop=True)
    out.insert(0, "antenna", np.arange(1, len(out) + 1))
    # Phase 2 schema extension: pos_type and include_in_beamforming columns
    # for bf_weights_generator (replaces Array64Config requirements).
    out["pos_type"] = np.where(out["functional"] == 1, "antenna", "unconnected")
    out["include_in_beamforming"] = out["functional"].astype(int)
    out = out[[
        "antenna", "x", "y", "z", "snap", "adc", "packet_idx",
        "functional", "pos_type", "include_in_beamforming",
        "row", "col",
        "lat", "lon", "alt",
        "snap_ip", "slot",
        "position_source", "antenna_part_num", "comments",
    ]]

    print(f"\nSanity: x range [{out['x'].min():.3f}, {out['x'].max():.3f}] m")
    print(f"        y range [{out['y'].min():.3f}, {out['y'].max():.3f}] m")
    print(f"        z range [{out['z'].min():.3f}, {out['z'].max():.3f}] m")
    y_n21 = out.loc[out["row"] == "N21", "y"].mean()
    y_c00 = out.loc[out["row"] == "C00", "y"].mean()
    print(f"        mean y(N21) = {y_n21:.3f}, mean y(C00) = {y_c00:.3f}, "
          f"diff = {y_c00 - y_n21:.3f} m (expect ~ -10.5)")
    assert out["packet_idx"].is_unique, "packet_idx not unique"

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    n_active = int((out.functional == 1).sum())
    print(f"\nWrote {output_csv} ({len(out)} rows, {n_active} active)")

    symlink_updated = False
    if update_symlink and output_csv.parent.resolve() == \
            Path(os.environ.get("CASM_LAYOUT_DIR", DEFAULT_LAYOUT_DIR)).resolve():
        _update_current_symlink(output_csv)
        symlink_updated = True

    diff = _check_casman_diff(out) if check_casman else None
    return {"output_csv": output_csv, "n_total": len(out),
            "n_active": n_active, "casman_diff": diff,
            "dataframe": out, "symlink_updated": symlink_updated}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the AntennaMapping-compatible layout CSV from "
                    "positions + wiring CSVs.")
    parser.add_argument("--positions", default=None,
                        help=f"Positions CSV (default: {DEFAULT_POSITIONS_CSV})")
    parser.add_argument("--wiring", default=None,
                        help=f"Wiring CSV (default: {DEFAULT_WIRING_CSV})")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: dated file in $CASM_LAYOUT_DIR)")
    parser.add_argument("--no-dated", action="store_true",
                        help=f"Write to legacy {DEFAULT_OUTPUT_CSV} instead of "
                             f"a dated file in $CASM_LAYOUT_DIR.")
    parser.add_argument("--no-symlink", action="store_true",
                        help="Skip atomic update of the `current` symlink.")
    parser.add_argument("--check-casman", action="store_true",
                        help="After building, diff against CAsMan and print conflicts")
    args = parser.parse_args(argv)
    run_build_layout(
        positions_csv=args.positions,
        wiring_csv=args.wiring,
        output_csv=args.output,
        dated=not args.no_dated,
        update_symlink=not args.no_symlink,
        check_casman=args.check_casman,
    )


if __name__ == "__main__":
    main()
