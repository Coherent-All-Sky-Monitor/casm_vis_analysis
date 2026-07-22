"""Sync the wiring CSV from CAsMan.

Pulls latest CAsMan DB (auto-sync on import; `--pull` forces fresh
download), enumerates all P1 antenna→SNAP chains in chassis 1, applies a
small overrides file for cases CAsMan doesn't track (e.g. OUTRIGGER), and
writes the result to `casm_wiring.csv`. Diffs the candidate against the
current file before writing.

Apply policy: CAsMan wins. Hand-edits to `casm_wiring.csv` do not
survive `--apply` — encode them as override rows instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

from casm_vis_analysis.layout._grid import parse_grid_code
from casm_vis_analysis.layout.casman_pull import pull_casman

DEFAULT_WIRING_CSV    = Path("/home/casm/software/dev/antenna_layouts/casm_wiring.csv")
DEFAULT_OVERRIDES_CSV = Path("/home/casm/software/dev/antenna_layouts/casm_wiring_overrides.csv")
DEFAULT_SNAP_MAP_CSV  = Path("/home/casm/software/dev/antenna_layouts/casm_snap_map.csv")

WIRING_COLS = ["snap_ip", "chassis", "slot", "feng_id", "adc",
               "plank", "element", "functional", "comments"]
OVERRIDE_COLS = WIRING_COLS + ["action"]
KEY = ("chassis", "slot", "adc")
MIN_REASONABLE_ANTENNAS = 5


def _row_key(r):
    return (int(r["chassis"]), str(r["slot"]), int(r["adc"]))


def build_wiring_candidate(*, snap_map: pd.DataFrame,
                           overrides: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build a wiring candidate by taking antenna<->ADC chains from CAsMan,
    filling (snap_ip, feng_id) from the locally trusted SNAP map, then
    applying `overrides` (if given) for cases CAsMan doesn't track.

    Reason: CAsMan's `snap_boards` table holds the *design-time* IP/feng_id
    mapping per slot, which has historically diverged from the on-floor
    truth. The chain itself (`get_snap_ports_for_antenna`) is reliable.
    """
    from casman.database.antenna_positions import get_all_antenna_positions
    from casman.antenna.chain import get_snap_ports_for_antenna

    snap_lookup = {
        (int(r["chassis"]), str(r["slot"])):
            (int(r["feng_id"]), str(r["snap_ip"]))
        for _, r in snap_map.iterrows()
    }

    rows = []
    skipped = []
    for p in get_all_antenna_positions():
        ports = get_snap_ports_for_antenna(p["antenna_number"])
        p1 = ports.get("p1") or {}
        if "chassis" not in p1 or "slot" not in p1 or "port" not in p1:
            continue
        chassis, slot, port = p1["chassis"], p1["slot"], p1["port"]
        if (chassis, slot) not in snap_lookup:
            skipped.append((p["antenna_number"], chassis, slot, port))
            continue
        decoded = parse_grid_code(p["grid_code"])
        if not decoded:
            continue
        plank, element = decoded
        feng_id, ip = snap_lookup[(chassis, slot)]
        rows.append({
            "snap_ip": ip,
            "chassis": chassis,
            "slot": slot,
            "feng_id": feng_id,
            "adc": int(port),
            "plank": plank,
            "element": element,
            "functional": 1,
            "comments": "from casman",
        })

    if skipped:
        print(f"Skipped {len(skipped)} CAsMan chains whose (chassis, slot) "
              f"is not in the SNAP map (these are not currently wired):")
        for ant, c, s, port in skipped[:8]:
            print(f"    {ant}  ->  {c}/{s}/A{port:02d}")
        if len(skipped) > 8:
            print(f"    ... and {len(skipped) - 8} more")

    cand = pd.DataFrame(rows, columns=WIRING_COLS)
    print(f"CAsMan candidate: {len(cand)} P1 rows in mapped slots")

    if overrides is not None:
        cand = _apply_overrides(cand, overrides)

    return cand[WIRING_COLS].sort_values(["feng_id", "adc"]).reset_index(drop=True)


def _apply_overrides(candidate: pd.DataFrame,
                     overrides: pd.DataFrame) -> pd.DataFrame:
    """Apply replace/disable/add to the CAsMan-derived candidate."""
    cand = candidate.copy()
    by_key = {_row_key(r): i for i, r in cand.iterrows()}

    # 1. replace
    for _, ov in overrides[overrides["action"] == "replace"].iterrows():
        k = _row_key(ov)
        if k not in by_key:
            print(f"WARNING: override 'replace' for {k}: row not in CAsMan, treating as 'add'.")
            cand = pd.concat([cand, pd.DataFrame([ov[WIRING_COLS]])], ignore_index=True)
            by_key[k] = len(cand) - 1
            continue
        for col in WIRING_COLS:
            cand.at[by_key[k], col] = ov[col]

    # 2. disable
    for _, ov in overrides[overrides["action"] == "disable"].iterrows():
        k = _row_key(ov)
        if k not in by_key:
            print(f"WARNING: override 'disable' for {k}: row not in CAsMan, skipping.")
            continue
        cand.at[by_key[k], "functional"] = 0
        if ov.get("comments"):
            cand.at[by_key[k], "comments"] = ov["comments"]

    # 3. add
    for _, ov in overrides[overrides["action"] == "add"].iterrows():
        k = _row_key(ov)
        if k in by_key:
            raise ValueError(f"override 'add' for {k}: row already exists in CAsMan; "
                             "use 'replace' instead.")
        cand = pd.concat([cand, pd.DataFrame([ov[WIRING_COLS]])], ignore_index=True)

    return cand.reset_index(drop=True)


def _diff(current: pd.DataFrame, candidate: pd.DataFrame):
    """Return (added, removed, changed) keyed by (chassis, slot, adc)."""
    cur = {_row_key(r): r for _, r in current.iterrows()}
    new = {_row_key(r): r for _, r in candidate.iterrows()}
    added = [new[k] for k in new if k not in cur]
    removed = [cur[k] for k in cur if k not in new]
    changed = []
    for k in cur.keys() & new.keys():
        diffs = {c: (cur[k][c], new[k][c]) for c in WIRING_COLS
                 if str(cur[k][c]) != str(new[k][c])}
        if diffs:
            changed.append((k, diffs))
    return added, removed, changed


def _print_diff(added, removed, changed):
    if not (added or removed or changed):
        print("\nNo changes — candidate matches current wiring CSV.")
        return
    if added:
        print(f"\n[+ ADDED] {len(added)} rows:")
        for r in added:
            print(f"    {r['chassis']}/{r['slot']}/A{int(r['adc']):02d}  "
                  f"{r['plank']} {r['element']}  feng={r['feng_id']}  "
                  f"{r['snap_ip']}  func={r['functional']}  {r['comments']}")
    if removed:
        print(f"\n[- REMOVED] {len(removed)} rows:")
        for r in removed:
            print(f"    {r['chassis']}/{r['slot']}/A{int(r['adc']):02d}  "
                  f"{r['plank']} {r['element']}  feng={r['feng_id']}  "
                  f"{r['snap_ip']}  func={r['functional']}  {r['comments']}")
    if changed:
        print(f"\n[~ CHANGED] {len(changed)} rows:")
        for k, diffs in changed:
            print(f"    {k[0]}/{k[1]}/A{k[2]:02d}: " +
                  ", ".join(f"{c}: {a!r} -> {b!r}" for c, (a, b) in diffs.items()))


def run_sync_wiring(*, target_csv: Path | str | None = None,
                    overrides_csv: Path | str | None = None,
                    snap_map_csv: Path | str | None = None,
                    dry_run: bool = True,
                    force_pull: bool = False,
                    force: bool = False) -> dict:
    """Regenerate the wiring CSV from CAsMan.

    Returns
    -------
    dict
        ``{'added': [...], 'removed': [...], 'changed': [...],
        'candidate': pd.DataFrame, 'wrote': bool}``
    """
    target_csv    = Path(target_csv)    if target_csv    else DEFAULT_WIRING_CSV
    overrides_csv = Path(overrides_csv) if overrides_csv else DEFAULT_OVERRIDES_CSV
    snap_map_csv  = Path(snap_map_csv)  if snap_map_csv  else DEFAULT_SNAP_MAP_CSV

    if not snap_map_csv.exists():
        raise FileNotFoundError(
            f"SNAP map not found: {snap_map_csv}. "
            f"This file holds the trusted (chassis, slot) -> (feng_id, snap_ip) "
            f"mapping for the on-floor wiring. CAsMan's snap_boards is not used "
            f"as authoritative because it has historically diverged.")
    snap_map = pd.read_csv(snap_map_csv)
    print(f"snap map: {len(snap_map)} entries ({snap_map_csv.name})")

    if force_pull:
        pull_result = pull_casman(force=True)
        if pull_result["source"] == "github" and pull_result["release_name"]:
            print(f"Forced pull: downloading {pull_result['release_name']} ...")

    if overrides_csv.exists():
        overrides = pd.read_csv(overrides_csv)
        print(f"overrides: {len(overrides)} rows ({overrides_csv.name})")
    else:
        overrides = None
        print(f"overrides: none ({overrides_csv} not found; pure CAsMan output)")

    cand = build_wiring_candidate(snap_map=snap_map, overrides=overrides)

    current = (pd.read_csv(target_csv) if target_csv.exists()
               else pd.DataFrame(columns=WIRING_COLS))
    added, removed, changed = _diff(current, cand)
    _print_diff(added, removed, changed)

    wrote = False
    if not dry_run:
        n_casman = len(cand[cand["comments"].fillna("").str.contains("from casman")])
        if n_casman < MIN_REASONABLE_ANTENNAS and not force:
            raise RuntimeError(
                f"CAsMan returned only {n_casman} chassis-1 P1 antennas "
                f"(< {MIN_REASONABLE_ANTENNAS}); refusing to overwrite "
                f"{target_csv}. Pass force=True (CLI: --force) to override.")
        if target_csv.exists():
            backup = target_csv.with_suffix(target_csv.suffix + ".bak")
            shutil.copy2(target_csv, backup)
            print(f"\nbacked up current -> {backup}")
        target_csv.parent.mkdir(parents=True, exist_ok=True)
        cand.to_csv(target_csv, index=False)
        wrote = True
        print(f"wrote {target_csv} ({len(cand)} rows)")
        print("\nNext: rebuild the consumer layout with:\n"
              "    casm-build-layout --check-casman")
    else:
        print("\n(dry-run; pass --apply to write)")

    return {"added": added, "removed": removed, "changed": changed,
            "candidate": cand, "wrote": wrote}


def main(argv=None):
    print("tip: 'casm-layout status|diff|apply' is the friendlier interface.",
          file=sys.stderr)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="Replace casm_wiring.csv (with .bak). Default is dry-run.")
    parser.add_argument("--pull", action="store_true",
                        help="Force a fresh GitHub pull of the CAsMan DB.")
    parser.add_argument("--force", action="store_true",
                        help="Apply even if CAsMan looks empty/broken.")
    parser.add_argument("--overrides", default=None,
                        help=f"Overrides CSV (default: {DEFAULT_OVERRIDES_CSV})")
    parser.add_argument("--snap-map", default=None,
                        help=f"Trusted (chassis, slot)->(feng_id, snap_ip) "
                             f"map (default: {DEFAULT_SNAP_MAP_CSV})")
    parser.add_argument("--target", default=None,
                        help=f"Target wiring CSV (default: {DEFAULT_WIRING_CSV})")
    args = parser.parse_args(argv)
    run_sync_wiring(
        target_csv=args.target,
        overrides_csv=args.overrides,
        snap_map_csv=args.snap_map,
        dry_run=not args.apply,
        force_pull=args.pull,
        force=args.force,
    )


if __name__ == "__main__":
    main()
