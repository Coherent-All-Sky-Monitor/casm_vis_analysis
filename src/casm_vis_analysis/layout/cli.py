"""casm-layout — the friendly front door to the antenna-layout pipeline.

The layout pipeline turns CAsMan (the assembly database, pulled from GitHub
releases) plus the surveyed positions CSV into the consumer layout CSV that
downstream tools read. This command wraps the two lower-level stages
(``casm-sync-wiring`` and ``casm-build-layout``) behind four verbs:

* ``casm-layout status``  — pull CAsMan, then print a one-line summary of how the
  current layout differs from what CAsMan now says. Read-only.
* ``casm-layout diff``    — the same, but print the full position-level diff plus
  the underlying wiring-row detail. Read-only.
* ``casm-layout preview`` — print the layout ``apply`` would write, marking each
  row added/changed/unchanged against the current layout; ``-o`` saves the full
  CSV. Read-only (apart from ``-o``).
* ``casm-layout apply``   — show the diff, confirm, then regenerate
  ``casm_wiring.csv`` and write a new dated layout CSV, updating the ``current``
  symlink.

Geographic coordinates come from the surveyed positions CSV; CAsMan decides
which antenna occupies which grid cell and how each feed is wired. A feed is
keyed by ``(snap, adc)`` and that is the level at which the diff is reported.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from casm_vis_analysis.layout import build, sync
from casm_vis_analysis.layout.casman_pull import pull_casman
from casm_vis_analysis.layout.diff import (
    diff_layouts,
    occupied_count,
    print_diff,
    render_preview_table,
    resolve_current_layout,
    summarize_diff,
)
from casm_vis_analysis.layout.sync import build_wiring_candidate


def _release_line(info: dict) -> str:
    name = info.get("release_name") or "unknown"
    if info.get("source") == "local":
        ts = info.get("timestamp") or "unknown time"
        return f"CAsMan release: {name} (offline: using local copy from {ts})"
    if info.get("downloaded"):
        return f"CAsMan release: {name} (pulled just now)"
    return f"CAsMan release: {name} (already up to date)"


def _has_diff(d: dict) -> bool:
    return any(d[k] for k in d)


class _Context:
    """Everything the shared core computes for a subcommand."""

    def __init__(self, cand_wiring, cand_layout, cur_path, cur_df,
                 wiring_csv, overrides_csv, snap_map_csv, positions_csv):
        self.cand_wiring = cand_wiring
        self.cand_layout = cand_layout
        self.cur_path = cur_path
        self.cur_df = cur_df
        self.wiring_csv = wiring_csv
        self.overrides_csv = overrides_csv
        self.snap_map_csv = snap_map_csv
        self.positions_csv = positions_csv


def _prepare(args) -> _Context:
    """Shared core used by status / diff / apply (steps 1-4)."""
    positions_csv = Path(args.positions) if args.positions else build.DEFAULT_POSITIONS_CSV
    overrides_csv = Path(args.overrides) if args.overrides else sync.DEFAULT_OVERRIDES_CSV
    snap_map_csv  = Path(args.snap_map)  if args.snap_map  else sync.DEFAULT_SNAP_MAP_CSV
    wiring_csv    = Path(args.wiring)    if args.wiring    else sync.DEFAULT_WIRING_CSV
    layout_dir    = Path(args.layout_dir) if args.layout_dir else build.DEFAULT_LAYOUT_DIR

    # When a layout dir is given, point the build stage at it too.
    if args.layout_dir:
        os.environ["CASM_LAYOUT_DIR"] = str(layout_dir)

    # 1. Pull (or report on) the CAsMan snapshot.
    info = pull_casman(offline=args.offline)
    print(_release_line(info))

    # 2. Load the SNAP map + overrides and build the wiring candidate.
    if not snap_map_csv.exists():
        print(f"error: SNAP map not found: {snap_map_csv}. "
              f"This file holds the trusted (chassis, slot) -> (feng_id, snap_ip) "
              f"mapping for the on-floor wiring. CAsMan's snap_boards is not used "
              f"as authoritative because it has historically diverged.",
              file=sys.stderr)
        raise SystemExit(1)
    snap_map = pd.read_csv(snap_map_csv)
    print(f"snap map: {len(snap_map)} entries ({snap_map_csv.name})")

    if overrides_csv.exists():
        overrides = pd.read_csv(overrides_csv)
        print(f"overrides: {len(overrides)} rows ({overrides_csv.name})")
    else:
        overrides = None
        print(f"overrides: none ({overrides_csv} not found; pure CAsMan output)")

    try:
        cand_wiring = build_wiring_candidate(snap_map=snap_map, overrides=overrides)
    except ImportError as e:
        print(f"error: the 'casman' package is required to build the wiring "
              f"candidate but could not be imported ({e}).", file=sys.stderr)
        raise SystemExit(1)

    # 3. Build the candidate consumer layout frame.
    positions = pd.read_csv(positions_csv)
    cand_layout = build.build_layout_dataframe(cand_wiring, positions)

    # 4. Resolve + load the current layout.
    cur_path = resolve_current_layout(layout_dir)
    if cur_path is not None:
        print(f"current layout: {cur_path.name}")
        # Read the columns the diff compares as strings with explicit str dtype:
        # a `slot`/`row`/etc. column mixing digit strings ("1") with the empty
        # cells of padding rows would otherwise be inferred float64 ("1"->"1.0")
        # and spuriously diff against the string-typed in-memory candidate.
        # Empty cells still come back NaN; diff._norm_str handles that.
        cur_df = pd.read_csv(cur_path, dtype={
            "row": str, "col": str, "snap_ip": str, "slot": str,
            "antenna_part_num": str, "comments": str,
        })
    else:
        print("no current layout found")
        cur_df = None

    return _Context(cand_wiring, cand_layout, cur_path, cur_df,
                    wiring_csv, overrides_csv, snap_map_csv, positions_csv)


def _cmd_status(args) -> int:
    ctx = _prepare(args)
    d = diff_layouts(ctx.cur_df, ctx.cand_layout)
    print()
    if _has_diff(d):
        print(summarize_diff(d))
        print("run 'casm-layout diff' for details")
    else:
        print("layout is up to date with CAsMan")
    return 0


def _cmd_diff(args) -> int:
    ctx = _prepare(args)
    d = diff_layouts(ctx.cur_df, ctx.cand_layout)
    print_diff(d)
    print()
    print(summarize_diff(d))

    print("\n--- wiring detail ---")
    current_wiring = (pd.read_csv(ctx.wiring_csv) if ctx.wiring_csv.exists()
                      else pd.DataFrame(columns=sync.WIRING_COLS))
    added, removed, changed = sync._diff(current_wiring, ctx.cand_wiring)
    sync._print_diff(added, removed, changed)
    return 0


def _cmd_preview(args) -> int:
    ctx = _prepare(args)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ctx.cand_layout.to_csv(out_path, index=False)
        n = len(ctx.cand_layout)
        m = int((ctx.cand_layout["functional"] == 1).sum())
        print(f"wrote {out_path} ({n} rows, {m} active)")
        return 0

    d = diff_layouts(ctx.cur_df, ctx.cand_layout)
    print()
    render_preview_table(ctx.cand_layout, ctx.cur_df, d)
    print()
    print(summarize_diff(d))
    n_pad = len(ctx.cand_layout) - occupied_count(ctx.cand_layout)
    print(f"(+{n_pad} padding rows for unconnected inputs not shown; "
          f"use -o to get the full CSV)")
    return 0


def _cmd_apply(args) -> int:
    ctx = _prepare(args)
    d = diff_layouts(ctx.cur_df, ctx.cand_layout)
    print_diff(d)
    print()
    print(summarize_diff(d))

    if not _has_diff(d):
        print("\nnothing to apply")
        return 0

    if not args.yes:
        resp = input("\nApply these changes? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted; nothing written.")
            return 0

    # The DB was already pulled in _prepare; do not pull again.
    try:
        sync.run_sync_wiring(
            dry_run=False,
            force_pull=False,
            force=args.force,
            target_csv=ctx.wiring_csv,
            overrides_csv=ctx.overrides_csv,
            snap_map_csv=ctx.snap_map_csv,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        print("(pass --force to bypass the sanity guard)", file=sys.stderr)
        return 1

    res = build.run_build_layout(
        positions_csv=ctx.positions_csv,
        wiring_csv=ctx.wiring_csv,
    )
    print(f"\ncurrent -> {Path(res['output_csv']).name}")
    return 0


def _shared_parent() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--offline", action="store_true",
                        help="Skip the GitHub pull and use the local CAsMan copy "
                             "(may be stale — no freshness check is performed).")
    parent.add_argument("--positions", default=None,
                        help=f"Positions CSV (default: {build.DEFAULT_POSITIONS_CSV})")
    parent.add_argument("--overrides", default=None,
                        help=f"Wiring overrides CSV (default: {sync.DEFAULT_OVERRIDES_CSV})")
    parent.add_argument("--snap-map", dest="snap_map", default=None,
                        help=f"Trusted (chassis, slot)->(feng_id, snap_ip) map "
                             f"(default: {sync.DEFAULT_SNAP_MAP_CSV})")
    parent.add_argument("--wiring", default=None,
                        help=f"Wiring CSV (default: {sync.DEFAULT_WIRING_CSV})")
    parent.add_argument("--layout-dir", dest="layout_dir", default=None,
                        help=f"Directory holding the layout CSVs + `current` symlink; "
                             f"apply writes new dated CSVs here "
                             f"(default: {build.DEFAULT_LAYOUT_DIR})")
    return parent


def main(argv=None):
    parent = _shared_parent()
    parser = argparse.ArgumentParser(
        prog="casm-layout",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command",
                                metavar="{status,diff,preview,apply}")

    sub.add_parser("status", parents=[parent],
                   help="Pull CAsMan and print a one-line diff summary (read-only).")
    sub.add_parser("diff", parents=[parent],
                   help="Print the full position-level + wiring-row diff (read-only).")
    p_preview = sub.add_parser("preview", parents=[parent],
                               help="Print the layout apply would write, marking each "
                                    "row against the current layout (read-only).")
    p_preview.add_argument("-o", "--output", default=None,
                           help="Instead of the table, write the full candidate layout "
                                "CSV to this file (exactly what apply would write).")
    p_apply = sub.add_parser("apply", parents=[parent],
                             help="Regenerate the wiring + layout CSVs after confirmation.")
    p_apply.add_argument("-y", "--yes", action="store_true",
                         help="Skip the interactive confirmation prompt.")
    p_apply.add_argument("--force", action="store_true",
                         help="Bypass the <5-antenna sanity guard in the wiring sync.")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 2

    dispatch = {"status": _cmd_status, "diff": _cmd_diff,
                "preview": _cmd_preview, "apply": _cmd_apply}
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
