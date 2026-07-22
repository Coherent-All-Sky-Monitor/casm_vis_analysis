"""Position-level diff between the current layout CSV and a fresh CAsMan candidate.

The consumer layout is keyed, per feed, by ``(snap, adc)`` — ``snap`` being the
feng_id. Users think in terms of *antenna positions* (which grid cell a feed
occupies and whether it is live), so this module diffs at that level rather than
the raw wiring-row level that :mod:`casm_vis_analysis.layout.sync` works in.

A feed is considered *occupied* when it carries an antenna, i.e. ``functional == 1``
or it has a grid cell (``row`` non-empty). Unconnected padding rows
(``functional == 0`` and no grid cell) are ignored.

Categories, per ``(snap, adc)``:

* ``added``    — occupied in the candidate, absent/unconnected in the current file
* ``removed``  — the opposite
* ``moved``    — occupied in both but the ``(row, col)`` grid cell changed
* ``enabled``  — same grid cell, ``functional`` flipped 0 -> 1
* ``disabled`` — same grid cell, ``functional`` flipped 1 -> 0
* ``changed``  — same grid cell and functional, but wiring metadata differs
                 (``snap_ip``, ``slot``, ``antenna_part_num``, ``comments``)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Metadata columns compared for the `changed` category (grid cell + functional
# already handled by the other categories).
META_COLS = ["snap_ip", "slot", "antenna_part_num", "comments"]

# Position-level categories (excludes `changed`, which is wiring-metadata).
POSITION_CATEGORIES = ["added", "removed", "moved", "enabled", "disabled"]

_DATED_RE = re.compile(r"^casm_antenna_layout_\d{4}-\d{2}-\d{2}\.csv$")
_LEGACY_NAME = "casm_antenna_layout_may2026.csv"


def resolve_current_layout(layout_dir: Path) -> Path | None:
    """Locate the current consumer layout CSV under ``layout_dir``.

    Preference order: the ``current`` symlink (resolved), else the newest
    ``casm_antenna_layout_YYYY-MM-DD.csv`` by name-date, else the legacy
    ``casm_antenna_layout_may2026.csv``, else ``None``.
    """
    layout_dir = Path(layout_dir)

    link = layout_dir / "current"
    if link.is_symlink() or link.exists():
        resolved = link.resolve()
        if resolved.exists():
            return resolved

    if layout_dir.is_dir():
        dated = sorted(p for p in layout_dir.iterdir()
                       if _DATED_RE.match(p.name))
        if dated:
            return dated[-1]

    legacy = layout_dir / _LEGACY_NAME
    if legacy.exists():
        return legacy

    return None


def _norm_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if pd.isna(v):
        return ""
    return str(v)


def _norm_float(v) -> float:
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _feed_map(df: pd.DataFrame) -> dict:
    """Build a ``(snap, adc) -> record`` map of normalized layout rows."""
    out = {}
    for _, r in df.iterrows():
        key = (int(r["snap"]), int(r["adc"]))
        rec = {
            "snap": key[0],
            "adc": key[1],
            "functional": int(r["functional"]) if not pd.isna(r["functional"]) else 0,
            "row": _norm_str(r.get("row")),
            "col": _norm_str(r.get("col")),
            "x": _norm_float(r.get("x")),
            "y": _norm_float(r.get("y")),
            "z": _norm_float(r.get("z")),
        }
        for c in META_COLS:
            rec[c] = _norm_str(r.get(c))
        rec["occupied"] = rec["functional"] == 1 or rec["row"] != ""
        out[key] = rec
    return out


def diff_layouts(current_df: pd.DataFrame | None,
                 candidate_df: pd.DataFrame) -> dict:
    """Compare two layout frames per ``(snap, adc)``.

    ``current_df=None`` means there is no current layout: everything occupied in
    the candidate is reported as ``added``.

    Returns a dict of lists of small plain-python dicts, ready to print or test.
    """
    cand = _feed_map(candidate_df)
    cur = _feed_map(current_df) if current_df is not None else {}

    diff = {k: [] for k in POSITION_CATEGORIES + ["changed"]}

    for key in sorted(set(cand) | set(cur)):
        c = cand.get(key)
        o = cur.get(key)
        occ_c = bool(c and c["occupied"])
        occ_o = bool(o and o["occupied"])

        if occ_c and not occ_o:
            diff["added"].append({
                "snap": c["snap"], "adc": c["adc"],
                "row": c["row"], "col": c["col"],
                "x": c["x"], "y": c["y"], "z": c["z"],
            })
        elif occ_o and not occ_c:
            diff["removed"].append({
                "snap": o["snap"], "adc": o["adc"],
                "row": o["row"], "col": o["col"],
                "x": o["x"], "y": o["y"], "z": o["z"],
            })
        elif occ_c and occ_o:
            if (c["row"], c["col"]) != (o["row"], o["col"]):
                diff["moved"].append({
                    "snap": c["snap"], "adc": c["adc"],
                    "old_row": o["row"], "old_col": o["col"],
                    "new_row": c["row"], "new_col": c["col"],
                    "dx": round(c["x"] - o["x"], 5),
                    "dy": round(c["y"] - o["y"], 5),
                    "dz": round(c["z"] - o["z"], 5),
                })
            elif c["functional"] != o["functional"]:
                cat = "enabled" if c["functional"] == 1 else "disabled"
                diff[cat].append({
                    "snap": c["snap"], "adc": c["adc"],
                    "row": c["row"], "col": c["col"],
                })
            else:
                changes = {col: [o[col], c[col]] for col in META_COLS
                           if o[col] != c[col]}
                if changes:
                    diff["changed"].append({
                        "snap": c["snap"], "adc": c["adc"],
                        "row": c["row"], "col": c["col"],
                        "changes": changes,
                    })

    return diff


def summarize_diff(diff: dict) -> str:
    """One-line human summary of a :func:`diff_layouts` result."""
    pos_total = sum(len(diff[k]) for k in POSITION_CATEGORIES)
    n_changed = len(diff["changed"])

    if pos_total == 0 and n_changed == 0:
        return "antenna layout is in sync with CAsMan"

    segs = []
    if pos_total:
        parts = [f"{len(diff[k])} {k}" for k in POSITION_CATEGORIES if diff[k]]
        word = "position" if pos_total == 1 else "positions"
        verb = "differs" if pos_total == 1 else "differ"
        segs.append(f"{pos_total} antenna {word} {verb} ({', '.join(parts)})")
    if n_changed:
        word = "change" if n_changed == 1 else "changes"
        segs.append(f"{n_changed} wiring-metadata {word}")

    return "; ".join(segs)


def _cell(row: str, col: str) -> str:
    label = f"{row} {col}".strip()
    return label if label else "(none)"


def print_diff(diff: dict) -> None:
    """Readable section-by-section listing, in the style of ``sync._print_diff``."""
    if not any(diff[k] for k in POSITION_CATEGORIES + ["changed"]):
        print("No differences — layout matches CAsMan.")
        return

    if diff["added"]:
        print(f"\n[+ ADDED] {len(diff['added'])} positions:")
        for r in diff["added"]:
            print(f"    snap={r['snap']:>2d} adc={r['adc']:>2d}  {_cell(r['row'], r['col']):>8s}  "
                  f"x={r['x']:>9.3f} y={r['y']:>9.3f} z={r['z']:>9.3f}")

    if diff["removed"]:
        print(f"\n[- REMOVED] {len(diff['removed'])} positions:")
        for r in diff["removed"]:
            print(f"    snap={r['snap']:>2d} adc={r['adc']:>2d}  {_cell(r['row'], r['col']):>8s}  "
                  f"x={r['x']:>9.3f} y={r['y']:>9.3f} z={r['z']:>9.3f}")

    if diff["moved"]:
        print(f"\n[~ MOVED] {len(diff['moved'])} positions:")
        for r in diff["moved"]:
            print(f"    snap={r['snap']:>2d} adc={r['adc']:>2d}  "
                  f"{_cell(r['old_row'], r['old_col']):>8s} -> {_cell(r['new_row'], r['new_col']):<8s}  "
                  f"dx={r['dx']:>+8.3f} dy={r['dy']:>+8.3f} dz={r['dz']:>+8.3f} m")

    if diff["enabled"]:
        print(f"\n[~ ENABLED] {len(diff['enabled'])} positions:")
        for r in diff["enabled"]:
            print(f"    snap={r['snap']:>2d} adc={r['adc']:>2d}  {_cell(r['row'], r['col']):>8s}")

    if diff["disabled"]:
        print(f"\n[~ DISABLED] {len(diff['disabled'])} positions:")
        for r in diff["disabled"]:
            print(f"    snap={r['snap']:>2d} adc={r['adc']:>2d}  {_cell(r['row'], r['col']):>8s}")

    if diff["changed"]:
        print(f"\n[~ CHANGED] {len(diff['changed'])} positions:")
        for r in diff["changed"]:
            detail = ", ".join(f"{col}: {old!r} -> {new!r}"
                               for col, (old, new) in r["changes"].items())
            print(f"    snap={r['snap']:>2d} adc={r['adc']:>2d}  "
                  f"{_cell(r['row'], r['col']):>8s}: {detail}")
