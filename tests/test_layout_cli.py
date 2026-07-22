"""Tests for the casm-layout CLI and its pipeline: diff, build, resolve.

All fixtures are synthetic — no CAsMan package, no network. Every call path
that would touch either is monkeypatched:

* ``build._annotate_with_casman`` (tries ``import casman``) -> passthrough
  that sets ``antenna_part_num=""``.
* ``cli.pull_casman`` / ``cli.build_wiring_candidate`` -> canned values.
* ``sync.build_wiring_candidate`` -> canned frame (``apply`` calls
  ``sync.run_sync_wiring``, which calls its own module-level
  ``build_wiring_candidate``, a *different* bound name than ``cli``'s).
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# The `casm_vis_analysis` top-level package is installed editable from a
# *different* checkout (the main repo, not this worktree) — see the .pth
# file in site-packages. That's fine for modules that exist in both, but
# `casm_vis_analysis.layout.cli` / `.diff` only exist on this worktree's
# branch. Nothing else in the test suite imports `casm_vis_analysis.layout`
# directly (grepped), so it's safe to steer that one subpackage at the
# worktree's copy. `import casm_vis_analysis` itself eagerly does
# `from casm_vis_analysis.layout import run_build_layout, run_sync_wiring`
# (see its __init__.py), which caches the *main-repo's* `layout`, `.build`
# and `.sync` submodules in sys.modules before we get a chance to touch
# `__path__` — so evict those cached entries and re-import after fixing
# `__path__` to prefer this worktree's src.
import casm_vis_analysis as _cva  # noqa: E402  (may already be cached from other test modules)

_WORKTREE_PKG_DIR = Path(__file__).resolve().parent.parent / "src" / "casm_vis_analysis"
if str(_WORKTREE_PKG_DIR) not in _cva.__path__:
    _cva.__path__.insert(0, str(_WORKTREE_PKG_DIR))

for _mod_name in list(sys.modules):
    if _mod_name == "casm_vis_analysis.layout" or _mod_name.startswith("casm_vis_analysis.layout."):
        del sys.modules[_mod_name]

from casm_vis_analysis.layout import build, cli, sync  # noqa: E402
from casm_vis_analysis.layout.diff import diff_layouts, resolve_current_layout, summarize_diff  # noqa: E402

assert cli.__file__.startswith(str(Path(__file__).resolve().parent.parent)), (
    f"casm_vis_analysis.layout.cli resolved outside this worktree: {cli.__file__}"
)

WIRING_COLS = sync.WIRING_COLS


# ---------------------------------------------------------------------------
# 1. diff_layouts / summarize_diff
# ---------------------------------------------------------------------------

def _row(snap, adc, functional=1, row="N01", col="E1",
         x=0.0, y=0.0, z=0.0, snap_ip="10.0.0.1", slot="1",
         antenna_part_num="A1", comments=""):
    return dict(snap=snap, adc=adc, functional=functional, row=row, col=col,
                x=x, y=y, z=z, snap_ip=snap_ip, slot=slot,
                antenna_part_num=antenna_part_num, comments=comments)


class TestDiffLayouts:
    def test_added_feed_occupies_previously_unconnected_cell(self):
        cur = pd.DataFrame([_row(1, 0, functional=0, row="", col="")])
        cand = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1",
                                   x=1.0, y=2.0, z=3.0)])
        d = diff_layouts(cur, cand)
        assert d["added"] == [{"snap": 1, "adc": 0, "row": "N01", "col": "E1",
                                "x": 1.0, "y": 2.0, "z": 3.0}]
        for k in ("removed", "moved", "enabled", "disabled", "changed"):
            assert d[k] == []

    def test_removed_feed_absent_from_candidate(self):
        cur = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1",
                                  x=1.0, y=2.0, z=3.0)])
        cand = pd.DataFrame([_row(1, 0, functional=0, row="", col="")])
        d = diff_layouts(cur, cand)
        assert d["removed"] == [{"snap": 1, "adc": 0, "row": "N01", "col": "E1",
                                  "x": 1.0, "y": 2.0, "z": 3.0}]
        for k in ("added", "moved", "enabled", "disabled", "changed"):
            assert d[k] == []

    def test_moved_reports_dx_dy_dz(self):
        cur = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1",
                                  x=0.0, y=0.0, z=0.0)])
        cand = pd.DataFrame([_row(1, 0, functional=1, row="N02", col="E1",
                                   x=1.5, y=2.5, z=0.5)])
        d = diff_layouts(cur, cand)
        assert d["moved"] == [{
            "snap": 1, "adc": 0,
            "old_row": "N01", "old_col": "E1",
            "new_row": "N02", "new_col": "E1",
            "dx": 1.5, "dy": 2.5, "dz": 0.5,
        }]
        for k in ("added", "removed", "enabled", "disabled", "changed"):
            assert d[k] == []

    def test_enabled_same_cell_functional_flip_to_1(self):
        # row assigned (reserved) but disabled -> enabled
        cur = pd.DataFrame([_row(1, 0, functional=0, row="N01", col="E1")])
        cand = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1")])
        d = diff_layouts(cur, cand)
        assert d["enabled"] == [{"snap": 1, "adc": 0, "row": "N01", "col": "E1"}]
        for k in ("added", "removed", "moved", "disabled", "changed"):
            assert d[k] == []

    def test_disabled_same_cell_functional_flip_to_0(self):
        cur = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1")])
        cand = pd.DataFrame([_row(1, 0, functional=0, row="N01", col="E1")])
        d = diff_layouts(cur, cand)
        assert d["disabled"] == [{"snap": 1, "adc": 0, "row": "N01", "col": "E1"}]
        for k in ("added", "removed", "moved", "enabled", "changed"):
            assert d[k] == []

    def test_changed_wiring_metadata_snap_ip(self):
        cur = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1",
                                  snap_ip="10.0.0.1")])
        cand = pd.DataFrame([_row(1, 0, functional=1, row="N01", col="E1",
                                   snap_ip="10.0.0.2")])
        d = diff_layouts(cur, cand)
        assert d["changed"] == [{
            "snap": 1, "adc": 0, "row": "N01", "col": "E1",
            "changes": {"snap_ip": ["10.0.0.1", "10.0.0.2"]},
        }]
        for k in ("added", "removed", "moved", "enabled", "disabled"):
            assert d[k] == []

    def test_current_none_all_occupied_added_padding_ignored(self):
        cand = pd.DataFrame([
            _row(1, 0, functional=1, row="N01", col="E1"),   # occupied -> added
            _row(1, 1, functional=0, row="", col=""),        # padding -> ignored
        ])
        d = diff_layouts(None, cand)
        assert len(d["added"]) == 1
        assert d["added"][0]["snap"] == 1
        assert d["added"][0]["adc"] == 0
        for k in ("removed", "moved", "enabled", "disabled", "changed"):
            assert d[k] == []

    def test_in_sync_all_categories_empty_and_message(self):
        frame = pd.DataFrame([
            _row(1, 0, functional=1, row="N01", col="E1"),
            _row(1, 1, functional=0, row="", col=""),
        ])
        d = diff_layouts(frame.copy(), frame.copy())
        for k in ("added", "removed", "moved", "enabled", "disabled", "changed"):
            assert d[k] == []
        assert summarize_diff(d) == "antenna layout is in sync with CAsMan"


class TestSummarizeDiffPhrasing:
    def _empty(self, **overrides):
        d = {"added": [], "removed": [], "moved": [], "enabled": [],
             "disabled": [], "changed": []}
        d.update(overrides)
        return d

    def test_singular_position(self):
        d = self._empty(added=[{"a": 1}])
        assert summarize_diff(d) == "1 antenna position differs (1 added)"

    def test_plural_positions_multi_category(self):
        d = self._empty(added=[{"a": 1}], removed=[{"a": 1}])
        assert summarize_diff(d) == "2 antenna positions differ (1 added, 1 removed)"

    def test_singular_wiring_change(self):
        d = self._empty(changed=[{"a": 1}])
        assert summarize_diff(d) == "1 wiring-metadata change"

    def test_plural_wiring_changes(self):
        d = self._empty(changed=[{"a": 1}, {"a": 2}])
        assert summarize_diff(d) == "2 wiring-metadata changes"

    def test_combined_position_and_wiring_segments(self):
        d = self._empty(moved=[{"a": 1}], changed=[{"a": 1}])
        assert summarize_diff(d) == (
            "1 antenna position differs (1 moved); 1 wiring-metadata change"
        )

    def test_in_sync_message(self):
        assert summarize_diff(self._empty()) == "antenna layout is in sync with CAsMan"


# ---------------------------------------------------------------------------
# 2. build_layout_dataframe
# ---------------------------------------------------------------------------

class TestBuildLayoutDataframe:
    @pytest.fixture(autouse=True)
    def _passthrough_annotate(self, monkeypatch):
        # Keep hermetic: real _annotate_with_casman tries `import casman`.
        monkeypatch.setattr(
            build, "_annotate_with_casman",
            lambda out: out.assign(antenna_part_num=""),
        )

    def _wiring(self):
        return pd.DataFrame([
            # plank alias N1 -> N01
            {"snap_ip": "10.0.0.5", "chassis": 1, "slot": "1", "feng_id": 5,
             "adc": 0, "plank": "N1", "element": "E1", "functional": 1,
             "comments": "from casman"},
            # origin itself
            {"snap_ip": "10.0.0.5", "chassis": 1, "slot": "2", "feng_id": 5,
             "adc": 1, "plank": "N21", "element": "E1", "functional": 1,
             "comments": "from casman"},
            # references a plank/element missing from positions
            {"snap_ip": "10.0.0.5", "chassis": 1, "slot": "3", "feng_id": 5,
             "adc": 2, "plank": "ZZ99", "element": "E1", "functional": 1,
             "comments": "from casman"},
        ], columns=WIRING_COLS)

    def _positions(self):
        return pd.DataFrame([
            {"plank": "N01", "element": "E1", "latitude_deg": 37.2314,
             "longitude_deg": -118.2941, "altitude_m": 1222.5, "source": "survey"},
            {"plank": "N21", "element": "E1", "latitude_deg": 37.2320,
             "longitude_deg": -118.2941, "altitude_m": 1222.5, "source": "survey"},
        ])

    def test_full_transform(self):
        wiring = self._wiring()
        wiring_alias_snapshot = wiring["plank"].tolist()
        positions = self._positions()

        out = build.build_layout_dataframe(wiring, positions)

        # unmatched (ZZ99) row dropped: only snap=5 adc=2 shows up as padding,
        # not as an active "from casman" row.
        zz_active = out[(out["adc"] == 2) & (out["comments"] == "from casman")]
        assert zz_active.empty
        pad_row = out[(out["snap"] == 5) & (out["adc"] == 2)].iloc[0]
        assert pad_row["functional"] == 0
        assert pad_row["comments"] == "unconnected"
        assert pad_row["row"] == ""

        # ENU origin (N21/E1) is ~ (0, 0, 0)
        origin_row = out[(out["row"] == "N21") & (out["col"] == "E1")].iloc[0]
        assert origin_row["x"] == pytest.approx(0.0, abs=1e-5)
        assert origin_row["y"] == pytest.approx(0.0, abs=1e-5)
        assert origin_row["z"] == pytest.approx(0.0, abs=1e-5)

        # packet_idx = snap*12 + adc, for every row (active + padded)
        assert (out["packet_idx"] == out["snap"] * 12 + out["adc"]).all()

        # exactly one snap (5) present -> 12 rows total (2 active + 10 padding)
        assert set(out["snap"].unique()) == {5}
        assert len(out) == 12
        assert (out["functional"] == 0).sum() == 10
        assert (out["functional"] == 1).sum() == 2

        # padding rows: functional=0, comments="unconnected"
        pads = out[out["functional"] == 0]
        assert (pads["comments"] == "unconnected").all()
        assert (pads["row"] == "").all()

        # pos_type / include_in_beamforming derived from functional
        active = out[out["functional"] == 1]
        assert (active["pos_type"] == "antenna").all()
        assert (active["include_in_beamforming"] == 1).all()
        assert (pads["pos_type"] == "unconnected").all()
        assert (pads["include_in_beamforming"] == 0).all()

        # documented column order
        assert list(out.columns) == [
            "antenna", "x", "y", "z", "snap", "adc", "packet_idx",
            "functional", "pos_type", "include_in_beamforming",
            "row", "col",
            "lat", "lon", "alt",
            "snap_ip", "slot",
            "position_source", "antenna_part_num", "comments",
        ]

        # input wiring frame not mutated: the plank alias must not leak back
        assert wiring["plank"].tolist() == wiring_alias_snapshot
        assert "N1" in wiring["plank"].tolist()
        assert "N01" not in wiring["plank"].tolist()


# ---------------------------------------------------------------------------
# 3. resolve_current_layout
# ---------------------------------------------------------------------------

class TestResolveCurrentLayout:
    def test_symlink_wins_over_newer_dated_file(self, tmp_path):
        older = tmp_path / "casm_antenna_layout_2026-07-01.csv"
        newer = tmp_path / "casm_antenna_layout_2026-07-15.csv"
        older.write_text("a\n")
        newer.write_text("b\n")
        (tmp_path / "current").symlink_to(older.name)

        result = resolve_current_layout(tmp_path)
        assert result == older.resolve()

    def test_no_symlink_picks_newest_dated_by_name(self, tmp_path):
        older = tmp_path / "casm_antenna_layout_2026-06-01.csv"
        newer = tmp_path / "casm_antenna_layout_2026-07-15.csv"
        older.write_text("a\n")
        newer.write_text("b\n")

        result = resolve_current_layout(tmp_path)
        assert result == newer

    def test_only_legacy_file_used(self, tmp_path):
        legacy = tmp_path / "casm_antenna_layout_may2026.csv"
        legacy.write_text("legacy\n")

        result = resolve_current_layout(tmp_path)
        assert result == legacy

    def test_empty_dir_returns_none(self, tmp_path):
        assert resolve_current_layout(tmp_path) is None

    def test_broken_symlink_falls_back_to_dated(self, tmp_path):
        dated = tmp_path / "casm_antenna_layout_2026-07-01.csv"
        dated.write_text("a\n")
        (tmp_path / "current").symlink_to(tmp_path / "does_not_exist.csv")

        result = resolve_current_layout(tmp_path)
        assert result == dated


# ---------------------------------------------------------------------------
# 4. CLI plumbing
# ---------------------------------------------------------------------------

def _canned_wiring():
    # NOTE: slot labels are deliberately non-numeric ("S1", not "1"). A
    # purely-digit "slot" column, once written to the layout CSV and
    # reloaded (pd.read_csv), gets coerced to float64 by pandas wherever
    # padding rows contribute empty cells (e.g. "1" round-trips to "1.0")
    # -- see the module docstring note below. That would make even a
    # genuinely in-sync layout look like it has spurious wiring-metadata
    # "changed" rows. Non-numeric slot IDs sidestep that CSV round-trip
    # quirk so these fixtures reflect the intended semantics rather than
    # an incidental dtype artifact.
    rows = []
    for i in range(1, 6):
        rows.append({
            "snap_ip": "10.0.0.5", "chassis": 1, "slot": f"S{i}", "feng_id": 5,
            "adc": i - 1, "plank": f"N0{i}", "element": "E1",
            "functional": 1, "comments": "from casman",
        })
    return pd.DataFrame(rows, columns=WIRING_COLS)


def _positions():
    rows = []
    for i in range(1, 6):
        rows.append({
            "plank": f"N0{i}", "element": "E1",
            "latitude_deg": 37.2300 + i * 0.0002, "longitude_deg": -118.2900,
            "altitude_m": 1222.0, "source": "survey",
        })
    rows.append({
        "plank": "N21", "element": "E1",
        "latitude_deg": 37.2320, "longitude_deg": -118.2900,
        "altitude_m": 1222.0, "source": "survey",
    })
    return pd.DataFrame(rows)


def _fake_pull_casman(*, offline=False, force=False):
    return {"release_name": "database-snapshot-test",
            "timestamp": "2026-07-21T00:00:00",
            "downloaded": False, "source": "github"}


def _fake_build_wiring_candidate(*, snap_map=None, overrides=None):
    return _canned_wiring().copy()


def _setup_layout_dir(tmp_path, monkeypatch, in_sync=False):
    """Build a tmp layout dir + monkeypatch every network/casman touchpoint."""
    monkeypatch.setattr(build, "_annotate_with_casman",
                        lambda out: out.assign(antenna_part_num=""))
    monkeypatch.setattr(cli, "pull_casman", _fake_pull_casman)
    monkeypatch.setattr(cli, "build_wiring_candidate", _fake_build_wiring_candidate)
    monkeypatch.setattr(sync, "build_wiring_candidate", _fake_build_wiring_candidate)
    monkeypatch.delenv("CASM_LAYOUT_DIR", raising=False)

    layout_dir = tmp_path / "layouts"
    layout_dir.mkdir()

    positions_csv = layout_dir / "positions.csv"
    _positions().to_csv(positions_csv, index=False)

    snap_map_csv = layout_dir / "snap_map.csv"
    pd.DataFrame({"chassis": [1], "slot": ["1"], "feng_id": [5],
                  "snap_ip": ["10.0.0.5"]}).to_csv(snap_map_csv, index=False)

    overrides_csv = layout_dir / "overrides.csv"  # deliberately absent

    wiring = _canned_wiring()
    wiring_csv = layout_dir / "casm_wiring.csv"
    if in_sync:
        wiring.to_csv(wiring_csv, index=False)
    else:
        # drop N05 row so the wiring-level diff shows an [+ ADDED]
        wiring.iloc[:-1].to_csv(wiring_csv, index=False)

    cand_layout = build.build_layout_dataframe(_canned_wiring(), _positions())

    if in_sync:
        current_layout = cand_layout.copy()
    else:
        current_layout = cand_layout.copy()
        mask = (current_layout["row"] == "N05") & (current_layout["functional"] == 1)
        current_layout.loc[mask, "functional"] = 0
        current_layout.loc[mask, "pos_type"] = "unconnected"
        current_layout.loc[mask, "include_in_beamforming"] = 0

    dated_csv = layout_dir / "casm_antenna_layout_2026-07-01.csv"
    current_layout.to_csv(dated_csv, index=False)
    (layout_dir / "current").symlink_to(dated_csv.name)

    return {
        "layout_dir": layout_dir,
        "positions_csv": positions_csv,
        "snap_map_csv": snap_map_csv,
        "overrides_csv": overrides_csv,
        "wiring_csv": wiring_csv,
        "dated_csv": dated_csv,
        "cand_layout": cand_layout,
    }


def _argv(cmd, paths, extra=None):
    argv = [cmd,
            "--positions", str(paths["positions_csv"]),
            "--overrides", str(paths["overrides_csv"]),
            "--snap-map", str(paths["snap_map_csv"]),
            "--wiring", str(paths["wiring_csv"]),
            "--layout-dir", str(paths["layout_dir"])]
    if extra:
        argv += extra
    return argv


class TestCliStatus:
    def test_status_reports_release_and_summary(self, tmp_path, monkeypatch, capsys):
        paths = _setup_layout_dir(tmp_path, monkeypatch)
        rc = cli.main(_argv("status", paths))
        out = capsys.readouterr().out
        assert rc == 0
        assert "CAsMan release: database-snapshot-test (already up to date)" in out
        assert "enabled" in out  # N05 flip surfaces in the summary
        assert "run 'casm-layout diff' for details" in out


class TestCliDiff:
    def test_diff_renders_position_and_wiring_sections(self, tmp_path, monkeypatch, capsys):
        paths = _setup_layout_dir(tmp_path, monkeypatch)
        rc = cli.main(_argv("diff", paths))
        out = capsys.readouterr().out
        assert rc == 0
        assert "[~ ENABLED]" in out
        assert "--- wiring detail ---" in out
        assert "[+ ADDED]" in out  # N05 wiring row missing from wiring_csv


class TestCliApply:
    def test_apply_yes_writes_new_dated_csv_and_repoints_symlink(
            self, tmp_path, monkeypatch, capsys):
        paths = _setup_layout_dir(tmp_path, monkeypatch)
        rc = cli.main(_argv("apply", paths, extra=["--yes"]))
        out = capsys.readouterr().out
        assert rc == 0

        backup = paths["wiring_csv"].with_suffix(".csv.bak")
        assert backup.exists()

        today = dt.date.today().isoformat()
        new_csv = paths["layout_dir"] / f"casm_antenna_layout_{today}.csv"
        assert new_csv.exists()
        assert new_csv != paths["dated_csv"]

        link = paths["layout_dir"] / "current"
        assert os.readlink(link) == new_csv.name

        assert f"current -> {new_csv.name}" in out

        # wiring csv now has the full 5-row candidate (N05 restored)
        written = pd.read_csv(paths["wiring_csv"])
        assert len(written) == 5
        assert "N05" in written["plank"].tolist()

    def test_apply_abort_on_no_writes_nothing(self, tmp_path, monkeypatch, capsys):
        paths = _setup_layout_dir(tmp_path, monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")

        before_bytes = paths["wiring_csv"].read_bytes()
        before_files = sorted(p.name for p in paths["layout_dir"].iterdir())

        rc = cli.main(_argv("apply", paths))
        out = capsys.readouterr().out
        assert rc == 0
        assert "aborted; nothing written." in out

        after_bytes = paths["wiring_csv"].read_bytes()
        after_files = sorted(p.name for p in paths["layout_dir"].iterdir())
        assert after_bytes == before_bytes
        assert after_files == before_files
        assert not paths["wiring_csv"].with_suffix(".csv.bak").exists()

    def test_apply_in_sync_skips_prompt(self, tmp_path, monkeypatch, capsys):
        paths = _setup_layout_dir(tmp_path, monkeypatch, in_sync=True)

        def _no_input(prompt=""):
            raise AssertionError("input() should not be called when in sync")
        monkeypatch.setattr("builtins.input", _no_input)

        rc = cli.main(_argv("apply", paths))
        out = capsys.readouterr().out
        assert rc == 0
        assert "nothing to apply" in out


# ---------------------------------------------------------------------------
# 5. Regression: numeric slot column survives the current-layout CSV round-trip
# ---------------------------------------------------------------------------

def _canned_wiring_numeric_slot():
    # Digit slot strings ("1"), unlike _canned_wiring's "S1" IDs. When the
    # built layout is written to CSV, the padding rows contribute empty slot
    # cells, so a bare pd.read_csv infers float64 and "1" round-trips to "1.0".
    rows = []
    for i in range(1, 6):
        rows.append({
            "snap_ip": "10.0.0.5", "chassis": 1, "slot": "1", "feng_id": 5,
            "adc": i - 1, "plank": f"N0{i}", "element": "E1",
            "functional": 1, "comments": "from casman",
        })
    return pd.DataFrame(rows, columns=WIRING_COLS)


class TestCliNumericSlotCsvRoundTrip:
    def test_status_in_sync_despite_numeric_slot_roundtrip(
            self, tmp_path, monkeypatch, capsys):
        # Without cli._prepare's explicit str dtype on read, the current-layout
        # CSV's digit slot "1" reloads as "1.0" and diffs against the
        # string-typed in-memory candidate's "1", spuriously reporting the feed
        # as a wiring-metadata `changed` row. This exercises the fixed read path
        # end-to-end via cli.main(["status", ...]) and asserts in-sync.
        monkeypatch.setattr(build, "_annotate_with_casman",
                            lambda out: out.assign(antenna_part_num=""))
        monkeypatch.setattr(cli, "pull_casman", _fake_pull_casman)
        wiring = _canned_wiring_numeric_slot()
        monkeypatch.setattr(cli, "build_wiring_candidate",
                            lambda *, snap_map=None, overrides=None: wiring.copy())
        monkeypatch.delenv("CASM_LAYOUT_DIR", raising=False)

        layout_dir = tmp_path / "layouts"
        layout_dir.mkdir()

        positions_csv = layout_dir / "positions.csv"
        _positions().to_csv(positions_csv, index=False)
        snap_map_csv = layout_dir / "snap_map.csv"
        pd.DataFrame({"chassis": [1], "slot": ["1"], "feng_id": [5],
                      "snap_ip": ["10.0.0.5"]}).to_csv(snap_map_csv, index=False)
        overrides_csv = layout_dir / "overrides.csv"  # deliberately absent
        wiring_csv = layout_dir / "casm_wiring.csv"
        wiring.to_csv(wiring_csv, index=False)

        # Current layout = the candidate itself, round-tripped through a real
        # CSV file on disk (with digit slots + at least one empty-slot pad row).
        cand_layout = build.build_layout_dataframe(wiring.copy(), _positions())
        assert "1" in cand_layout["slot"].tolist()          # active rows
        assert "" in cand_layout["slot"].tolist()            # padding rows
        assert (cand_layout["functional"] == 0).any()        # >=1 padding row

        dated_csv = layout_dir / "casm_antenna_layout_2026-07-01.csv"
        cand_layout.to_csv(dated_csv, index=False)
        (layout_dir / "current").symlink_to(dated_csv.name)
        # Confirm the landmine exists on a naive read: "1" coerces to "1.0".
        assert "1.0" in pd.read_csv(dated_csv)["slot"].astype(str).tolist()

        paths = {"positions_csv": positions_csv, "overrides_csv": overrides_csv,
                 "snap_map_csv": snap_map_csv, "wiring_csv": wiring_csv,
                 "layout_dir": layout_dir}
        rc = cli.main(_argv("status", paths))
        out = capsys.readouterr().out
        assert rc == 0
        assert "layout is up to date with CAsMan" in out
        assert "wiring-metadata" not in out
        assert "changed" not in out.lower()
