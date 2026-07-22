# Changelog

## [Unreleased]

### Added
- `casm-layout` CLI (`layout/cli.py`): friendlier front door over the two-stage layout pipeline, with three verbs — `status` (one-line diff summary), `diff` (full position-level + wiring-row diff), `apply` (confirm, then regenerate `casm_wiring.csv` and the dated consumer layout CSV). Every invocation pulls the latest CAsMan DB snapshot from GitHub first (checksum-skip if unchanged; `--offline` to skip). Registered as a `_COMMANDS` entry so `run("casm-layout ...")` works from notebooks, and as a `[project.scripts]` entry point in `pyproject.toml`.
- `layout/diff.py`: position-level diff between the current layout CSV and a freshly built CAsMan candidate, keyed by `(snap, adc)`. `resolve_current_layout` locates the current layout file (`current` symlink, else newest dated CSV, else legacy filename); `diff_layouts` categorizes changes as added/removed/moved/enabled/disabled/changed; `summarize_diff` renders the one-line summary; `print_diff` renders the full section listing.
- `layout/casman_pull.py`: `pull_casman(offline=False, force=False)` — explicit, controllable CAsMan GitHub-releases pull (replaces relying on CAsMan's inert import-time auto-sync check), with an offline/local-copy fallback and a loud stderr warning when the network pull fails.

### Changed
- `layout/sync.py`: extracted `build_wiring_candidate(snap_map, overrides)` out of `run_sync_wiring` as a standalone, file-I/O-free function (takes/returns DataFrames) so `casm-layout` can build a candidate without writing to disk; `run_sync_wiring` now calls it internally. `main()` now prints a one-line tip pointing at `casm-layout` before running.
- `layout/build.py`: extracted `build_layout_dataframe(wiring_df, positions_df)` out of `run_build_layout` as a standalone, file-I/O-free transform (merge on plank/element, ENU projection, CAsMan annotation, padding, column selection); `run_build_layout` now calls it internally. `main()` now prints a one-line tip pointing at `casm-layout` before running.
- `casm-sync-wiring` and `casm-build-layout` remain available unchanged for scripted/advanced use, now documented as legacy in favor of `casm-layout`.

## [2026-05-16]

### Added
- `beam_power.py`: applies calibration weights and fringe-stop, returns beam-power vs time per source
- `beam_validation.py`: offline validation of int8 beamforming weights via beam-grid analysis; public API is `validate_source`, `find_source_beam_transits`, `load_beams_from_int8`
- `offsource.py`: quiet-window finder, static-visibility builder, and static-visibility subtractor
- `plotting/delay_diag.py`: delay diagnostic plots
- `casm-fit-positions` CLI entry point (`cli:fit_positions_main`)
- `casm-validate-bf-weights` CLI entry point (`cli:validate_bf_weights_main`)
- `casm-viz-data-span` CLI entry point (`cli:data_span_main`)
- Two walkthrough notebooks: `casm_calibration_beamforming_walkthrough.ipynb`, `casm_pulsar_search_walkthrough.ipynb` (saved to canonical notebook dir with outputs)
- `tests/test_offsource.py`, `tests/test_beam_validation.py`

### Fixed
- `casm-fit-positions --rfi-mask`: `NameError` on `freq_mhz` used before assignment; check now runs inside the per-obs loop after `freq_mhz` is assigned
- `plot_phase_vs_freq` return-type contract: now consistently returns `list[Figure]`; `runners.py` callers updated to `.extend` rather than `.append`
- `beam_power_vs_time` cal-frequency alignment: was length-only; now checks values so a mismatched frequency grid raises immediately
- `np.load(allow_pickle=False)` enforced on all user-provided RFI mask paths in `cli.py` and `runners.py`
- `load_beams_from_int8`: added schema, version, and required-key guards on HDF5 open; malformed files raise `ValueError` naming the missing field

### Changed
- `.gitignore` extended with scratch patterns: `notebooks/images/`, `svd_*.png`, `*.jpg`, `pdmp.*`, `delay_diagnostics_example.py`
- `notebooks/end_to_end.ipynb`: outputs stripped (36 MB -> 61 KB)
- `notebooks/end_to_end_verify.ipynb`: outputs stripped (12 MB -> 48 KB); canonical walkthroughs retain outputs
