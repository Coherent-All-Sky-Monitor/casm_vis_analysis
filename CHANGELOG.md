# Changelog

## [Unreleased] — 2026-05-16

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
