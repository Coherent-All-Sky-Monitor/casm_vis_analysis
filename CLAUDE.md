# casm_vis_analysis

Fringe-stopping, delay correction, and diagnostic plotting for CASM correlator visibilities.

## Project Rules
- Use `casm_io` for all data loading — no hardcoded file parsing
- RFI mask is always user-provided — no built-in defaults
- Print array shapes when loading data (verbose mode)
- Test each module independently with synthetic fixtures
- Fringe-stopped visibilities are the API contract for casm_calibration (NOT delay-corrected)
- Frequency order: descending (native) by default, user can override to ascending
- Antenna labels: use `AntennaMapping.format_antenna()` from casm_io

## Package Layout
- `src/casm_vis_analysis/` — installable package (src layout)
- `tests/` — pytest tests with synthetic fixtures
- Entry points defined in `pyproject.toml`

### Modules
| Module | Purpose |
|--------|---------|
| `sources.py` | Source position/ENU, transit window detection |
| `fringe_stop.py` | Baseline ENU, geometric delay, fringe-stopping |
| `delay.py` | Delay fitting/correction (linear, per_freq_phasor), antenna decomposition |
| `output.py` | Output directory creation, NPZ saving |
| `runners.py` | Programmatic orchestrators: `run_autocorr`, `run_waterfall`, `run_fringe_stop` |
| `layout/` | Antenna-layout pipeline: `run_sync_wiring` (CAsMan -> wiring CSV) and `run_build_layout` (wiring + positions -> consumer CSV) |
| `layout/cli.py` | `casm-layout` — front door over the pipeline: `status`/`diff`/`apply` |
| `layout/casman_pull.py` | `pull_casman(offline=, force=)` — explicit GitHub-releases pull of the CAsMan DB snapshot, with local-copy fallback |
| `layout/diff.py` | Position-level diff (keyed by `(snap, adc)`) between the current layout CSV and a freshly built CAsMan candidate: `resolve_current_layout`, `diff_layouts`, `summarize_diff`, `print_diff` |
| `cli.py` | Argparse wrappers around runners (one per CLI entry point) |
| `plotting/autocorr.py` | Autocorrelation power spectra |
| `plotting/waterfall.py` | Upper-triangle waterfall matrix |
| `plotting/fringe_diag.py` | Fringe-stop diagnostic waterfalls (grouped by SNAP pair) |
| `plotting/phase_freq.py` | Phase vs frequency plots |

### CLI Entry Points
- `casm-autocorr` → `cli:autocorr_main` — autocorrelation power spectra per SNAP
- `casm-waterfall` → `cli:waterfall_main` — upper-triangle waterfall matrix
- `casm-fringe-stop` → `cli:fringe_stop_main` — fringe-stop + delay correction + diagnostics
- `casm-layout` → `layout.cli:main` — `status`/`diff`/`apply` front door over the layout pipeline (primary)
- `casm-sync-wiring` → `layout.sync:main` — legacy; regenerate `casm_wiring.csv` from CAsMan
- `casm-build-layout` → `layout.build:main` — legacy; produce the AntennaMapping consumer CSV

### Common CLI Flags
All three commands share: `--data-dir`, `--obs`, `--format`, `--layout`, `--output-dir`, `--freq-order`, `--time-start`, `--time-end`, `--time-tz`, `--nfiles`, `--skip-nfiles`, `--show`

### Jupyter / Programmatic Usage
Two equivalent options:

```python
# 1. Direct Python API (preferred for notebooks — typed kwargs, returns dict)
from casm_vis_analysis import run_autocorr, run_waterfall, run_fringe_stop
out = run_fringe_stop(
    data_dir="/path", obs="2026-04-29-19:40:39", format="layout_64ant",
    layout="/path/layout.csv", ref_ant=10, source="sun",
    nfiles=1, show=True,
)
# out is a dict: vis, vis_stopped, freq_mhz, time_unix, delay_fits, figures, ...

# 2. argv-style dispatcher (for CLI-string compatibility)
from casm_vis_analysis import run
run("casm-autocorr --obs ... --data-dir ... --format ... --show")
```

Both routes call the same `runners.py` orchestration. CLI commands also call the runners — there is one orchestration code path, used by all three surfaces.

### Data selection: obs-mode vs time-range-mode
Each runner accepts the data either way. Pick one:

| Pass... | Behavior |
|---------|----------|
| `obs="YYYY-MM-DD-HH:MM:SS"` (with optional `data_dir`) | `VisibilityReader` opens that single observation; `time_start`/`time_end`/`nfiles`/`skip_nfiles` trim within it. |
| `obs=None` + `time_start`/`time_end` (with optional `data_dir` or `data_root`) | `casm_io.correlator.read_visibilities` auto-discovers every observation overlapping the range, stitches them, and warns about gaps. `data_root` (default `/mnt`) is scanned for `visibilities_*` subdirs when `data_dir` is omitted. |

CLI: omit `--obs` and pass `--time-start`/`--time-end` (and optionally `--data-root`) to use auto-discovery.

### `run_waterfall` reads only the plotted baselines
`run_waterfall` builds the input set from the layout (or `snaps`) first, then
passes it to `_read_vis` as `inputs=`. casm_io's reader keeps only the
upper-triangle baselines among those inputs, memmapped, so the rest of the
correlator matrix is never read. Output plots are unchanged from before this
change. For a day of data this drops memory use from roughly 130 GB to a few
GB. The returned `vis` is the sub-triangle of `inputs` (indexable with
`triu_flat_index(len(inputs), rank_i, rank_j)`); the dict also carries
`inputs` and `nsig` describing it. `run_autocorr` and `run_fringe_stop` are
unaffected — they still read the full baseline set. Covered by
`tests/test_waterfall_subset.py` (rank-remap selects the same cells as the
full array, and the subset is smaller).

### Layout sync workflow

Two-stage pipeline under the hood. CAsMan -> `casm_wiring.csv` -> consumer layout CSV consumed by `AntennaMapping.load`. Geographic coordinates come from the surveyed positions CSV (`antenna_layout_april_ovro.csv`); CAsMan decides which antenna occupies which grid cell (row/col) and how each feed is wired. A feed is keyed by `(snap, adc)` — that's the level diffs are reported at.

`casm-layout` (`layout/cli.py`, registered in `_COMMANDS` in `__init__.py` so `run("casm-layout ...")` works) is the primary CLI — three verbs, all of which pull the latest CAsMan DB snapshot from GitHub first (checksum-skip if unchanged; `--offline` skips the network and falls back to the local copy with a loud warning):

```
casm-layout status   # pull CAsMan, one-line diff summary (read-only)
casm-layout diff     # pull CAsMan, full position-level + wiring-row diff (read-only)
casm-layout apply    # show diff, confirm ("Apply these changes? [y/N]", skip with -y/--yes),
                      # then rewrite casm_wiring.csv (.bak backup) + rebuild the dated
                      # casm_antenna_layout_YYYY-MM-DD.csv, updating the `current` symlink
```

Shared flags: `--offline`, `--positions`, `--overrides`, `--snap-map`, `--wiring`, `--layout-dir`. `apply` also takes `-y`/`--yes` and `--force` (bypass the <5-antenna sanity guard). There is no `--pull` and no `--dry-run` on `casm-layout` — `status`/`diff` *are* the dry-run.

Python API: `layout.casman_pull.pull_casman`, `layout.sync.build_wiring_candidate`, `layout.build.build_layout_dataframe`, `layout.diff.{resolve_current_layout, diff_layouts, summarize_diff}`.

Legacy two-stage commands still work standalone (each prints a tip pointing at `casm-layout`):

Python:
```python
from casm_vis_analysis import run_sync_wiring, run_build_layout
run_sync_wiring(dry_run=True)            # show diff vs current wiring CSV
run_sync_wiring(dry_run=False)           # apply (backup .bak, write new)
run_build_layout(check_casman=True)      # rebuild consumer CSV + diff
```

CLI:
```
casm-sync-wiring                  # dry-run
casm-sync-wiring --apply
casm-build-layout --check-casman
```

Conflict policy: **CAsMan wins**. Hand-edits to `casm_wiring.csv` do not survive `apply` — encode them as override rows in `casm_wiring_overrides.csv` (`add` / `disable` / `replace`, keyed by `(chassis, slot, adc)`). Default location: `/home/casm/software/dev/antenna_layouts/casm_wiring_overrides.csv`. Sync aborts if CAsMan returns < 5 chassis-1 P1 antennas (pass `--force` to override).

### CLI / Runner Recipe (for new commands)
1. Add `run_<name>(*, ...)` to `runners.py` — typed kwargs, returns dict with arrays + figures, honours `show=True`/`False`.
2. Add `_main(argv=None)` to `cli.py`: `_common_parser()` + extra args, then forward to `run_<name>(**args_dict)`.
3. Add entry point in `pyproject.toml` under `[project.scripts]`.
4. Register in `_COMMANDS` dict in `__init__.py` so `run()` can dispatch it.
5. Export `run_<name>` from `__init__.py`.
6. When `show=True`: skip `make_output_dir` (unless needed for non-plot output like NPZ), pass `None` for all plot output paths, call `plt.show()` at the end.
7. Plotting functions must return figures without closing when output path is `None`.

### `casm-fringe-stop` Extra Flags
`--ref-ant`, `--source`, `--sign`, `--min-alt`, `--save-npz`, `--rfi-mask`, `--delay-model`

## casm_io API Reference (v0.2.0+)
- `VisibilityReader(data_dir, base_str, fmt)` — class-based reader
  - `.read(ref=, targets=, freq_order=, time_start=, time_end=, verbose=)` → VisibilityResult (dict-style: `vis`, `freq_mhz`, `time_unix`)
- `load_format(name_or_path)` → VisibilityFormat (`nsig`, `nchan`, `dt_raw_s`, etc.)
  - Built-in names: `layout_32ant`, `layout_64ant`
- `AntennaMapping.load(csv)` → `.packet_index(aid)`, `.snap_adc(aid)`, `.get_positions()`, `.format_antenna(aid)`, `.active_antennas()`
- `triu_flat_index(n, i, j)`, `triu_to_ij(n, flat_idx)` — baseline indexing
- Constants: `OVRO_LAT_DEG`, `OVRO_LON_DEG`, `OVRO_ELEV_M`, `C_LIGHT_M_S`

## Key Function Signatures

### sources.py
- `source_position(name, time_unix)` → SkyCoord
- `source_altaz(name, time_unix)` → (alt_deg, az_deg)
- `source_enu(name, time_unix)` → ndarray (T, 3)
- `find_transit_window(name, time_unix, min_alt_deg=10.0)` → (i_start, i_end)

### fringe_stop.py
- `compute_baselines_enu(positions_enu, ref_idx, target_idxs)` → ndarray (n_bl, 3)
- `geometric_delay(source_enu, baseline_enu)` → ndarray (T,) or (T, n_bl)
- `fringe_stop(vis, freq_mhz, tau_s, sign=-1)` → dict with keys: `vis_raw`, `vis_stopped`, `vis_for_calibration`, `geometric_phase`, `tau_s`, `sign`, `freq_mhz`

### delay.py
- `fit_delay(vis_fs, freq_mhz, time_mask=None, model="linear")` → dict (model params)
- `apply_delay(vis, freq_mhz, fit_params, model="linear")` → ndarray
- `build_delay_design_matrix(n_ant, baseline_pairs)` → ndarray (n_bl, n_ant)
- `solve_antenna_delays(baseline_delays, design_matrix, weights=None, ref_ant_idx=0)` → ndarray (n_ant,)
- `DELAY_MODELS` — registry dict with `"linear"` and `"per_freq_phasor"` keys

### output.py
- `make_output_dir(base_dir, obs_id)` → dict with keys: `base`, `autocorr`, `waterfall`, `fringe_stop`
- `save_results(path, **arrays)` → None (saves NPZ)

### plotting
- `plot_autocorr(vis, freq_mhz, antenna_labels, time_avg=True, freq_mask=None, output_path=None, ncols=4)` → Figure
- `plot_waterfall(vis, freq_mhz, time_unix, nsig, packet_indices, antenna_labels, snap_adc_labels, split_max=16, output_dir=None)` → list[Figure]
- `plot_fringe_diagnostic(panels, time_unix, freq_mhz, target_labels, target_snaps, ref_snap, output_dir=None)` → list[Figure]
- `plot_phase_vs_freq(panels, freq_mhz, baseline_labels=None, unwrap=True, output_path=None)` → Figure

## Repo Boundary
- **This repo**: loading, plotting, fringe-stop, delay correction (diagnostic)
- **casm_calibration** (future): SVD beamforming — consumes `vis_for_calibration` from fringe_stop

## Data Shapes Convention
- Visibilities: `(T, F, n_bl)` complex64
- Frequencies: `(F,)` float64 in MHz
- Time: `(T,)` float64 Unix timestamps
- Source ENU: `(T, 3)` float64
- Baselines ENU: `(n_bl, 3)` float64

## Testing
```bash
pytest tests/ -v
```
35 tests, all passing. Tests use synthetic fixtures — no real data needed.
