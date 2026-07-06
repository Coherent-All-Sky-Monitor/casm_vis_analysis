# Walkthrough Notebooks

Two end-to-end tutorial notebooks live in `notebooks/` with their outputs committed as the verified reference. Start here before reaching for individual APIs.

## `notebooks/casm_calibration_beamforming_walkthrough.ipynb`

Covers the complete SVD calibration and beamformer validation pipeline:

- Loading visibilities with `read_visibilities` and the time-range auto-discovery mode
- Applying an RFI mask
- Running off-source static-vis subtraction (`build_static_visibility`, `subtract_static_visibility`)
- Fringe-stopping toward the Sun (`fringe_stop`, `sign=-1`)
- Running SVD calibration (via `casm_calibration`)
- Calling `beam_power_vs_time` to verify the cal phases the array up toward multiple sources
- Validating a deployed int8 weights file with `validate_source`

## `notebooks/casm_pulsar_search_walkthrough.ipynb`

Covers the beamformed visibilities to filterbank to fold pipeline:

- Reading a beam dump from `/mnt/nvme4/data/casm/beam_dumps/`
- Converting to SIGPROC filterbank with `casm_combine_time_contiguous_multi_beam_dump_to_sigproc.py`
- Folding with `prepfold` inside the Presto Apptainer container
- Interpreting the fold output and verifying against a known pulsar ephemeris

## Canonical notebook location

Both notebooks are maintained at:

```
/home/casm/software/dev/casm_refactor_notebooks/notebooks/
```

The copies in `notebooks/` here are symlinked or duplicated for standalone reference. When updating notebooks, edit the canonical location.

## Running the notebooks

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_refactor_notebooks/notebooks
jupyter lab
```

The notebooks expect real data on `/mnt`. They are not reproducible in a test environment without data. For unit testing of individual functions, use the synthetic fixtures in `tests/`.
