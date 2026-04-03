# casm_vis_analysis

Fringe-stopping, delay correction, and diagnostic plotting for CASM correlator visibilities.

## Install 

```bash
source ~/software/dev/casm_venvs/casm_offline_env/bin/activate
cd /home/casm/software/dev/casm_vis_analysis
pip install -e .
```



## CLI Commands

All commands below use the March 21 sun transit as an example dataset. Common flags shared across commands:

```
--data-dir /mnt/nvme3/data/casm/visibilities_64ant/
--format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json
--obs 2026-03-20-05:55:45
--layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv
```

### casm-viz-data-span

List available observations and their time spans in a data directory.

```bash
casm-viz-data-span \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json
```

### Selecting visibility data

Every command requires an observation ID via `--obs`. Within that observation, you can select which data to analyse in two ways:

**A) By file count:** Use `--nfiles` (number of files to read) and optionally `--skip-nfiles` (files to skip from the start).
```bash
--obs 2026-03-20-05:55:45 --nfiles 2 --skip-nfiles 10
```

**B) By time range:** Use `--time-start`, `--time-end`, and `--time-tz`.
```bash
--obs 2026-03-20-05:55:45 \
--time-start '2026-03-21 12:00:00' --time-end '2026-03-21 13:00:00' --time-tz US/Pacific
```

If neither is specified, all files in the observation are read.

### casm-autocorr

Plot autocorrelation power spectra grouped by SNAP board.

```bash
casm-autocorr \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv \
  --time-start '2026-03-21 12:00:00' --time-end '2026-03-21 13:00:00' --time-tz US/Pacific \
  --output-dir ./output
```

### casm-waterfall

Plot upper-triangle waterfall matrix (diagonal=power, upper=phase).

```bash
casm-waterfall \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv \
  --time-start '2026-03-21 12:00:00' --time-end '2026-03-21 13:00:00' --time-tz US/Pacific \
  --output-dir ./output
```

### casm-fringe-stop

Fringe-stop visibilities toward a source with delay correction diagnostics.

```bash
casm-fringe-stop \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv \
  --ref-ant 3 \
  --source sun \
  --sign -1 \
  --time-start '2026-03-21 10:00:00' --time-end '2026-03-21 15:00:00' --time-tz US/Pacific \
  --output-dir ./output
```

### casm-fit-positions

Fit antenna x/y positions via solar fringe-stopping. Use `--cross-plank` for automatic cross-plank reference selection to avoid intra-plank cross-talk.

```bash
casm-fit-positions \
  --data-dir /mnt/nvme3/data/casm/visibilities_64ant/ \
  --obs 2026-03-20-05:55:45 \
  --format ~/software/dev/casm_io/casm_io/correlator/configs/layout_64ant.json \
  --layout ~/software/dev/antenna_layouts/antenna_layout_mar21.csv \
  --ref-ant 3 \
  --cross-plank \
  --source sun \
  --sign -1 \
  --metric circvar \
  --axis x \
  --x-range='-4,4' --x-step 0.05 \
  --time-start '2026-03-21 10:00:00' --time-end '2026-03-21 15:00:00' --time-tz US/Pacific \
  --output-dir ./output \
  --output-layout corrected_layout.csv
```

## Testing

```bash
pytest tests/ -v
```
