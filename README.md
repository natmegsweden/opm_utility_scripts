# opm_utility_scripts

Utility package for Fieldline OPM-MEG data processing.

## Installation

```bash
# Core install (HPI coregistration + channel utilities)
pip install .

# Include tools that need pandas (helmetscan_2d, check_events)
pip install ".[tools]"

# Include the 3-D VTK visualiser (also needs pandas, PyQt5, vtk)
pip install ".[vtk]"
```

After installation the `opmutil` executable is available on `PATH`.

For local development without installation, `run_hpi.py` continues to work
as before (see [Legacy usage](#legacy-usage)).

## Package layout

```
opm_utility_scripts/
├── __init__.py                       Public API
├── cli.py                            opmutil console-script entry point
├── channels.py                       Channel utility functions
├── viz.py                            Visualisation utilities
├── io.py                             File I/O and dialog helpers
├── analog/
│   ├── __init__.py
│   ├── mapping.py                    Generate analog channel mapping
│   ├── rename.py                     Rename analog channels in FIF files
│   └── analog_channel_mapping.json  Default mapping (ai1–ai20 → labelled channels)
├── hpi/
│   ├── __init__.py
│   ├── _core.py                      Shared HPI pipeline (fit_hpi, apply_transform)
│   ├── coregister.py                 Unified single/multi-file entry point
│   └── check.py                      Dipole-fit HPI quality check
└── tools/
    ├── __init__.py
    ├── check_events.py               Inspect trigger events in FIF files
    ├── helmetscan_2d.py              2-D helmet scan layout plot
    ├── helmetscan_3d_vtk.py          3-D VTK helmet scan visualisation (requires vtk, PyQt5)
    ├── hedscan_layout.mat            HEDscan layout file
    ├── fieldlinebeta2bz_helmet.mat   Fieldline helmet layout file
    └── data_sensors.csv              Sensor slot template for 3-D visualisation
```

## opmutil CLI

`opmutil` is the single executable for all HPI and utility scripts.

```
usage: opmutil [-h] [--version] {check,coregister} ...
```

### `opmutil coregister`

Unified replacement for the old `add_hpi_CP.py` / `add_hpi_multi_CP.py` pair.
Select one or more data files; the transform is fitted once and applied to all
selected files.  Output suffix: `_proc-hpi+ds_raw.fif`.

Any omitted argument opens a GUI file dialog, so the command works fully
non-interactively, fully interactively, or anywhere in between.

```bash
# Fully interactive (GUI dialogs for all inputs)
opmutil coregister

# Fully non-interactive
opmutil coregister \
    --data AudOdd_raw.fif RSEO_raw.fif \
    --hpi  HPIBefore_raw.fif \
    --pol  digitisation.json \
    --freq 33 --sfreq 1000 \
    --save --overwrite --plot
```

### `opmutil check`

Checks an HPI recording by fitting magnetic dipoles to the detected coil
fields.  Four modes depending on which arguments are supplied:

| Arguments | Mode |
|-----------|------|
| `--hpi` only | HPI-only — GOF + device-frame coil positions |
| `--pol` only | Polhemus-only — dig point summary + head-frame distances |
| `--hpi` + `--pol` | Full coregistration — GOF, residuals, recommendations |
| neither | GUI dialogs for both files |

```bash
# HPI-only (no polhemus required)
opmutil check --hpi HPIbefore_raw.fif

# Full coregistration
opmutil check --hpi HPIbefore_raw.fif --pol digitisation.json

# Override drive frequency and GOF threshold; verbose output
opmutil check --hpi HPIbefore_raw.fif --pol digitisation.json \
    --freq 33 --gof 0.95 --detailed
```

## Submodules (python -m)

All submodules remain runnable with `python -m` directly:

```bash
# HPI coregistration
python -m opm_utility_scripts.hpi.coregister

# HPI quality check
python -m opm_utility_scripts.hpi.check --hpi HPIbefore_raw.fif

# Inspect trigger events
python -m opm_utility_scripts.tools.check_events --file my_raw.fif --stim di38

# 2-D helmet layout
python -m opm_utility_scripts.tools.helmetscan_2d

# Regenerate analog channel mapping JSON
python -m opm_utility_scripts.analog.mapping

# Rename analog channels in a FIF file
python -m opm_utility_scripts.analog.rename --file my_raw.fif --newfile my_renamed_raw.fif
```

## Legacy usage

`run_hpi.py` is kept as a compatibility wrapper for running from the source
tree without installing:

```bash
python run_hpi.py coregister
python run_hpi.py check --hpi HPIbefore_raw.fif
python run_hpi.py check --hpi HPIbefore_raw.fif --pol digitisation.json
```

It delegates directly to `opmutil`'s `cli.main()` and accepts the same
arguments.

## Public API

```python
from opm_utility_scripts.channels import (
    find_zero_location_channels,   # was TC_findzerochans
    get_hpi_output_channels,       # was TC_get_hpiout_names
    pick_low_noise_meg_chs,
)
from opm_utility_scripts.viz import plot_3d, plot_psd, rotate_points, create_aligned_grid
from opm_utility_scripts.io import get_file, get_files, get_input, get_boolean, write_bw_marker_file
from opm_utility_scripts.analog.mapping import generate_analog_channel_mapping   # was generate_mapping
from opm_utility_scripts.analog.rename import rename_channels
```

## Dependencies

| Group | Packages | Install extra |
|-------|----------|---------------|
| Core | `mne>=1.8`, `numpy>=1.26`, `scipy>=1.12`, `matplotlib>=3.8` | *(default)* |
| Tools | `pandas>=2.2` | `[tools]` |
| VTK visualiser | `pandas>=2.2`, `PyQt5>=5.15`, `vtk>=9.3` | `[vtk]` |

## Known pre-existing issues (preserved, not changed)

- `hpi/check.py`: uses `raw.info` (not `epochs.info`) for the channel lookup
  inside `main()`.  Behaviour is preserved from the original script.
