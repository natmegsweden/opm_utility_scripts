# opm_utility_scripts

Utility package for Fieldline OPM-MEG data processing.

## Package layout

```
opm_utility_scripts/
├── __init__.py                       Public API
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

## Scripts

### `hpi/coregister.py`

Unified replacement for the old `add_hpi_CP.py` / `add_hpi_multi_CP.py` pair.
Select one or more data files; the transform is fitted once and applied to all
selected files.  Output suffix: `_proc-hpi+ds_raw.fif`.

```bash
python -m opm_utility_scripts.hpi.coregister
```

### `hpi/check.py`

Checks an HPI recording by fitting magnetic dipoles to the detected coil
fields.  Shows goodness-of-fit values and 3-D sensor/coil plots.

```bash
python -m opm_utility_scripts.hpi.check
```

### `tools/check_events.py`

Inspect trigger timing in a FIF file.

```bash
python -m opm_utility_scripts.tools.check_events --file my_raw.fif --stim di38
```

### `tools/helmetscan_2d.py`

2-D layout showing which sensor is in which helmet slot.

```bash
python -m opm_utility_scripts.tools.helmetscan_2d
```

### `analog/mapping.py`

Regenerate `analog/analog_channel_mapping.json`:

```bash
python -m opm_utility_scripts.analog.mapping
```

### `analog/rename.py`

Rename analog channels in a FIF file:

```bash
python -m opm_utility_scripts.analog.rename --file my_raw.fif --newfile my_renamed_raw.fif
```

## Known pre-existing issues (preserved, not changed)

- `hpi/check.py`: uses `raw.info` (not `epochs.info`) for the channel lookup
  inside `main()`.  Behaviour is preserved from the original script.
- `hpi/_core.py`: `coil_amplitudes` is only assigned inside the `n_hpis >= 3`
  branch of the per-coil loop.  If no coil ever meets that condition,
  `UnboundLocalError` is raised at the assertion after the loop.  This
  pre-existing bug is preserved with an explanatory comment.
