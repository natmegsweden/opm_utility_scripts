# Package description

Utility package for Fieldline OPM-MEG data processing.

For a practical walkthrough of HPI fitting, grid selection, and fit inspection,
see the [HPI Recommendations](hpi/hpi-recommendations.md).

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
as before (see [Legacy usage](hpi/hpi.md#legacy-usage)).

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
│   ├── _core.py                      Shared HPI pipeline (fit_hpi, fit_hpi_amplitudes,
│   │                                 apply_transform, save_raw, compute_fit_diagnostics)
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

## Dependencies

| Group | Packages | Install extra |
|-------|----------|---------------|
| Core | `mne>=1.12`, `numpy>=1.26`, `scipy>=1.12`, `matplotlib>=3.8` | *(default)* |
| Tools | `pandas>=2.2` | `[tools]` |
| VTK visualiser | `pandas>=2.2`, `PyQt5>=5.15`, `vtk>=9.3` | `[vtk]` |
