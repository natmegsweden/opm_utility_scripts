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

## opmutil CLI

`opmutil` is the single executable for all HPI and utility scripts.

```
usage: opmutil [-h] [--version] {check,coregister} ...
```

### `opmutil coregister`

Unified replacement for the old `add_hpi_CP.py` / `add_hpi_multi_CP.py` pair.
Select one or more data files; the transform is fitted once and applied to all
selected files. Output suffix: `_proc-hpi_raw.fif`, or
`_proc-hpi+ds_raw.fif` when resampling changes the sampling frequency.

Omitted required inputs are requested through command-line prompts, so the
command works non-interactively, interactively, or anywhere in between.

```bash
# Fully interactive (command-line prompts)
opmutil coregister

# Fully non-interactive
opmutil coregister \
    --data AudOdd_raw.fif RSEO_raw.fif \
    --hpi  HPIBefore_raw.fif \
    --pol  digitisation.json \
    --freq 33 --sfreq 1000 \
    --save --overwrite --plot

# With noise-reference bad-channel detection and a custom GOF threshold
opmutil coregister \
    --data AudOdd_raw.fif \
    --hpi  HPIBefore_raw.fif \
    --pol  digitisation.json \
    --reffile RestingState_raw.fif \
    --freq 33 --sfreq 1000 --gof 0.9 \
    --save

# Uncentred matching only (not the complete historical legacy engine)
opmutil coregister \
    --data AudOdd_raw.fif \
    --hpi  HPIBefore_raw.fif \
    --pol  digitisation.json \
    --no-center-matching \
    --save

# Without rigid-refinement optimization
opmutil coregister \
    --data AudOdd_raw.fif \
    --hpi  HPIBefore_raw.fif \
    --pol  digitisation.json \
    --optimization none \
    --save
```

| Flag | Description | Default |
|------|-------------|---------|
| `--data` | One or more data files to apply the transform to | *(ask)* |
| `--hpi` | Raw HPI recording | *(ask)* |
| `--pol` | Polhemus digitisation file (`.json` or `.fif`) | *(ask)* |
| `--reffile` | Optional reference recording (e.g. resting state) used for background-power-based noisy channel detection | `auto` tries a clean HPI tail if omitted |
| `--freq` | HPI drive frequency in Hz | *(ask)* |
| `--gof` | Minimum dipole GOF for a coil to be included in the device-to-head transform fit | `0.95` |
| `--no-center-matching` | Compatibility alias for `--matching-strategy coordinate_nearest`; changes matching only, independently of refinement | centred |
| `--optimization` | Post-registration refinement: `none` or `rigid_gof` (bounded L-BFGS-B maximizing field-fit GOF); `rigid` is an alias for `rigid_gof` | `rigid_gof` |
| `--sfreq` | Target sampling frequency in Hz | *(ask)* |
| `--save` / `--overwrite` / `--plot` | Save outputs, overwrite existing FIF files, show alignment plot (not automatically saved) | off |

The following behavior controls are shared by `coregister` and `check`:

| Flag | Description | Default |
|------|-------------|---------|
| `--bad-channel-policy {auto,reference,none}` | `auto`: use the reference or a clean final five-second HPI tail, warning and skipping detection if unavailable; `reference`: require `--reffile`; `none`: skip noise detection. Explicit bads and invalid geometry are still excluded | `auto` |
| `--activation-window-s` | Actual amplitude-fit duration in seconds, centred on the detected activation midpoint; insufficient activation or out-of-bounds windows raise an error | `2.0` |
| `--gof-comparison {inclusive,strict}` | Include coils with GOF `>= --gof` or `> --gof`, respectively | `inclusive` |
| `--matching-strategy {centroid_nearest,coordinate_nearest}` | Nearest-target matching after centring each point cloud, or on raw coordinates; mutually exclusive with `--no-center-matching` | `centroid_nearest` |
| `--allow-repeated-matches` | Permit repeated nearest targets; degenerate rigid fits still raise an error | off (unique targets required) |
| `--settings-json [DIR]` | `coregister` writes one sidecar per saved data output beside that output by default; this option is only needed to choose an optional existing sidecar directory. `check` writes its fit-stage record only when requested | enabled for `coregister`; disabled for `check` |

Centring removes bulk translation offsets, not arbitrary inter-frame rotations.
Matching and refinement are independent controls. There is no branch-selector
option: the historical approach names below are comparisons, not CLI presets.

### Settings sidecars

With `coregister --save`, a settings sidecar is written **per
individual transformed data output** by default, next to that output, not once per HPI
recording or batch. Its name is
`hpi_<output-basename-with-.fif-replaced-by-.json>`. For example:

| Transformed output | Settings sidecar in the same directory |
|--------------------|------------------------------------------------|
| `subject_proc-hpi_raw.fif` | `hpi_subject_proc-hpi_raw.json` |
| `subject_proc-hpi+ds_raw.fif` | `hpi_subject_proc-hpi+ds_raw.json` |

```bash
opmutil coregister \
    --data subject_raw.fif resting_raw.fif \
    --hpi HPIBefore_raw.fif --pol digitisation.json \
    --freq 33 --sfreq 1000 --save
```

Each output has its own sidecar even though both share the fitted transform.
An optional `--settings-json /existing/sidecar-directory` changes the sidecar
directory, not the per-output naming convention. Sidecars record input/output
basenames, library versions, effective fit settings, a compact results summary,
and `fit_results`: the device-to-head transform (translation in metres),
optimizer method/status, and one entry per HPI coil with its dipole GOF and
inclusion decision. Included coils also record their matched Polhemus point
(zero-based index), post-fit residual in metres, and Polhemus-position GOF.
Non-finite diagnostic values are represented as JSON `null`. Sidecars are
provenance records, not JSON configuration files read by the CLI.

For the API, fit once and write a sidecar after saving **each** output:

```python
from pathlib import Path
import mne
from opm_utility_scripts.hpi import fit_hpi, apply_transform, save_raw
from opm_utility_scripts.hpi._core import write_settings_json

hpifile, polfile = "HPIBefore_raw.fif", "digitisation.json"
fit = fit_hpi(hpifile, polfile, 33, optim="rigid_gof")
target_sfreq = 1000
for datafile in ("subject_raw.fif", "resting_raw.fif"):
    source_sfreq = mne.io.read_info(datafile, verbose="error")["sfreq"]
    suffix = "_proc-hpi"
    if int(target_sfreq) != int(source_sfreq):
        suffix += "+ds"
    raw = apply_transform(datafile, fit, new_sfreq=target_sfreq)
    output = Path(save_raw(raw, datafile, suffix + "_raw.fif", overwrite=True))
    sidecar = output.with_name("hpi_" + output.with_suffix(".json").name)
    write_settings_json(
        sidecar, fit, hpifile=hpifile, polfile=polfile,
        datafile=datafile, output_file=output,
    )
```

The lower-level `fit_hpi(..., settings_json="fit.json")` and
`fit_hpi_amplitudes(..., settings_json="amplitudes.json")` instead write an
explicit fit-stage record: they do not apply or save data outputs. Use the
per-output loop above for transformed-data provenance. `check` also has no
transformed data outputs; its settings record describes the check/fit stage.

### Version-specific command examples

The following commands show current CLI settings corresponding as closely as
possible to each historical release. They are **not version selectors** and do
not reproduce every historical implementation detail; all use the current
sequential fitter. Replace paths and frequency with values for your recording.

**v0.3.0**:

```bash
opmutil coregister \
    --data subject_raw.fif \
    --hpi HPIBefore_raw.fif --pol digitisation.json \
    --freq 33 --bad-channel-policy auto \
    --activation-window-s 2 --gof 0.95 --gof-comparison inclusive \
    --matching-strategy centroid_nearest \
    --optimization rigid_gof --save
```

This is the closest match to current defaults: consistent OPM magnetometer ordering,
unique matches, and bounded field-GOF refinement. When saving transformed FIFs,
coregister writes a provenance sidecar for each output by default; use
`--settings-json /existing/sidecar-directory` to change the sidecar directory.

**v0.2.0**:

```bash
opmutil coregister \
    --data subject_raw.fif \
    --hpi HPIBefore_raw.fif --pol digitisation.json \
    --freq 33 --bad-channel-policy auto \
    --activation-window-s 2 --gof 0.95 --gof-comparison inclusive \
    --matching-strategy centroid_nearest --allow-repeated-matches \
    --optimization rigid_gof --save
```

The historical v0.2.0 implementation used noise exclusions after amplitude
fitting, a six-second crop whose first two seconds were fitted, and asymmetric
optimizer bounds. Those details cannot be selected by the current CLI; this
command is only the closest available mapping.

**v0.1.0**:

```bash
opmutil coregister \
    --data subject_raw.fif \
    --hpi HPIBefore_raw.fif --pol digitisation.json \
    --freq 33 --bad-channel-policy none \
    --activation-window-s 2 --gof 0.9 --gof-comparison strict \
    --matching-strategy coordinate_nearest --allow-repeated-matches \
    --optimization none --save
```

This does not select the historical duplicate-frequency amplitude model,
integer-frequency peak spacing, or legacy input parsing. The current sequential
fitter and its non-degeneracy checks remain in effect. To test whether the
localization search grid explains a result difference, add
`--localization-grid legacy`; this selects the stock-MNE initial-guess grid
without changing the rest of the current pipeline.

The `--localization-grid` option defaults to `fine` (2 mm spacing). Its
`legacy` value uses the stock MNE search grid: 10 mm spacing, 5 mm inset, and a
sphere radius based on the minimum MEG coil integration-point radius. This is
an isolated grid comparison, not full legacy parity; the local optimizer,
amplitude fitting, and other current checks remain unchanged. The exact stock
grid is also MNE-version-dependent. `tests/test_hpi_versions.py` accepts the
same option to apply either grid consistently across its v0.1.0/v0.2.0/v0.3.0
settings runs.

The comparison matrix summarizes the historical behavior versus current
controls. All approaches in the historical comparison used sequential coil
fitting.

| Feature | `v0.3.0` | `v0.2.0` | `v0.1.0` |
|---------|---------|---------|---------|
| Noise handling | Reference/clean-tail exclusions before amplitude estimation | Reference/clean-tail detection after amplitude estimation | No reference-noise detection |
| Amplitude window | Actual centred two-second fit | Six-second crop around midpoint, but fit uses its first two seconds | Approximately centred two-second crop |
| Sensor handling | OPM magnetometers kept in consistent channel order across processing stages | Magnetometer-sized slope matrix without consistently enforced downstream ordering | Device-frame magnetometers only |
| Default inclusion | GOF `>= 0.95` | GOF `>= 0.95` | GOF `> 0.9`; explicitly supplied threshold uses `>=` |
| Matching | Centroid-nearest; duplicate targets rejected | Centroid-nearest; no duplicate-target rejection | Uncentred nearest targets |
| Refinement | Field-GOF rigid refinement, rotation bounds ±5° | Field-GOF rigid refinement, rotation bounds −5° to +10° | Closed-form rigid registration only |
| Refined diagnostics | Residuals and Polhemus GOFs recomputed after successful refinement; failed optimization retains initial transform | Returned residuals can remain initial-fit values | Fixed-position Polhemus GOF unavailable |

**v0.2.0 versus v0.3.0 nuance:** the current default controls are closest to
`v0.3.0`, not a byte-for-byte reproduction of `v0.2.0`. Merely changing
the GOF threshold or matching strategy cannot restore v0.2.0's noise-detection
timing, offset fit window, asymmetric optimizer bounds or stale residuals.
The geometric residual may increase after field-GOF refinement because that
objective does not minimize point-to-point distance. Different displayed
residuals can therefore reflect initial versus refined transforms, while
window/channel changes can also alter GOFs.

This pipeline is for OPM recordings, whose sensors are magnetometers, so there
is no sensor-type option. The pipeline selects magnetometer channels and keeps
them in the same order throughout amplitude fitting, localization, and
transform fitting. The SQUID device sometimes used to acquire digitisation
points does not affect OPM sensor selection; only its digitisation points are
used by coregistration.

The following distinct JSON snippets are **partial `settings` fragments** in
the current sidecar vocabulary. They omit data-dependent channel lists and
resolved noise policy. They are not complete sidecars or loadable presets.

**`v0.3.0`-like mapping** (closest to current defaults):

```json
{
  "sensor_selection": "opm_magnetometers",
  "bad_channel_policy": {"requested": "auto"},
  "activation_window": {"duration_s": 2.0},
  "coil_inclusion": {"gof_limit": 0.95, "comparison": "inclusive"},
  "matching": {"strategy": "centroid_nearest", "unique_matches": true},
  "transform_refinement": {
    "method": "rigid_gof",
    "rotation_bound_deg": 5.0,
    "translation_bound_mm": 5.0
  }
}
```

**`v0.2.0`-like mapping** (two-second duration only approximates its historical
offset fit; current noise exclusions still occur before fitting and current
refinement bounds remain ±5°):

```json
{
  "sensor_selection": "opm_magnetometers",
  "bad_channel_policy": {"requested": "auto"},
  "activation_window": {"duration_s": 2.0},
  "coil_inclusion": {"gof_limit": 0.95, "comparison": "inclusive"},
  "matching": {"strategy": "centroid_nearest", "unique_matches": false},
  "transform_refinement": {"method": "rigid_gof"}
}
```

**`v0.1.0`-like mapping** (default strict threshold and no refinement):

```json
{
  "sensor_selection": "opm_magnetometers",
  "bad_channel_policy": {"requested": "none"},
  "activation_window": {"duration_s": 2.0},
  "coil_inclusion": {"gof_limit": 0.9, "comparison": "strict"},
  "matching": {"strategy": "coordinate_nearest", "unique_matches": false},
  "transform_refinement": {"method": "none"}
}
```

This does not select the historical duplicate-frequency amplitude model or
integer-frequency peak spacing. The stock MNE localizer can be selected
separately with `--localization-grid legacy`; it does not suppress the current
fixed-position Polhemus GOF diagnostic. The historical v0.1.0 implementation
accepted verified head-frame FIF/DigMontage digitisation, not JSON. Its
configurable fitting sample rate is also not reproduced: the current HPI fit
uses 1000 Hz internally; `coregister --sfreq` controls transformed-data
resampling, not HPI fitting. Allowing repeated matches does not disable current
non-degeneracy checks.

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

# With noise-reference bad-channel detection and rigid-refinement optimization
opmutil check --hpi HPIbefore_raw.fif --pol digitisation.json \
    --reffile RestingState_raw.fif --optimization rigid

# Uncentred matching only (not the complete historical legacy engine)
opmutil check --hpi HPIbefore_raw.fif --pol digitisation.json \
    --no-center-matching
```

| Flag | Description | Default |
|------|-------------|---------|
| `--hpi` | Path to the raw HPI `.fif` file | *(ask)* |
| `--pol` | Path to Polhemus file (`.fif` or `.json`) | *(ask)* |
| `--freq` | HPI drive frequency in Hz | `33` |
| `--reffile` | Optional reference recording used for background-power-based noisy channel detection | `auto` tries a clean HPI tail if omitted |
| `--gof` | Minimum dipole GOF for a coil to be included in the device-to-head transform fit | `0.95` |
| `--detailed` | Show full diagnostics in `--hpi` + `--pol` mode | off |
| `--optimization` | `none` or `rigid_gof`; `rigid` is a compatibility alias. Applies in full coregistration mode | `rigid_gof` |
| `--no-center-matching` | Alias for `--matching-strategy coordinate_nearest`, independent of refinement; applies in full coregistration mode | centred |

See the shared behavior controls above for noise policy, sensor population,
activation window, GOF comparison, matching and settings output.

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

The `channels`, `viz`, `io`, and `analog.*` symbols below are also re-exported
(lazily) directly from the top-level `opm_utility_scripts` package, e.g.
`from opm_utility_scripts import plot_hpi_alignment` works. The `hpi.*`
symbols are not re-exported at the top level and must be imported from the
`hpi` subpackage as shown.

```python
from opm_utility_scripts.channels import (
    find_zero_location_channels,   # was TC_findzerochans
    get_hpi_output_channels,       # was TC_get_hpiout_names
    pick_low_noise_meg_chs,
)
from opm_utility_scripts.viz import (
    plot_3d, plot_hpi_alignment, plot_psd, rotate_points, create_aligned_grid,
)
from opm_utility_scripts.io import (
    get_file, get_files, get_input, get_boolean, write_bw_marker_file,
    load_datafile, load_hpifile, load_polhemus, select_best_hpi_file,
)
from opm_utility_scripts.analog.mapping import generate_analog_channel_mapping   # was generate_mapping
from opm_utility_scripts.analog.rename import rename_channels
from opm_utility_scripts.hpi import (
    fit_hpi, fit_hpi_amplitudes, apply_transform, save_raw, compute_fit_diagnostics,
)
```

## Dependencies

| Group | Packages | Install extra |
|-------|----------|---------------|
| Core | `mne>=1.12`, `numpy>=1.26`, `scipy>=1.12`, `matplotlib>=3.8` | *(default)* |
| Tools | `pandas>=2.2` | `[tools]` |
| VTK visualiser | `pandas>=2.2`, `PyQt5>=5.15`, `vtk>=9.3` | `[vtk]` |
