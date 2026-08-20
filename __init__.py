"""
opm_utility_scripts — utility functions for Fieldline OPM-MEG data.

Public API
----------
From ``channels``:
    find_zero_location_channels, get_hpi_output_channels, pick_low_noise_meg_chs

From ``viz``:
    create_aligned_grid, plot_3d, plot_hpi_alignment, rotate_points, plot_psd

From ``io``:
    get_boolean, get_file, get_files, get_input, write_bw_marker_file,
    load_datafile, load_polhemus, load_hpifile, select_best_hpi_file

From ``analog.mapping``:
    generate_analog_channel_mapping

From ``analog.rename``:
    rename_channels
"""

# Mapping of public name → (submodule, attribute) for lazy resolution.
# Importing the package itself is now instant; submodules load on first access.
_LAZY = {
    # channels
    'find_zero_location_channels': ('.channels', 'find_zero_location_channels'),
    'get_hpi_output_channels':     ('.channels', 'get_hpi_output_channels'),
    'pick_low_noise_meg_chs':      ('.channels', 'pick_low_noise_meg_chs'),
    # viz
    'create_aligned_grid':  ('.viz', 'create_aligned_grid'),
    'plot_3d':              ('.viz', 'plot_3d'),
    'plot_hpi_alignment':   ('.viz', 'plot_hpi_alignment'),
    'rotate_points':        ('.viz', 'rotate_points'),
    'plot_psd':             ('.viz', 'plot_psd'),
    # io
    'get_boolean':           ('.io', 'get_boolean'),
    'get_file':              ('.io', 'get_file'),
    'get_files':             ('.io', 'get_files'),
    'get_input':             ('.io', 'get_input'),
    'load_datafile':         ('.io', 'load_datafile'),
    'load_hpifile':          ('.io', 'load_hpifile'),
    'load_polhemus':         ('.io', 'load_polhemus'),
    'select_best_hpi_file':  ('.io', 'select_best_hpi_file'),
    'write_bw_marker_file':  ('.io', 'write_bw_marker_file'),
    # analog
    'generate_analog_channel_mapping': ('.analog.mapping', 'generate_analog_channel_mapping'),
    'rename_channels':                 ('.analog.rename',  'rename_channels'),
    # hpi subpackage
    'hpi': ('.hpi', None),
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    if name not in _LAZY:
        raise AttributeError(f"module 'opm_utility_scripts' has no attribute {name!r}")
    submod, attr = _LAZY[name]
    import importlib
    mod = importlib.import_module(submod, package=__name__)
    value = mod if attr is None else getattr(mod, attr)
    # Cache in module namespace so subsequent accesses are free.
    globals()[name] = value
    return value
