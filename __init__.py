"""
opm_utility_scripts — utility functions for Fieldline OPM-MEG data.

Public API
----------
From ``channels``:
    find_zero_location_channels, get_hpi_output_channels, pick_low_noise_meg_chs

From ``viz``:
    create_aligned_grid, plot_3d, rotate_points, plot_psd

From ``io``:
    get_boolean, get_file, get_files, get_input, write_bw_marker_file,
    load_datafile, load_polhemus, load_hpifile, select_best_hpi_file

From ``analog.mapping``:
    generate_analog_channel_mapping

From ``analog.rename``:
    rename_channels
"""

from .channels import find_zero_location_channels, get_hpi_output_channels, pick_low_noise_meg_chs
from .viz import create_aligned_grid, plot_3d, rotate_points, plot_psd
from .io import (
    get_boolean,
    get_file,
    get_files,
    get_input,
    load_datafile,
    load_hpifile,
    load_polhemus,
    select_best_hpi_file,
    write_bw_marker_file,
)
from .analog.mapping import generate_analog_channel_mapping
from .analog.rename import rename_channels

__all__ = [
    # channels
    'find_zero_location_channels',
    'get_hpi_output_channels',
    'pick_low_noise_meg_chs',
    # viz
    'create_aligned_grid',
    'plot_3d',
    'rotate_points',
    'plot_psd',
    # io
    'get_boolean',
    'get_file',
    'get_files',
    'get_input',
    'load_datafile',
    'load_hpifile',
    'load_polhemus',
    'select_best_hpi_file',
    'write_bw_marker_file',
    # analog
    'generate_analog_channel_mapping',
    'rename_channels',
]
