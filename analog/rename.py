#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rename analog channels in an OPM-MEG FIF file.

Applies a channel-name/kind/unit mapping (dict or JSON file) to a raw
recording and optionally saves the result.

Usage (command line)::

    python -m opm_utility_scripts.analog.rename \\
        --file my_raw.fif \\
        --newfile my_renamed_raw.fif
"""

import json
from os.path import isfile
from pathlib import Path
from typing import Union

import mne

# Default mapping file lives alongside this source file.
_DEFAULT_MAPPING = Path(__file__).parent / 'analog_channel_mapping.json'


def rename_channels(fif: Union[str, mne.io.BaseRaw], mapping, newpath=None):
    """
    Rename analog channels in a raw FIF recording.

    Parameters
    ----------
    fif : str or mne.io.BaseRaw
        Path to the raw FIF file, or an already-loaded Raw object.
    mapping : str or dict
        Path to a JSON mapping file, or a dict returned by
        :func:`generate_analog_channel_mapping`.
    newpath : str, optional
        Save path for the modified recording.  If *None* the result is
        returned but not saved.

    Returns
    -------
    mne.io.Raw
        Raw object with renamed channels.

    Notes
    -----
    Channel renaming modifies ``raw.info`` in-place before saving.  If a key
    in *mapping* does not have a ``"type"`` sub-key (e.g. due to the
    historical ``"tptypeye"`` typo), that entry is silently skipped for
    kind/unit assignment.  Fix the mapping file to correct this.
    """
    # --- Load raw ---
    if isinstance(fif, str):
        raw = mne.io.read_raw(fif, preload=False, verbose='error')
    elif isinstance(fif, mne.io.BaseRaw):
        raw = fif.copy()
    else:
        raise TypeError(f"fif must be str or mne.io.BaseRaw, got {type(fif)}")

    # --- Load mapping ---
    if isinstance(mapping, str):
        with open(mapping) as json_file:
            mapping = json.load(json_file)

    # --- Rename ---
    for oldname, new in mapping.items():
        for chi, ch in enumerate(raw.info["chs"]):
            if ch["ch_name"] == oldname:
                ch["ch_name"] = new["newname"]
                if "type" in new:  # guard: silently skip entries without a "type" key
                    ch["kind"] = new["type"]["kind"]
                    ch["unit"] = new["type"]["unit"]

        for chi, ch_name in enumerate(raw.info["ch_names"]):
            if ch_name == oldname:
                raw.info["ch_names"][chi] = new["newname"]

    raw.load_data()
    if newpath is not None:
        raw.save(newpath, overwrite=True)
        print('done!')
    return raw


def _args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Rename analog channels in a raw FIF file.")
    parser.add_argument('--file', type=str, help="Path to the raw (fif) file")
    parser.add_argument('--newfile', type=str, help="Path to save new raw (fif) file.  Overwrites if omitted.")
    parser.add_argument('--map', type=str, help="Select analog channel mapping file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _args_parser()
    fif_path = args.file if args.file else None
    if not fif_path or not isfile(fif_path):
        print("Invalid or missing file path.  Please provide a valid raw (fif) file.")
    map_path = args.map if args.map else str(_DEFAULT_MAPPING)
    if not args.map:
        print(f"Using default mapping: {map_path}")
    else:
        print(args.map)
    newfile = args.newfile if args.newfile else fif_path
    if not args.newfile:
        print('Overwriting raw file')
    rename_channels(fif_path, map_path, newfile)
