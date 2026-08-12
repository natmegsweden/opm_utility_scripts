#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analog channel mapping for OPM-MEG recordings.

Generates a mapping from raw hardware channel names (ai1, ai2, …) to
human-readable names with correct MNE channel kinds and units.

Run as a script to regenerate ``analog_channel_mapping.json`` in the same
directory:

    python -m opm_utility_scripts.analog.mapping
"""

import json
from pathlib import Path

from mne._fiff.constants import FIFF

# JSON file lives alongside this source file.
_MAPPING_FILE = Path(__file__).parent / 'analog_channel_mapping.json'


def generate_analog_channel_mapping():
    """
    Return the analog channel mapping dict.

    The dict maps hardware channel names (e.g. ``"ai1"``) to sub-dicts with
    keys ``"newname"`` and ``"type"``.  ``"type"`` is itself a dict with
    ``"kind"`` and ``"unit"`` (integer FIFF constants).

    Returns
    -------
    dict
        Mapping from old channel name to ``{"newname": str, "type": {...}}``.
    """
    eog = dict(kind=FIFF.FIFFV_EOG_CH, unit=FIFF.FIFF_UNIT_V)
    emg = dict(kind=FIFF.FIFFV_EMG_CH, unit=FIFF.FIFF_UNIT_V)
    ecg = dict(kind=FIFF.FIFFV_ECG_CH, unit=FIFF.FIFF_UNIT_V)
    resp = dict(kind=FIFF.FIFFV_RESP_CH, unit=FIFF.FIFF_UNIT_V)
    bio = dict(kind=FIFF.FIFFV_BIO_CH, unit=FIFF.FIFF_UNIT_V)
    misc = dict(kind=FIFF.FIFFV_MISC_CH, unit=FIFF.FIFF_UNIT_V)

    mapping = {
        "ai1":  {"newname": "Acc2X",  "type": misc},   # Card 1
        "ai2":  {"newname": "Acc2Y",  "type": misc},
        "ai3":  {"newname": "Acc2Z",  "type": misc},    # typo "tptypeye" fixed
        # "ai4":  {"newname": "-",     "type": misc},
        "ai5":  {"newname": "ECG",    "type": ecg},     # Card 2
        "ai6":  {"newname": "EOG1",   "type": eog},
        "ai7":  {"newname": "EOG2",   "type": eog},
        "ai8":  {"newname": "RESP",   "type": resp},
        "ai9":  {"newname": "Acc1X",  "type": misc},    # Card 3
        "ai10": {"newname": "Acc1Y",  "type": misc},
        "ai11": {"newname": "Acc1Z",  "type": misc},
        # "ai12": {"newname": "-",     "type": misc},
        # "ai13": {"newname": "-",     "type": misc},   # Card 4
        # "ai14": {"newname": "-",     "type": misc},
        "ai15": {"newname": "EyeLX",  "type": misc},
        "ai16": {"newname": "EyeLY",  "type": misc},
        "ai17": {"newname": "EyeLP",  "type": misc},    # Card 5
        "ai18": {"newname": "EyeRX",  "type": misc},
        "ai19": {"newname": "EyeRY",  "type": misc},
        "ai20": {"newname": "EyeRP",  "type": misc},
        # "ai21": {"newname": "-",     "type": misc},   # Card 6
        # "ai22": {"newname": "-",     "type": misc},
        # "ai23": {"newname": "-",     "type": misc},
        # "ai24": {"newname": "-",     "type": misc},
    }
    return mapping


if __name__ == "__main__":
    mapping = generate_analog_channel_mapping()
    with open(_MAPPING_FILE, "w") as f:
        json.dump(mapping, f, indent=4)
    print(f"Wrote {_MAPPING_FILE}")
