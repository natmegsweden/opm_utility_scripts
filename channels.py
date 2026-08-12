"""Channel utility functions for OPM-MEG data."""

import mne
import numpy as np
import scipy.signal as sig
from mne._fiff.pick import pick_types


def find_zero_location_channels(info, tolerance=0.02):
    """
    Identify MEG channels with zero or invalid locations.

    Finds magnetometer channels positioned at the origin (0,0,0) which
    typically indicates faulty sensor positioning or missing location data.

    Args:
        info (mne.Info): MNE info object containing channel information
        tolerance (float): Distance tolerance in metres (default: 0.02 m = 2 cm)

    Returns:
        numpy.ndarray: Array of channel names with zero/invalid locations

    Note:
        Default tolerance of 2 cm removes channels within a sphere of the origin.
    """
    bads_fl = np.array([])
    picks = pick_types(info, meg='mag')
    lst = list(bads_fl)
    for j in picks:
        ch = info['chs'][j]
        if np.isclose(sum(ch['loc'][0:3]), 0.0, atol=1e-3).all():
            lst.append(ch['ch_name'])
    bads_fl = np.asarray(lst)
    return bads_fl


def get_hpi_output_channels(raw):
    """
    Extract HPI output channel names and indices from raw data.

    Identifies miscellaneous channels containing 'out' in their names,
    which typically correspond to HPI coil output signals.  Channels are
    included only when their variance exceeds 1e-25 (i.e. they contain an
    actual signal rather than a flat line).

    Args:
        raw (mne.io.Raw): Raw data containing HPI output channels

    Returns:
        tuple: (hpi_names, hpi_indices)
            - hpi_names (list): Channel names containing HPI outputs
            - hpi_indices (numpy.ndarray): Corresponding channel indices
    """
    hpi_names = list()

    hpi_raw = raw.compute_psd(picks="misc", verbose='error')

    for name in hpi_raw.info['ch_names']:
        if 'out' in name:
            if raw.copy().pick([name])._data.var() > 1e-25:
                hpi_names += [name]

    hpi_indices = np.zeros(len(hpi_names), dtype=np.int64)
    i = 0
    j = 0
    for ch in raw.info['ch_names']:
        for hpi in hpi_names:
            if hpi in ch:
                hpi_indices[j] = i
                j = j + 1
        i = i + 1

    return hpi_names, hpi_indices


def pick_low_noise_meg_chs(raw, n_std=2, fmax=150):
    """
    Return a list of MEG channel names whose noise exceeds n_std standard
    deviations above the mean noise floor.

    Adapted from the Fieldline HPI script.

    Args:
        raw (mne.io.Raw): Raw data object.
        n_std (int): Number of standard deviations above which a channel is
            considered noisy (default: 2).
        fmax (float): Upper frequency limit for noise estimation (default: 150 Hz).

    Returns:
        list[str]: Names of channels classified as noisy.
    """
    raw_copy = raw.copy()
    raw_copy.pick_types(meg=True)
    data = raw_copy.get_data()
    fs = raw_copy.info['sfreq']

    f, Pxx = sig.welch(data, fs=fs, nperseg=fs // 2, average='median')
    Axx = np.sqrt(Pxx)

    fidx = np.where(f < fmax)[0]
    noise = np.min(Axx[:, fidx], axis=1)
    noise_u = np.mean(noise)
    noise_s = np.std(noise)

    quiet_idx = np.where(noise < noise_u + n_std * noise_s)[0]
    noisy_chs = [n for i, n in enumerate(raw_copy.info['ch_names']) if i not in quiet_idx]

    if len(noisy_chs) > 0:
        print(f'Discarding {len(noisy_chs)} noisy channels: {" ".join(noisy_chs)}')

    return noisy_chs
