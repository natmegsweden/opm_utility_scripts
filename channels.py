"""Channel utility functions for OPM-MEG data."""

import mne
import numpy as np
import scipy.signal as sig
from mne._fiff.pick import pick_types


def find_zero_location_channels(info, tolerance=None):
    """
    Identify MEG channels with zero or invalid locations.

    Finds magnetometer channels whose position is at the origin (0,0,0) or
    contains NaN / Inf values, either of which causes MNE's external-basis SVD
    (``_setup_ext_proj``) to fail with a ``ValueError``.

    Args:
        info (mne.Info): MNE info object containing channel information
        tolerance (float | None): Distance tolerance in metres.
            **Deprecated** — this parameter is accepted for backward
            compatibility but has no effect. The current implementation
            uses a component-wise threshold of 1 mm on each axis (i.e.
            the position must be within a 1 mm axis-aligned box around
            the origin), not a spherical tolerance. Passing a non-default
            value emits a warning.

    Returns:
        numpy.ndarray: Array of channel names with zero/invalid locations

    Note:
        The threshold is component-wise ``isclose(0.0, atol=1e-3)`` (1 mm
        per axis), not a Euclidean-distance threshold. Earlier documentation
        incorrectly described the unused parameter as a 2 cm sphere.
    """
    if tolerance is not None:
        import warnings
        warnings.warn(
            "The 'tolerance' parameter of find_zero_location_channels is "
            "deprecated and has no effect. The current implementation uses a "
            "fixed per-component threshold of 1 mm on each axis.",
            DeprecationWarning, stacklevel=2,
        )
    picks = pick_types(info, meg='mag')
    lst = []
    for j in picks:
        ch = info['chs'][j]
        loc = ch['loc'][:3]
        if (
            not np.all(np.isfinite(loc))
            or np.all(np.isclose(loc, 0.0, atol=1e-3))
        ):
            lst.append(ch['ch_name'])
    return np.asarray(lst)


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
            # get_data(picks=...) returns just the requested channel(s)
            # without deep-copying the entire (preloaded) raw object, unlike
            # raw.copy().pick([name]) which duplicates the full data array.
            if raw.get_data(picks=[name]).var() > 1e-25:
                hpi_names += [name]

    # Exact lookup preserves one index per discovered name even when names
    # overlap (e.g. hpiout1 and hpiout11).
    name_to_index = {name: index for index, name in enumerate(raw.ch_names)}
    hpi_indices = np.asarray([name_to_index[name] for name in hpi_names], dtype=np.int64)

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
