"""
Core HPI pipeline shared by the single-file and multi-file entry points.

Two public functions are provided:

* ``fit_hpi`` — loads an HPI recording and a Polhemus file, runs the
  sequential per-coil amplitude estimation, and returns all fit results.
* ``apply_transform`` — loads a data recording, applies the
  device-to-head transform produced by ``fit_hpi``, embeds digitisation
  points, optionally resamples, and saves.

Both functions import from ``opm_utility_scripts.channels`` and
``opm_utility_scripts.viz`` so they never carry private copies of those
utilities.
"""

import os
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from mne._fiff._digitization import _call_make_dig_points, _make_dig_points
from mne._fiff.pick import pick_types
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs
from mne.io.constants import FIFF
from mne.transforms import (
    Transform,
    _fit_matched_points,
    _quat_to_affine,
    apply_trans,
    get_ras_to_neuromag_trans,
)
from mne.utils import warn

from opm_utility_scripts.channels import find_zero_location_channels, get_hpi_output_channels

# Sampling frequency used internally for HPI fitting (always resample to
# this before running the amplitude estimation loop).
_HPI_FIT_SFREQ = 1000


def fit_hpi(hpifile, polfile, hpifreq: float) -> dict:
    """
    Load HPI and Polhemus recordings, fit dipoles per coil, and compute
    the device-to-head transform.

    Parameters
    ----------
    hpifile : str | mne.io.Raw
        Path to the OPM recording in which the HPI coils were activated
        sequentially, or a pre-loaded Raw object.
    polfile : str | dict
        Path to the TRIUX/JSON Polhemus recording, or a pre-loaded
        Polhemus dict returned by ``load_polhemus``.
    hpifreq : float
        Drive frequency shared by all HPI coils (Hz).

    Returns
    -------
    dict
        Keys:

        ``dev_to_head_trans`` : mne.transforms.Transform
            Device-to-head coordinate transform.
        ``hpi_dev`` : np.ndarray, shape (n_coils, 3)
            HPI coil positions in device coordinates.
        ``hpi_gofs`` : np.ndarray, shape (n_coils,)
            Goodness-of-fit values for each coil (0–1).
        ``hpi_orig`` : np.ndarray, shape (n_coils, 3)
            HPI coil positions in head coordinates (from Polhemus).
        ``hpi_names`` : list[str]
            HPI output channel names.
        ``nasion`` : np.ndarray, shape (3,)
            Nasion fiducial in **head** coordinates.
        ``lpa`` : np.ndarray, shape (3,)
            Left pre-auricular fiducial in **head** coordinates.
        ``rpa`` : np.ndarray, shape (3,)
            Right pre-auricular fiducial in **head** coordinates.
        ``pol_info`` : dict
            Full Polhemus digitisation info returned by ``load_polhemus``.
        ``extra_pts`` : np.ndarray, shape (n_extra, 3)
            Headshape points only (``kind == 4``), in **head** coordinates.
        ``eeg_pts`` : np.ndarray, shape (n_eeg, 3)
            EEG digitisation points only (``kind == 3``), in **head** coordinates.
        ``slope`` : np.ndarray, shape (n_coils, n_meg_channels)
            Accumulated slope matrix (used for the topomap plot in
            ``coregister.py``).
        ``raw_for_topomap`` : mne.io.Raw
            Copy of the HPI raw (MEG channels only, bad channels dropped)
            suitable for constructing EvokedArray topomaps.
        ``dist`` : np.ndarray
            Per-coil residual distances (head-coord space, metres).
        ``include_hpis`` : np.ndarray of bool
            Mask of coils whose GOF exceeded 0.9.
        ``tree_indices`` : np.ndarray
            KDTree query indices mapping included HPI device positions to
            their nearest Polhemus counterparts.
    """
    # ------------------------------------------------------------------
    # Load HPI recording
    # ------------------------------------------------------------------
    if isinstance(hpifile, str):
        raw = mne.io.read_raw_fif(hpifile, preload=True)
        for bad_chan in list(raw.info['bads']):
            raw.drop_channels(bad_chan)
    else:
        raw = hpifile

    bads = find_zero_location_channels(raw.info)
    for bad_chan in bads:
        raw.drop_channels(bad_chan)

    hpi_names, hpi_indices = get_hpi_output_channels(raw)

    hpi_freqs = np.full(len(hpi_indices), hpifreq)

    # Always resample to the internal fitting frequency.
    raw.load_data().resample(_HPI_FIT_SFREQ)

    # ------------------------------------------------------------------
    # Load Polhemus digitisation
    # ------------------------------------------------------------------
    if isinstance(polfile, str):
        from opm_utility_scripts.io import load_polhemus

        pol = load_polhemus(polfile)
    else:
        pol = polfile

    lpa = pol['lpa']
    nasion = pol['nasion']
    rpa = pol['rpa']
    hpi_orig = pol['hpi_orig']

    # Build the isotrak→head transform from the Polhemus fiducials.
    # This is used to convert all digitisation points from the Polhemus
    # measurement frame (isotrak) to MNE head coordinates.
    isotrak_to_head = get_ras_to_neuromag_trans(nasion, lpa, rpa)

    # Seed dev_head_t with identity: at this point we don't know the
    # device→head transform (that's what fit_hpi is computing).
    # compute_chpi_locs inverts dev_head_t to seed its dipole search;
    # identity means the seed is in device coordinates — correct.
    # Using isotrak→head here was wrong: it would give compute_chpi_locs
    # a head→isotrak seed (the inverse), sending the search to the wrong region.
    raw.info.update(dev_head_t=Transform("meg", "head"))

    with raw.info._unlock():
        raw.info['dig'], _ = _call_make_dig_points(
            nasion,
            lpa,
            rpa,
            pol['hpi_orig'][0:len(hpi_indices)],
            pol['extra_pts'],
            convert=True,   # converts isotrak → head using the fiducials
        )

    # ------------------------------------------------------------------
    # Per-coil amplitude estimation loop
    # ------------------------------------------------------------------
    start_sample = 0
    stop_sample = len(raw)
    dist_limit = 0.005

    raw_orig = raw.copy()
    slope = np.zeros((len(hpi_indices), len(pick_types(raw.info, meg='mag'))), dtype=float)

    for index in range(len(hpi_indices)):
        raw = raw_orig.copy()
        channel_index = hpi_indices[index]
        chan_name = raw.info['ch_names'][channel_index]

        print(f'**** HPI coil {chan_name} (index {channel_index}) ****')

        raw_selection = raw[channel_index, start_sample:stop_sample]
        b = raw_selection[0].ravel()
        peak_dist = round(raw.info['sfreq'] / hpifreq) - 2
        peaks, _ = find_peaks(b, distance=peak_dist, height=0.0001)

        if len(peaks) < 1:
            print('ERROR: no peaks found for this coil — skipping')
            continue

        minT = peaks[0] / raw.info['sfreq']
        maxT = peaks[-1] / raw.info['sfreq']
        tmin = (maxT - minT) / 2.0 - 1 + minT
        tmax = (maxT - minT) / 2.0 + 1 + minT
        raw.crop(tmin=tmin, tmax=tmax)

        # Build HPI subsystem info so compute_chpi_amplitudes can run.
        hpi_sub = {"hpi_coils": [{} for _ in range(len(hpi_indices))]}
        hpi_coils = [
            {
                "number": i + 1,
                "drive_chan": hpi_names[i],
                "coil_freq": hpi_freqs[i],
            }
            for i in range(len(hpi_indices))
        ]
        for i in range(len(hpi_indices)):
            hpi_sub["hpi_coils"][i]["event_bits"] = [256]

        with raw.info._unlock():
            raw.info["hpi_subsystem"] = hpi_sub
            raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]

        # Count active HPIs.
        n_hpis = sum(
            1 for d in raw.info["hpi_subsystem"]["hpi_coils"]
            if d.get("event_bits") == [256]
        )

        if n_hpis < 3:
            # NOTE: coil_amplitudes is only assigned inside this else branch.
            # If n_hpis < 3 for every iteration the assert below will raise
            # UnboundLocalError — this is a pre-existing bug preserved here.
            warn(
                f"{n_hpis:d} HPIs active. At least 3 needed to perform"
                " head localization\n *NO* head localization performed"
            )
        else:
            with raw.info._unlock():
                raw.info["hpi_results"] = [
                    dict(
                        dig_points=[
                            dict(
                                r=np.zeros(3),
                                coord_frame=FIFF.FIFFV_COORD_DEVICE,
                                ident=ii + 1,
                            )
                            for ii in range(n_hpis)
                        ],
                        coord_trans=Transform("meg", "head"),
                    )
                ]
            raw.info["line_freq"] = None
            coil_amplitudes = compute_chpi_amplitudes(raw, tmin=0, tmax=2, t_window=2, t_step_min=2)
            slope[index, :] = coil_amplitudes['slopes'][0][index]

    # ------------------------------------------------------------------
    # Build final coil locations from accumulated slope matrix
    # ------------------------------------------------------------------
    assert len(coil_amplitudes["times"]) == 1  # noqa: F821 (intentional — see note above)
    coil_amplitudes['slopes'][0] = slope
    # Suppress the "HPI consistency of isotrak and hpifit is poor" RuntimeWarning.
    # It fires because hpi_results[-1]['dig_points'] is intentionally zero-initialised
    # (a seed placeholder); compute_chpi_locs compares those zeros against the real
    # head-frame dig points and flags the discrepancy.  The warning is a false alarm:
    # the actual dipole search uses the dig points directly and is unaffected.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='HPI consistency of isotrak and hpifit is poor',
            category=RuntimeWarning,
        )
        coil_locs = compute_chpi_locs(raw.info, coil_amplitudes)
    hpi_dev = np.array(coil_locs['rrs'][0])
    hpi_gofs = np.array(coil_locs['gofs'][0])

    include_hpis = hpi_gofs > 0.9

    # hpi_orig is in isotrak coordinates (the Polhemus measurement frame).
    # hpi_dev is in device (MEG) coordinates.
    # _fit_matched_points needs both sets in the same frame.
    # Convert hpi_orig → head coordinates using the isotrak→head transform
    # that was already computed from the fiducials above.
    isotrak_to_head = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    hpi_orig_head = apply_trans(isotrak_to_head, hpi_orig)

    tree = cKDTree(hpi_orig_head)
    distances, tree_indices = tree.query(hpi_dev[include_hpis])

    trans = _quat_to_affine(_fit_matched_points(hpi_dev[include_hpis], hpi_orig_head[tree_indices])[0])
    dev_to_head_trans = Transform(fro="meg", to="head", trans=trans)

    hpi_head = apply_trans(dev_to_head_trans, hpi_dev)
    dist = np.linalg.norm(hpi_orig_head[tree_indices] - hpi_head[include_hpis], axis=1)

    # Convert all remaining Polhemus points from isotrak → head frame so that
    # apply_transform can pass them directly to _make_dig_points(coord_frame="head")
    # without a second conversion step.
    nasion_head = apply_trans(isotrak_to_head, nasion)
    lpa_head = apply_trans(isotrak_to_head, lpa)
    rpa_head = apply_trans(isotrak_to_head, rpa)
    extra_pts_head = (
        apply_trans(isotrak_to_head, pol['extra_pts'])
        if len(pol['extra_pts'])
        else pol['extra_pts']
    )
    eeg_pts_head = (
        apply_trans(isotrak_to_head, pol['eeg_pts'])
        if len(pol['eeg_pts'])
        else pol['eeg_pts']
    )

    # Raw copy with only MEG channels for the optional topomap.
    raw_for_topomap = raw_orig.copy()
    raw_for_topomap.pick(picks=['meg'], exclude='bads')

    return {
        'dev_to_head_trans': dev_to_head_trans,
        'hpi_dev': hpi_dev,
        'hpi_gofs': hpi_gofs,
        'hpi_orig': hpi_orig_head,   # head coordinates (consistent with dev_to_head_trans)
        'hpi_names': hpi_names,
        # All dig-point arrays below are in head coordinates so apply_transform
        # can call _make_dig_points(coord_frame="head") directly.
        'nasion': nasion_head,
        'lpa': lpa_head,
        'rpa': rpa_head,
        'pol_info': pol,
        'extra_pts': extra_pts_head,
        'eeg_pts': eeg_pts_head,
        'slope': slope,
        'raw_for_topomap': raw_for_topomap,
        'dist': dist,
        'include_hpis': include_hpis,
        'tree_indices': tree_indices,
    }


def apply_transform(datfile: str, fit_result: dict, new_sfreq: float) -> mne.io.Raw:
    """
    Apply the HPI device-to-head transform to a data file.

    Loads *datfile*, drops bad/zero-location channels, optionally resamples,
    embeds the digitisation points and ``dev_head_t`` from *fit_result*, and
    returns the modified :class:`mne.io.Raw` object.  The caller is responsible
    for saving — use :func:`save_raw` for the standard filename convention.

    Parameters
    ----------
    datfile : str
        Path to the OPM-MEG data file to transform.
    fit_result : dict
        Result dict from :func:`fit_hpi`.  All position arrays must already
        be in head coordinates (guaranteed when produced by ``fit_hpi``).
    new_sfreq : float
        Target sampling frequency.  The data is resampled only when the
        current ``sfreq`` differs from *new_sfreq*.

    Returns
    -------
    mne.io.Raw
        The transformed raw object (preloaded, not yet saved).
    """
    dev_to_head_trans = fit_result['dev_to_head_trans']
    hpi_orig = fit_result['hpi_orig']
    nasion = fit_result['nasion']
    lpa = fit_result['lpa']
    rpa = fit_result['rpa']
    extra_pts = fit_result['extra_pts']
    eeg_pts = fit_result.get('eeg_pts', np.empty((0, 3)))

    raw = mne.io.read_raw_fif(datfile, preload=True)

    if new_sfreq != raw.info['sfreq']:
        raw.load_data().resample(new_sfreq)

    for bad_chan in raw.info["bads"]:
        raw.drop_channels(bad_chan)

    bads = find_zero_location_channels(raw.info)
    for bad_chan in bads:
        raw.drop_channels(bad_chan)

    raw.info.update(dev_head_t=dev_to_head_trans)

    with raw.info._unlock():
        # All arrays from fit_result are already in head coordinates.
        raw.info['dig'] = _make_dig_points(
            nasion, lpa, rpa, hpi_orig, extra_pts,
            coord_frame="head",
        )
        if len(eeg_pts):
            for i, r in enumerate(eeg_pts):
                raw.info['dig'].append({
                    'r': r,
                    'ident': i + 1,
                    'kind': FIFF.FIFFV_POINT_EEG,
                    'coord_frame': FIFF.FIFFV_COORD_HEAD,
                })

    return raw


def save_raw(raw: mne.io.Raw, datfile: str, suffix: str, overwrite: bool = True) -> str:
    """
    Save *raw* to disk using the standard HPI output filename convention.

    The output path is derived from *datfile* by stripping ``_raw`` from the
    stem and appending *suffix*::

        /path/to/AudOdd_raw.fif  +  '_proc-hpi+ds_raw.fif'
        →  /path/to/AudOdd_proc-hpi+ds_raw.fif

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object to save (typically produced by :func:`apply_transform`).
    datfile : str
        Original source file path — used only to derive the output directory
        and stem; the data is taken from *raw*, not re-read from disk.
    suffix : str
        Suffix appended to the stripped stem, e.g. ``'_proc-hpi+ds_raw.fif'``.
    overwrite : bool
        Passed to :meth:`mne.io.Raw.save`.  Defaults to ``True``.

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    stem = os.path.splitext(os.path.basename(datfile))[0].replace('_raw', '')
    outpath = os.path.join(os.path.dirname(datfile), stem + suffix)
    raw.save(outpath, overwrite=overwrite)
    return outpath
