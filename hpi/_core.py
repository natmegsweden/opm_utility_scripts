"""
Core HPI pipeline shared by the single-file and multi-file entry points.

Three public functions are provided:

* ``fit_hpi_amplitudes`` — loads an HPI recording, runs the sequential
  per-coil amplitude estimation, and returns device-space coil positions,
  GOFs, and related data.  Does **not** require a Polhemus file.
* ``fit_hpi`` — calls ``fit_hpi_amplitudes`` then loads the Polhemus file,
  computes the device-to-head transform, and returns a full result dict.
* ``apply_transform`` — loads a data recording, applies the
  device-to-head transform produced by ``fit_hpi``, embeds digitisation
  points, optionally resamples, and saves.

Both functions import from ``opm_utility_scripts.channels`` and
``opm_utility_scripts.viz`` so they never carry private copies of those
utilities.
"""

import itertools
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from mne import Info, pick_info
from mne._fiff._digitization import _call_make_dig_points, _make_dig_points
from mne._fiff.pick import pick_types, pick_channels
from mne.chpi import (
    compute_chpi_amplitudes,
    compute_whitener,
    make_ad_hoc_cov,
    _concatenate_coils,
    _create_meg_coils,
    # _magnetic_dipole_delta # New location in mne 1.13 (_chpi_numba.py) defined in script
    _magnetic_dipole_field_vec,
    _get_hpi_initial_fit,
    _check_chpi_param,
    _fit_magnetic_dipole,
)
from mne.io.constants import FIFF
from mne.transforms import (
    Transform,
    _fit_matched_points,
    _quat_to_affine,
    apply_trans,
    get_ras_to_neuromag_trans,
    combine_transforms,
    invert_transform,
)
from mne.bem import ConductorModel
from mne.dipole import _make_guesses
from mne.utils import warn, ProgressBar, _check_option, _validate_type
from mne.utils.check import _verbose_safe_false
from ..channels import find_zero_location_channels, get_hpi_output_channels

# Sampling frequency used internally for HPI fitting (always resample to
# this before running the amplitude estimation loop).
_HPI_FIT_SFREQ = 1000

def _make_opm_guesses(meg_coils):
    R = np.linalg.norm(meg_coils[0], axis=1).max()

    sphere = ConductorModel(
        layers=[dict(rad=R)],
        r0=np.zeros(3),
        is_sphere=True,
    )

    guesses = _make_guesses(
        sphere,
        0.002,
        0.0,
        0.001,
    )[0]["rr"]

    guesses = guesses[
        np.linalg.norm(guesses, axis=1)
        <= np.linalg.norm(meg_coils[0], axis=1).min()
    ]

    return guesses

# New location in mne 1.13 (_chpi_numba.py) 
def _magnetic_dipole_delta(fwd, whitener, B, B2):
    # Here we use .T to get whitener to Fortran order, which speeds things up
    fwd = fwd @ whitener.T
    u, s, v = np.linalg.svd(fwd, full_matrices=False)
    one = v @ B
    Bm2 = one @ one
    return B2 - Bm2, u, s, one

def _gof_at_fixed_pos(slope_row, pos_dev, whitener, meg_coils):
    """Evaluate dipole GOF at a *fixed* device-space position.

    Unlike the floating-dipole fit in ``compute_chpi_opm_locs``, this does not
    optimise the position — it evaluates how well a dipole *at ``pos_dev``*
    explains the measured field pattern ``slope_row``.

    GOF = 1 − ||B_whitened − B_model(pos)||² / ||B_whitened||²

    Parameters
    ----------
    slope_row : np.ndarray, shape (n_meg,)
        One row of the slope matrix (measured field for one coil).
    pos_dev : np.ndarray, shape (3,)
        Fixed dipole position in device coordinates (metres).
    whitener : np.ndarray
        Whitening matrix from ``compute_whitener``.
    meg_coils : object
        Concatenated MEG coil geometry from ``_concatenate_coils``.

    Returns
    -------
    float
        GOF in [0, 1].  Returns ``nan`` if signal power is negligible.
    """

    B  = np.dot(whitener, slope_row)
    B2 = float(np.dot(B, B))
    if B2 < 1e-30:
        return float('nan')
    fwd = _magnetic_dipole_field_vec(pos_dev[np.newaxis], meg_coils, 'info')
    residual, *_ = _magnetic_dipole_delta(fwd, whitener, B, B2)
    return float(1.0 - residual / B2)

def perturb_transform(T0, params):
    rvec = params[:3]
    tvec = params[3:]
    R = Rotation.from_rotvec(rvec).as_matrix()
    delta = np.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = tvec
    T = delta @ T0["trans"]
    return Transform(
        fro=T0["from"],
        to=T0["to"],
        trans=T,
    )

def find_bads(reffile=None, hpifreq=None, match_channels=None):
    """Detect noisy MEG channels from a reference recording (e.g. resting state).

    Parameters
    ----------
    reffile : str | None
        Path to a reference FIF recording used for background-power-based
        outlier detection. When ``None`` (default), no reference file is
        available and bad-channel detection is skipped entirely — an empty
        list is returned.
    hpifreq : float | None
        HPI drive frequency in Hz, used only to label the diagnostic plot.
    match_channels : list[str] | None
        Channel names to restrict the reference recording to before running
        outlier detection — pass the *target* recording's own remaining
        channel names here (e.g. the HPI raw's ``info['ch_names']`` after
        its own zero-location/explicit-bad channels have already been
        dropped). When ``reffile`` is an independent recording (not simply
        a cropped copy of the target raw), its channel population, its own
        ``info['bads']``, and its own zero-location channels generally
        differ from the target's. Without this restriction the background
        power mean/std/threshold used for the z-score cut are computed over
        the *reference file's* population, not the target's — channels
        present only in the reference pollute the statistics, and channels
        present only in the target are never tested at all. Restricting to
        the intersection here makes the returned bad-channel list, and the
        statistics behind it, reflect exactly the channel set that will
        actually be used downstream. When ``None`` (default), no
        restriction is applied — matches the previous behaviour.

    Returns
    -------
    list[str]
        Names of channels flagged as noisy. Empty when ``reffile`` is
        ``None``.
    """
    if reffile is None:
        return [], None, []

    # If reffile is defined as a string, load it as a raw object. Otherwise, assume it's already a raw object.
    if isinstance(reffile, str):
        raw = mne.io.read_raw_fif(reffile)
        raw.load_data()
    else:
        # Raw object is deferred from last 5 seconds of HPI recorging
        raw = reffile

    # Remove bad-marked and unlocalized channels in a single batched call.
    # Dropping channels one at a time reallocates the full (preloaded) data
    # array on every call, which is expensive when there are many of them.
    to_drop = list(dict.fromkeys(
        list(raw.info["bads"]) + list(find_zero_location_channels(raw.info))
    ))
    if to_drop:
        raw.drop_channels(to_drop)

    # Restrict to the channels also present in the target recording so the
    # outlier statistics (and the returned bad list) are computed over the
    # exact channel population that will actually be used downstream — see
    # the ``match_channels`` docstring above for why this matters whenever
    # ``reffile`` is an independently-loaded recording.
    if match_channels is not None:
        keep_set = set(match_channels)
        extra = [ch for ch in raw.ch_names if ch not in keep_set]
        if extra:
            raw.drop_channels(extra)

    # Detect outliers: use meg=True (covers both mag and grad in MEG),
    # eeg=False (exclude EEG channels), exclude='bads' to skip pre-marked bads.
    picks = pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    spectrum = raw.compute_psd(picks=picks, method="welch", fmin=70, fmax=80, n_fft=5000, n_per_seg=5000)
    psds = spectrum.get_data()
    background_power = psds.mean(axis=1)

    # Record the channel names corresponding to the PSD picks — used later as
    # slope_ch_names so downstream topomap geometry can be validated.
    slope_ch_names = [raw.ch_names[p] for p in picks]

    good_idx = np.arange(len(background_power))

    for _ in range(5): #iteratively remove outliers based on z-score>3
        mean_power = np.mean(background_power[good_idx])
        std_power = np.std(background_power[good_idx])
        threshold = mean_power + 3 * std_power
        new_good_idx = np.where(background_power <= threshold)[0]
        if len(new_good_idx) == len(good_idx):
            break
        good_idx = new_good_idx

    bad_idx = np.setdiff1d(np.arange(len(background_power)), good_idx)
    bad_chs = [slope_ch_names[idx] for idx in bad_idx]

    x = np.arange(len(background_power))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, background_power, 'ko', label='Background power')
    ax.axhline( # Threshold
        threshold,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f'Threshold ({threshold:.2e})'
    )
    ax.plot( # Outliers
        bad_idx,
        background_power[bad_idx],
        'r+',
        markersize=10,
        label='Bad channels'
    )

    for idx in bad_idx:
        ax.text(
            idx,
            background_power[idx],
            bad_chs,
            rotation=45,
            fontsize=8,
            color='red'
        )

    ax.set_xlabel('Channel')
    ax.set_ylabel('Background PSD')
    freq_label = f'{hpifreq} Hz' if hpifreq is not None else 'unknown Hz'
    ax.set_title(f'Bad Channel Detection Around HPI Frequency ({freq_label})')
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    #plt.show()
    
    return bad_chs, fig, slope_ch_names

def compute_chpi_opm_locs(
    info,
    chpi_amplitudes,
    t_step_max=1.0,
    too_close="raise",
    adjust_dig=False,
    *,
    verbose=None,
):
    """Compute locations of each cHPI coils over time.

    Parameters
    ----------
    %(info_not_none)s
    %(chpi_amplitudes)s
        Typically obtained by :func:`mne.chpi.compute_chpi_amplitudes`.
    t_step_max : float
        Maximum time step to use.
    too_close : str
        How to handle HPI positions too close to the sensors,
        can be ``'raise'`` (default), ``'warning'``, or ``'info'``.
    %(adjust_dig_chpi)s
    %(verbose)s

    Returns
    -------
    %(chpi_locs)s

    See Also
    --------
    compute_chpi_amplitudes
    compute_head_pos
    read_head_pos
    write_head_pos
    extract_chpi_locs_ctf

    Notes
    -----
    This function is designed to take the output of
    :func:`mne.chpi.compute_chpi_amplitudes` and:

    1. Get HPI coil locations (as digitized in ``info['dig']``) in head coords.
    2. If the amplitudes are 98%% correlated with last position
       (and Δt < t_step_max), skip fitting.
    3. Fit magnetic dipoles using the amplitudes for each coil frequency.

    The number of fitted points ``n_pos`` will depend on the velocity of head
    movements as well as ``t_step_max`` (and ``t_step_min`` from
    :func:`mne.chpi.compute_chpi_amplitudes`).

    .. versionadded:: 0.20
    """

    # Set up magnetic dipole fits
    _check_option("too_close", too_close, ["raise", "warning", "info"])
    _check_chpi_param(chpi_amplitudes, "chpi_amplitudes")
    _validate_type(info, Info, "info")
    _validate_type(info["dev_head_t"], Transform, "info['dev_head_t']")
    sin_fits = chpi_amplitudes  # use the old name below
    del chpi_amplitudes
    proj = sin_fits["proj"]
    meg_picks = pick_channels(info["ch_names"], proj["data"]["col_names"], ordered=True)
    info = pick_info(info, meg_picks)  # makes a copy
    with info._unlock():
        info["projs"] = [proj]
    del meg_picks, proj
    meg_coils = _concatenate_coils(_create_meg_coils(info["chs"], "accurate"))

    # Set up external model for interference suppression
    safe_false = _verbose_safe_false()
    cov = make_ad_hoc_cov(info, verbose=safe_false)
    whitener, _ = compute_whitener(cov, info, verbose=safe_false)

    # Make location guesses
    guesses = _make_opm_guesses(meg_coils)
    R = np.linalg.norm(meg_coils[0], axis=1).max()
    fwd = _magnetic_dipole_field_vec(guesses, meg_coils, too_close)
    fwd = np.dot(fwd, whitener.T)
    fwd.shape = (guesses.shape[0], 3, -1)
    fwd = np.linalg.svd(fwd, full_matrices=False)[2]
    guesses = dict(rr=guesses, whitened_fwd_svd=fwd)
    del fwd, R

    iter_ = list(zip(sin_fits["times"], sin_fits["slopes"]))
    chpi_locs = dict(times=[], rrs=[], gofs=[], moments=[])
    # setup last iteration structure
    hpi_dig_dev_rrs = apply_trans(
        invert_transform(info["dev_head_t"])["trans"],
        _get_hpi_initial_fit(info, adjust=adjust_dig),
    )
    last = dict(
        sin_fit=None,
        coil_fit_time=sin_fits["times"][0] - 1,
        coil_dev_rrs=hpi_dig_dev_rrs,
    )
    n_hpi = len(hpi_dig_dev_rrs)
    del hpi_dig_dev_rrs
    for fit_time, sin_fit in ProgressBar(iter_, mesg="cHPI locations "):
        # skip this window if bad
        if not np.isfinite(sin_fit).all():
            continue

        # check if data has sufficiently changed
        if last["sin_fit"] is not None:  # first iteration
            corrs = np.array(
                [np.corrcoef(s, lst)[0, 1] for s, lst in zip(sin_fit, last["sin_fit"])]
            )
            corrs *= corrs
            # check to see if we need to continue
            if (
                fit_time - last["coil_fit_time"] <= t_step_max - 1e-7
                and (corrs > 0.98).sum() >= 3
            ):
                # don't need to refit data
                continue

        # update 'last' sin_fit *before* inplace sign mult
        last["sin_fit"] = sin_fit.copy()

        #
        # 2. Fit magnetic dipole for each coil to obtain coil positions
        #    in device coordinates
        #
        coil_fits = [
            _fit_magnetic_dipole(f, x0, too_close, whitener, meg_coils, guesses)
            for f, x0 in zip(sin_fit, last["coil_dev_rrs"])
        ]
        rrs, gofs, moments = zip(*coil_fits)
        chpi_locs["times"].append(fit_time)
        chpi_locs["rrs"].append(rrs)
        chpi_locs["gofs"].append(gofs)
        chpi_locs["moments"].append(moments)
        last["coil_fit_time"] = fit_time
        last["coil_dev_rrs"] = rrs
    n_times = len(chpi_locs["times"])
    shapes = dict(
        times=(n_times,),
        rrs=(n_times, n_hpi, 3),
        gofs=(n_times, n_hpi),
        moments=(n_times, n_hpi, 3),
    )
    for key, val in chpi_locs.items():
        chpi_locs[key] = np.array(val, float).reshape(shapes[key])
    return chpi_locs

def fit_hpi_amplitudes(hpifile, hpifreq: float) -> dict:
    """
    Load an HPI recording and estimate per-coil dipole positions and GOFs.

    This is the HPI-only stage of the pipeline — no Polhemus file is needed.
    It is called internally by :func:`fit_hpi` and can also be used directly
    by :mod:`opm_utility_scripts.hpi.check` for HPI-only quality checks.

    Parameters
    ----------
    hpifile : str | mne.io.Raw
        Path to the raw HPI recording, or a pre-loaded Raw object.
    hpifreq : float
        Drive frequency shared by all HPI coils (Hz).

    Returns
    -------
    dict
        Keys:

        ``hpi_names`` : list[str]
            HPI output channel names (from ``get_hpi_output_channels``).
        ``hpi_indices`` : np.ndarray
            Channel indices of the HPI output channels in ``raw_orig``.
        ``hpi_freqs`` : np.ndarray, shape (n_coils,)
            Drive frequency (Hz) used for each coil.  All entries are
            currently identical (``hpifreq`` repeated) since coils are
            fit sequentially at one shared frequency, but this is kept
            per-coil so callers can detect distinct-frequency setups.
        ``slope`` : np.ndarray, shape (n_coils, n_meg_mag_channels)
            Accumulated amplitude slope matrix.
        ``slope_ch_names`` : list[str]
            Ordered MEG sensor names identifying the slope columns.
        ``original_coil_indices`` : np.ndarray
            Surviving coil indices in the original drive-channel ordering.
        ``peak_tlast`` : float
            Last detected drive peak time in seconds.
        ``raw_orig`` : mne.io.Raw
            The resampled raw with HPI subsystem metadata set.
            Caller must embed ``dig`` before calling ``compute_chpi_opm_locs``.
        ``coil_amplitudes`` : dict
            The ``compute_chpi_amplitudes`` result with the accumulated
            slope matrix injected.  Pass this to ``compute_chpi_opm_locs``
            after embedding proper dig points into ``raw_orig.info``.
    """

    # ------------------------------------------------------------------
    # Load HPI recording
    # ------------------------------------------------------------------
    if isinstance(hpifile, str):
        raw = mne.io.read_raw_fif(hpifile, preload=True, verbose='error')
        bad_marked = list(raw.info['bads'])
        if bad_marked:
            raw.drop_channels(bad_marked)
    else:
        raw = hpifile.copy()
        if raw.info['bads']:
            raw.drop_channels(list(raw.info['bads']))
        
    # Drop channels with zero/invalid location before any MEG-geometry-based
    # computation. Left in place, these cause a division by zero (r_n == 0)
    # when MNE builds the spherical-harmonic external-interference basis in
    # compute_chpi_amplitudes -> _setup_ext_proj, propagating NaN/Inf into an
    # SVD call and crashing with "array must not contain infs or NaNs".
    # Batched into a single drop_channels() call — dropping one at a time
    # reallocates the full preloaded data array on every call.
    zero_loc = list(find_zero_location_channels(raw.info))
    if zero_loc:
        raw.drop_channels(zero_loc)

    hpi_names, hpi_indices = get_hpi_output_channels(raw)
    hpi_freqs = np.full(len(hpi_indices), hpifreq)

    # Some acquisition systems (e.g. FieldLine OPM) write line_freq=0 to mean
    # "no notch filter configured" instead of leaving it unset. MNE's
    # compute_chpi_amplitudes only guards against `None` and divides by
    # info['line_freq'] otherwise, so a literal 0 raises ZeroDivisionError
    # in _setup_hpi_amplitude_fitting. Normalise 0 -> None here; real line
    # frequencies (50/60 Hz) are left untouched.
    if raw.info.get('line_freq') == 0:
        with raw.info._unlock():
            raw.info['line_freq'] = 50.0 # Or None, can't be 0.

    # Always resample to the internal fitting frequency.
    raw.load_data().resample(_HPI_FIT_SFREQ, verbose='error')

    # Seed dev_head_t with identity so compute_chpi_opm_locs searches in
    # device coordinates (correct — we have not computed the transform yet).
    raw.info.update(dev_head_t=Transform("meg", "head"))

    # ------------------------------------------------------------------
    # Per-coil amplitude estimation loop
    # ------------------------------------------------------------------
    raw_orig = raw.copy()
    # Record the MEG-mag column ordering at the *start* (before any per-coil
    # crops alter raw).  The same ordering must be used for the slope matrix,
    # the projector column names, whitener, meg_coils geometry, fixed-position
    # pol_gofs, and topomap — all downstream paths assume a consistent
    # sensor ordering derived from the original full recording.
    # Use meg=True, eeg=False to pick all MEG sensor types (mag + grad)
    # matching MNE convention, not just magnetometers.
    _meg_mag_picks = pick_types(raw.info, meg=True, eeg=False, exclude='bads')
    slope_ch_names = [raw.ch_names[p] for p in _meg_mag_picks]
    n_meg = len(_meg_mag_picks)
    slope = np.zeros((len(hpi_indices), n_meg), dtype=float)
    n_hpis = 0
    i_hpis = []
    peak_tmax = []
    for index in range(len(hpi_indices)):
        raw = raw_orig.copy()
        channel_index = hpi_indices[index]
        chan_name = raw.info['ch_names'][channel_index]

        b = raw[channel_index, :][0].ravel()
        peak_dist = round(raw.info['sfreq'] / hpifreq) - 2
        peaks, _ = find_peaks(b, distance=peak_dist, height=0.0001)

        if len(peaks) < 1:
            print('ERROR: no peaks found for this coil — skipping')
            continue

        minT = peaks[0] / raw.info['sfreq']
        maxT = peaks[-1] / raw.info['sfreq']
        peak_tmax.append(maxT)

        # Centered 2-second crop: midpoint ± 1 s, explicitly guard against
        # negative times and insufficient recording/activation length.
        mid = (minT + maxT) / 2.0
        if maxT - minT < 2.0 - 2.0 / raw.info['sfreq']:
            raise ValueError(f"Cannot fit coil {chan_name}: activation is shorter than 2 seconds.")
        tmin = mid - 1.0
        tmax = mid + 1.0
        # Guard: clip to valid recording boundaries (no silent shortening).
        if tmin < raw.times[0]:
            raise ValueError(
                f"Cannot fit coil {chan_name}: centred crop tmin={tmin:.3f}s "
                f"is before recording start ({raw.times[0]:.3f}s). "
                f"Activation [{minT:.3f}, {maxT:.3f}]s is too early."
            )
        if tmax > raw.times[-1]:
            raise ValueError(
                f"Cannot fit coil {chan_name}: centred crop tmax={tmax:.3f}s "
                f"is beyond recording end ({raw.times[-1]:.3f}s). "
                f"Activation [{minT:.3f}, {maxT:.3f}]s is too late."
            )
        raw.crop(tmin=tmin, tmax=tmax)

        # Ensure the cropped duration is at least 2 seconds
        # (sample-rounding robust).
        actual_duration = (raw.last_samp - raw.first_samp + 1) / raw.info['sfreq']
        if actual_duration < 1.99:
            raise ValueError(
                f"Cannot fit coil {chan_name}: centred crop [{tmin:.3f}, {tmax:.3f}]s "
                f"is only {actual_duration:.4f}s after sample-rounding (< 2s)."
            )

        # Build HPI subsystem info for single coil so compute_chpi_amplitudes can run.
        hpi_sub = dict()

        hpi_sub["hpi_coils"] = []
        hpi_sub["hpi_coils"].append({})

        hpi_coils=[]
        hpi_coils.append({})

        # Use the *full* drive-channel name of the current coil, not the
        # first character of the first coil's name.
        drive_chan = hpi_names[index]
        coil_freq = hpi_freqs[index]

        # build coil structure
        hpi_coils[0]["number"] = 1
        hpi_coils[0]["drive_chan"] = drive_chan
        hpi_coils[0]["coil_freq"] = coil_freq

        hpi_sub["hpi_coils"][0]["event_bits"] = [256]

        with raw.info._unlock():
            raw.info["hpi_subsystem"] = hpi_sub
            raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]

        # Fit at [0, 2] from crop start (= midpoint-1 to midpoint+1).
        coil_amplitudes = compute_chpi_amplitudes(raw, tmin=0, tmax=2, t_window=2, t_step_min=2, verbose='error')
        slope[index,:] = coil_amplitudes['slopes'][0][0]
        i_hpis.append(index)
        n_hpis+=1
        
    if n_hpis == 0:
        raise ValueError(
            "No HPI coils had detectable peaks. Cannot proceed; "
            "check the HPI drive signal and recording."
        )
    # Filter per-coil metadata to match only coils with peaks, preserving
    # the original index mapping: hpi_names, hpi_indices, hpi_freqs, and
    # slope rows are filtered together consistently.  i_hpis holds the
    # original indices of coils that succeeded, and slope rows for failed
    # coils remain as unfilled zeros (they will be dropped below).
    slope = slope[i_hpis]         # keep only rows for coils with peaks
    hpi_names = [hpi_names[i] for i in i_hpis]
    hpi_indices = np.asarray(hpi_indices)[i_hpis]
    hpi_freqs   = hpi_freqs[i_hpis]
    peak_tlast  = max(peak_tmax)

    # Adding full hpi struct to info
    hpi_sub = dict()
    hpi_sub["hpi_coils"] = []
    for _ in range(len(hpi_indices)):
        hpi_sub["hpi_coils"].append({})

    hpi_coils=[]
    for _ in range(len(hpi_indices)):
        hpi_coils.append({})

    drive_channels = hpi_names
    default_freqs = hpi_freqs
    for i in range(len(hpi_indices)):
        # build coil structure
        hpi_coils[i]["number"] = i + 1
        hpi_coils[i]["drive_chan"] = drive_channels[i]
        hpi_coils[i]["coil_freq"] = default_freqs[i]
        hpi_sub["hpi_coils"][i]["event_bits"] = [256]

    with raw.info._unlock():
        raw.info["hpi_subsystem"] = hpi_sub
        raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]
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

    assert len(coil_amplitudes["times"]) == 1
    coil_amplitudes['slopes'] = np.zeros((1,slope.shape[0],slope.shape[1]))
    coil_amplitudes['slopes'][0] = slope

    if n_hpis < 3:
        warn(
            f"{n_hpis:d} HPIs active. At least 3 needed to perform"
            "head localization\n *NO* head localization performed"
        )  
        
    # ------------------------------------------------------------------
    # Inject accumulated slope back into coil_amplitudes.
    # Also copy the HPI subsystem metadata from the last loop iteration's
    # `raw` onto `raw_orig` so compute_chpi_opm_locs can find hpi_results,
    # hpi_subsystem, hpi_meas, and line_freq on the info it will be called with.
    # ------------------------------------------------------------------

    with raw_orig.info._unlock():
        raw_orig.info['hpi_subsystem'] = raw.info.get('hpi_subsystem')
        raw_orig.info['hpi_meas']      = raw.info.get('hpi_meas')
        raw_orig.info['hpi_results']   = raw.info.get('hpi_results')
        raw_orig.info['line_freq']     = raw.info.get('line_freq')

    return {
        'hpi_names':       hpi_names,
        'hpi_indices':     hpi_indices,
        'hpi_freqs':       hpi_freqs,
        'slope':           slope,
        'slope_ch_names':  slope_ch_names,
        'raw_orig':        raw_orig,
        'coil_amplitudes': coil_amplitudes,
        'peak_tlast':       peak_tlast,
        'original_coil_indices': np.asarray(i_hpis, dtype=int),
    }

def _detect_noise(raw, reffile, hpifreq, peak_tlast_override=None):
    """Detect noisy MEG channels and return (bads_list, bads_fig).

    Parameters
    ----------
    raw : mne.io.Raw
        The HPI recording (preloaded, zero-location channels already dropped).
    reffile : str | mne.io.Raw | None
        Path to a reference recording, a pre-loaded Raw object, or None.
        When None, the last 5 seconds of *raw* are used as the reference
        segment, provided they are at least 5 seconds after the final
        activation peak (``peak_tlast_override``) or the last actual drive
        peak found by interleaving the HPI drive channels.
    hpifreq : float | None
        HPI drive frequency for plot labelling.
    peak_tlast_override : float | None
        If provided, use this as the last activation peak time instead of
        detecting peaks from scratch.  Used when ``_detect_noise`` is
        called *before* amplitude estimation, so the per-coil peak
        detection has not run yet.

    Returns
    -------
    bads : list[str]
        Channel names flagged as noisy.  Empty when detection is skipped.
    bads_fig : matplotlib.figure.Figure or None
    """
    if reffile is None:
        # Auto-detect from the tail of *raw*.
        tmax = raw.times[-1]
        duration = tmax - raw.times[0]
        # Compute the last actual drive peak from HPI output channels before
        # peak_tlast_override (if available), or from the raw recording itself.
        if peak_tlast_override is not None:
            last_peak = peak_tlast_override
        else:
            # Find drive peaks by scanning HPI output channels and interleaving.
            hpi_names_detect, _ = get_hpi_output_channels(raw)
            all_peaks = []
            for ch_name in hpi_names_detect:
                if ch_name in raw.info['ch_names']:
                    sig, = raw[raw.info['ch_names'].index(ch_name), :][0]
                    peak_dist = round(raw.info['sfreq'] / max(hpifreq, 1.0)) - 2
                    pk, _ = find_peaks(sig, distance=peak_dist, height=0.0001)
                    if len(pk):
                        all_peaks.append(pk[-1] / raw.info['sfreq'])
            last_peak = max(all_peaks) if all_peaks else 0.0

        twindow = 5.0
        # Guard: only fall back to last-5-seconds if the last peak is early
        # enough that 5 s of clean tail remains.
        if last_peak + 5.0 > duration:
            print("WARNING: HPI recording is too short to extract a reference "
                  "segment for bad-channel detection. Skipping.")
            return [], None
        print("Extracting 5 s reference segment from the end of the HPI "
              "recording for bad-channel detection.")
        tmin = tmax - twindow
        ref = raw.copy().crop(tmin=tmin, tmax=tmax)
        # Resample the reference segment to 1000 Hz before noise PSD so the
        # spectral resolution is consistent regardless of the original sfreq.
        ref.load_data().resample(1000.0, verbose='error')
    else:
        if isinstance(reffile, str):
            ref = mne.io.read_raw_fif(reffile, preload=True, verbose='error')
            ref.resample(1000.0, verbose='error')
        else:
            ref = reffile.copy()
            ref.load_data().resample(1000.0, verbose='error')

    bads, fig, _ = find_bads(ref, hpifreq, match_channels=raw.info["ch_names"])
    return bads, fig


def fit_hpi(hpifile, polfile, hpifreq: float,
            gof_limit: float = 0.95,
            landmark_weight: float = 1.0, optim: str = "rigid",
            reffile: str = None, center_matching: bool = True) -> dict:
    """
    Load HPI and Polhemus recordings, fit dipoles per coil, and compute
    the device-to-head transform.

    Calls :func:`fit_hpi_amplitudes` for the HPI stage, then loads the
    Polhemus file and computes the device-to-head transform.

    Parameters
    ----------
    hpifile : str | mne.io.Raw
        Path to the OPM recording in which the HPI coils were activated
        sequentially, or a pre-loaded Raw object.
    polfile : str | dict | mne.channels.DigMontage
        Path to the TRIUX/JSON Polhemus recording, a pre-loaded
        Polhemus dict returned by ``load_polhemus``, or a
        :class:`mne.channels.DigMontage` as returned by
        ``mne.channels.read_dig_fif()``.  Strings and DigMontage objects
        are passed through :func:`~opm_utility_scripts.io.load_polhemus`;
        dicts are used directly.
    hpifreq : float
        Drive frequency shared by all HPI coils (Hz).
    gof_limit : float (default 0.95)
        Minimum dipole GOF for a coil to be included in the device-to-head
        transform fit. Coils with ``hpi_gofs < gof_limit`` are excluded
        from the point-matching/rigid-transform step (Stage 5).
    landmark_weight : float
        Deprecated inactive compatibility parameter. No landmark-weighted
        objective is implemented; a non-default value emits a warning.
    optim : str (default "rigid")
        Optimization method applied after the initial HPI→Polhemus
        coregistration (Stage 5). ``"none"``: no refinement. ``"rigid"``
        (default): refine the device-to-head transform with a bounded
        L-BFGS-B search over a small rigid perturbation (±5° rotation,
        ±5 mm translation) that maximises the mean dipole-fit GOF at the
        digitised polhemus positions (Stage 6).
    reffile : str | None
        Path to a reference recording (e.g. resting state) used for
        background-power-based noisy-channel detection during the HPI
        amplitude-estimation stage. Optional — when ``None`` (default),
        the final five seconds are used only if they follow HPI activation.
        Detection and exclusions happen before amplitude estimation.
    center_matching : bool (default True)
        Whether to subtract each point cloud's centroid before the
        nearest-neighbour (cKDTree) match between fitted device-frame HPI
        coil positions and head-frame Polhemus positions. Centring makes
        the match invariant to a bulk translation offset, but not arbitrary
        inter-frame rotation. Set to
        ``False`` to match on the raw (uncentred) coordinates instead,
        reproducing the legacy pipeline's matching behaviour. This is
        independent of ``optim``: it changes *which points are matched*
        during the closed-form fit (Stage 5), not whether a post-fit
        optimisation is applied afterwards. Intended for regression
        testing / legacy-parity comparisons rather than routine use.
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
            Headshape points in **head** coordinates.
        ``eeg_pts`` : np.ndarray, shape (n_eeg, 3)
            EEG digitisation points in **head** coordinates.
        ``slope`` : np.ndarray, shape (n_coils, n_meg_channels)
            Accumulated slope matrix (used for topomaps in ``coregister.py``).
        ``raw_for_topomap`` : mne.io.Raw
            MEG-only copy of the HPI raw suitable for EvokedArray topomaps.
        ``slope_ch_names`` : list[str]
            Ordered sensor names shared by slopes and topomap geometry.
        ``bads`` : list[str]
            Detected noisy channels excluded before amplitude estimation.
        ``bads_fig`` : matplotlib.figure.Figure | None
            Noise diagnostic figure, when noise detection ran.
        ``optim`` : str
            Requested refinement method.
        ``opt_status`` : int | None
            Optimizer status, or None when refinement was not run.
        ``opt_success`` : bool | None
            Whether a finite successful refinement was accepted.
        ``dist`` : np.ndarray
            Per-coil residual distances in head space (metres).
        ``include_hpis`` : np.ndarray of bool
            Mask of coils whose GOF ≥ ``gof_limit``.
        ``tree_indices`` : np.ndarray
            KDTree indices mapping included device coils to Polhemus targets.
        ``pol_gofs`` : np.ndarray, shape (n_included_coils,)
            GOF of the Polhemus-digitised position against the measured MEG
            field for each *included* coil.  Computed by evaluating the
            magnetic dipole forward model at the polhemus position (in device
            frame) without any optimisation.  A low ``pol_gof`` means the
            digitised position does not explain the sensor data — indicating
            a polhemus digitisation error rather than an HPI recording error.
    """

    # ------------------------------------------------------------------
    # Stage 0: Validate and warn for documented-but-unimplemented options
    # ------------------------------------------------------------------
    # landmark_weight is accepted for API compatibility but not implemented.
    # The rigid fit uses HPI coil positions, not weighted fiducial landmarks.
    if landmark_weight != 1.0:
        warnings.warn(
            "landmark_weight != 1.0 is accepted for API compatibility but "
            "has no effect: the current rigid fit uses HPI coil positions "
            "only, without landmark constraints.",
            UserWarning, stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Stage 1: HPI amplitude estimation (no polhemus needed)
    # ------------------------------------------------------------------
    # Before fitting amplitudes, detect and remove bad/noisy channels
    # from the HPI recording.  This must happen *before* amplitude
    # estimation so the projector, whitener, and sensor geometry all
    # reflect the cleaned channel set — not merely a diagnostic copy.
    # ------------------------------------------------------------------
    # Stage 1a: Load and clean the HPI recording
    # ------------------------------------------------------------------
    if isinstance(hpifile, str):
        raw0 = mne.io.read_raw_fif(hpifile, preload=True, verbose='error')
        bad_marked = list(raw0.info['bads'])
        if bad_marked:
            raw0.drop_channels(bad_marked)
    else:
        raw0 = hpifile.copy().load_data()
        if raw0.info['bads']:
            raw0.drop_channels(list(raw0.info['bads']))

    # Drop zero/invalid-location channels before any MEG-geometry computation.
    zero_loc = list(find_zero_location_channels(raw0.info))
    if zero_loc:
        raw0.drop_channels(zero_loc)

    # Use the same internal sampling rate for noise PSD and amplitude fitting.
    raw0.load_data().resample(_HPI_FIT_SFREQ, verbose='error')

    # Detect and remove noisy channels.
    bads, bads_fig = _detect_noise(raw0, reffile, hpifreq, peak_tlast_override=None)
    bads_present = [i for i in bads if i in raw0.info["ch_names"]]
    print(f"Detected {len(bads_present)} bad channel(s) for exclusion.")
    if bads_present:
        raw0.drop_channels(bads_present)

    # Now run amplitude estimation on the cleaned raw0.
    amp = fit_hpi_amplitudes(raw0, hpifreq)
    hpi_names       = amp['hpi_names']
    hpi_indices     = amp['hpi_indices']
    hpi_freqs       = amp['hpi_freqs']
    slope           = amp['slope']
    raw_orig        = amp['raw_orig']
    coil_amplitudes = amp['coil_amplitudes']
    peak_tlast       = amp['peak_tlast']

    # ------------------------------------------------------------------
    # Stage 2: Load Polhemus and embed digitisation into raw
    # ------------------------------------------------------------------
    if isinstance(polfile, dict):
        pol = polfile
    else:
        from ..io import load_polhemus
        pol = load_polhemus(polfile)

    lpa      = pol['lpa']
    nasion   = pol['nasion']
    rpa      = pol['rpa']
    hpi_orig = pol['hpi_orig']

    # ------------------------------------------------------------------
    # Guard: polhemus HPI count must cover the active coils.
    # If the dig has fewer positions than active coils the fit will crash
    # with a shape mismatch in compute_chpi_opm_locs.  Surface this clearly.
    # ------------------------------------------------------------------
    n_pol_hpi = len(hpi_orig)
    n_active  = len(hpi_indices)
    if n_pol_hpi < n_active:
        raise ValueError(
            f'Polhemus has {n_pol_hpi} HPI dig point(s) but {n_active} active '
            f'HPI coil(s) were detected ({hpi_names}). '
            f'Every active coil needs a digitised position. '
            f'Check that the correct polhemus file is being used and that all '
            f'active coils were digitised.'
        )

    # ------------------------------------------------------------------
    # Coordinate-frame handling:
    # JSON polhemus → isotrak frame → must convert to head frame.
    # FIF polhemus  → already in head frame → use as-is.
    # ------------------------------------------------------------------
    pol_source = pol.get('source', 'json')

    if pol_source == 'fif':
        # Points are already in head coordinates — no conversion needed.
        hpi_orig_head  = hpi_orig
        nasion_head    = nasion
        lpa_head       = lpa
        rpa_head       = rpa
        extra_pts_head = pol['extra_pts']
        eeg_pts_head   = pol['eeg_pts']

        # Embed dig points in head frame directly.
        with raw_orig.info._unlock():
            raw_orig.info['dig'] = _make_dig_points(
                nasion_head, lpa_head, rpa_head,
                hpi_orig_head[0:len(hpi_indices)],
                extra_pts_head,
                coord_frame='head',
            )
    else:
        # JSON / isotrak frame — build isotrak→head transform from fiducials.
        isotrak_to_head = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        hpi_orig_head  = apply_trans(isotrak_to_head, hpi_orig)
        nasion_head    = apply_trans(isotrak_to_head, nasion)
        lpa_head       = apply_trans(isotrak_to_head, lpa)
        rpa_head       = apply_trans(isotrak_to_head, rpa)
        extra_pts_head = (
            apply_trans(isotrak_to_head, pol['extra_pts'])
            if len(pol['extra_pts']) else pol['extra_pts']
        )
        eeg_pts_head = (
            apply_trans(isotrak_to_head, pol['eeg_pts'])
            if len(pol['eeg_pts']) else pol['eeg_pts']
        )

        # Embed dig points, converting isotrak → head using fiducials.
        with raw_orig.info._unlock():
            raw_orig.info['dig'], _ = _call_make_dig_points(
                nasion, lpa, rpa,
                pol['hpi_orig'][0:len(hpi_indices)],
                pol['extra_pts'],
                convert=True,
            )

    # ------------------------------------------------------------------
    # Stage 4: Compute coil locations now that dig is properly set
    # ------------------------------------------------------------------
    # Suppress the "HPI consistency of isotrak and hpifit is poor" warning —
    # it fires because hpi_results dig_points are intentionally zero-initialised
    # (a seed placeholder). The actual dipole search uses the isotrak dig directly.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message='HPI consistency of isotrak and hpifit is poor',
            category=RuntimeWarning,
        )
        coil_locs = compute_chpi_opm_locs(raw_orig.info, coil_amplitudes)

    hpi_dev  = np.array(coil_locs['rrs'][0])
    hpi_gofs = np.array(coil_locs['gofs'][0])

    # ------------------------------------------------------------------
    # Stage 5: Compute device-to-head transform
    # ------------------------------------------------------------------
    print(f'GOF threshold: {gof_limit:.2f} ')
    include_hpis = hpi_gofs >= gof_limit

    dev_pts  = hpi_dev[include_hpis]       # fitted positions, device frame
    n_inc    = len(dev_pts)
    if n_inc < 3:
        raise ValueError(
            f"Only {n_inc} HPI coil(s) passed the GOF threshold "
            f"({gof_limit:.2f}). At least 3 are required for a well-determined "
            f"rigid transform. Redo the HPI recording."
        )
    n_pol    = len(hpi_orig_head)
    
    # Find matching coils in fits and polhemus by finding closest points 
    # between the two. This is taking advantage of the fact that we know that 
    # HEDSCAN device coordiantes are similar to head coordinates and transform 
    # will entail small rotations (<< 90°).
    if center_matching:
        # Shift both point clouds to their own centroid first, so the match
        # is invariant to a bulk translation offset between the device and
        # head frames (not just rotation) — avoids problems with bad coil
        # placement.
        ref_pts = hpi_orig_head - hpi_orig_head.mean(axis=0)
        query_pts = hpi_dev[include_hpis] - hpi_dev[include_hpis].mean(axis=0)
    else:
        # Legacy behaviour: match on raw (uncentred) coordinates. Only
        # invariant to rotation, not translation offset — kept for
        # regression testing / legacy-parity comparisons.
        ref_pts = hpi_orig_head
        query_pts = hpi_dev[include_hpis]
    tree = cKDTree(ref_pts)
    distances, tree_indices = tree.query(query_pts) # find closest points
    incl_idx = np.flatnonzero(include_hpis)

    # Guard against duplicate KD-tree assignments: two fitted coils mapping
    # to the same digitised point produces a degenerate rigid fit.
    if len(set(tree_indices)) < len(tree_indices):
        # Find duplicated polhemus indices (they appear more than once in
        # tree_indices). Report the fitted coil names that hit each duplicate.
        uniq, counts = np.unique(tree_indices, return_counts=True)
        dup_pol_idx = uniq[counts > 1]
        dup_msg_parts = []
        for pol_i in dup_pol_idx:
            dup_coil_names = [hpi_names[incl_idx[i]]
                              for i in range(len(tree_indices))
                              if tree_indices[i] == pol_i]
            dup_msg_parts.append(f"pol#{pol_i+1} ← {{{', '.join(dup_coil_names)}}}")
        raise ValueError(
            f"Duplicate KD-tree assignment: fitted coils map to the same "
            f"polhemus HPI point(s): {'; '.join(dup_msg_parts)}. "
            f"This produces a degenerate transform. Check that each HPI coil "
            f"has a distinct digitised position."
        )

    # Calculate transform
    trans = _quat_to_affine(_fit_matched_points(hpi_dev[include_hpis], hpi_orig_head[tree_indices])[0])
    dev_to_head_trans = Transform(fro="meg", to="head", trans=trans)

    # Compute per-coil residuals (single source of truth — used for both
    # the diagnostic print below and the stored 'dist' in the return dict).
    incl_idx = np.where(include_hpis)[0]
    hpi_head = apply_trans(dev_to_head_trans, hpi_dev)
    dist = np.linalg.norm(hpi_orig_head[tree_indices] - hpi_head[include_hpis], axis=1)

    # Diagnostic: per-coil assignment table (coil → polhemus point, post-fit distance).
    print('  HPI coil assignment:')
    for rank, pol_i in enumerate(tree_indices):
        ch_name = hpi_names[incl_idx[rank]]
        pol_pos = hpi_orig_head[pol_i] * 1000
        print(f'    {ch_name} → pol#{pol_i+1} '
              f'[{pol_pos[0]:.1f},{pol_pos[1]:.1f},{pol_pos[2]:.1f}] mm  '
              f'post-fit dist={dist[rank]*1000:.1f} mm')

    _DIST_WARN_MM = 10.0
    if np.any(dist * 1000 > _DIST_WARN_MM):
        bad = [hpi_names[incl_idx[i]]
               for i in range(len(dist)) if dist[i] * 1000 > _DIST_WARN_MM]
        warnings.warn(
            f"Large HPI coil residuals for: {bad}. "
            f"Max residual: {dist.max()*1000:.1f} mm (threshold {_DIST_WARN_MM:.0f} mm). "
            f"Coil(s) may have moved between digitisation and recording.",
            RuntimeWarning, stacklevel=2
        )

    # ------------------------------------------------------------------
    # Stage 6: Polhemus-position GOF (pre-refinement baseline)
    # ------------------------------------------------------------------
    raw_for_topomap = raw_orig.copy()
    # Use meg=True, eeg=False, exclude='bads' for consistent sensor picks
    # matching MNE convention (not a string 'meg' which only picks mag).
    topo_picks = pick_types(raw_for_topomap.info, meg=True, eeg=False, exclude='bads')
    raw_for_topomap.pick(picks=topo_picks)
    slope_ch_names = amp['slope_ch_names']

    # Assert that the slope matrix columns match the topomap channel ordering
    # before building geometry — mismatch causes silent wrong results.
    _meg_mag_picks_full = pick_types(raw_orig.info, meg=True, eeg=False)
    full_meg_ch_names = [raw_orig.ch_names[p] for p in _meg_mag_picks_full]
    # The slope was computed over meg=True, eeg=False picks, and both
    # geometry builds use the same channel ordering. Assert here.
    assert slope_ch_names == raw_for_topomap.ch_names and slope.shape[1] == len(slope_ch_names), (
        f"Mismatch between slope column channels and topomap channels. "
        f"Slope has {len(full_meg_ch_names)} cols: {full_meg_ch_names}, "
        f"topomap has {len(slope_ch_names)}: {slope_ch_names}"
    )

    cov      = make_ad_hoc_cov(raw_for_topomap.info, verbose=False)
    whitener, _ = compute_whitener(cov, raw_for_topomap.info, verbose=False)
    meg_coils   = _concatenate_coils(
        _create_meg_coils(raw_for_topomap.info['chs'], 'accurate')
    )
    head2dev    = invert_transform(dev_to_head_trans)
    incl_idx    = np.where(include_hpis)[0]
    pol_gofs    = np.array([
        _gof_at_fixed_pos(
            slope[ch_i],
            apply_trans(head2dev, hpi_orig_head[pol_i]),
            whitener,
            meg_coils,
        )
        for ch_i, pol_i in zip(incl_idx, tree_indices)
    ])

    # Track state for return dict
    _opt_status = None
    opt_success = None

    if optim == 'rigid':
        # Optimize transform.
        # Nested closures (not module-level) so they see fit_hpi's locals
        # directly: dev_to_head_trans, hpi_orig_head, whitener, meg_coils,
        # incl_idx, tree_indices, slope.
        def mean_gof(params):
            trans = perturb_transform(dev_to_head_trans, params)
            head2dev = invert_transform(trans)
            gofs = np.array([
                _gof_at_fixed_pos(
                    slope[ch_i],
                    apply_trans(head2dev, hpi_orig_head[pol_i]),
                    whitener,
                    meg_coils,
                )
                for ch_i, pol_i in zip(incl_idx, tree_indices)
            ])
            return gofs.mean()

        def objective(params):
            return -mean_gof(params)

        bounds = [
            (-np.deg2rad(5), np.deg2rad(5)),   # rx
            (-np.deg2rad(5), np.deg2rad(5)),   # ry
            (-np.deg2rad(5), np.deg2rad(5)),   # rz
            (-0.005, 0.005),                   # tx 5 mm
            (-0.005, 0.005),                   # ty
            (-0.005, 0.005),                   # tz
        ]
        result = minimize(
            objective,
            x0=np.zeros(6),
            method="L-BFGS-B",
            bounds=bounds,
        )
        opt_status = result.status
        opt_success = result.success and np.all(np.isfinite(result.x)) and np.isfinite(result.fun)
        _opt_status = opt_status

        if opt_success:
            opt_trans = perturb_transform(
                dev_to_head_trans,
                result.x,
            )
            dev_to_head_trans = opt_trans

            raw_for_topomap = raw_orig.copy()
            topo_picks = pick_types(raw_for_topomap.info, meg=True, eeg=False, exclude='bads')
            raw_for_topomap.pick(picks=topo_picks)
            assert slope_ch_names == raw_for_topomap.ch_names and slope.shape[1] == len(slope_ch_names), (
                f"Mismatch between slope column channels and topomap channels "
                f"in refinement rebuild."
            )
            cov      = make_ad_hoc_cov(raw_for_topomap.info, verbose=False)
            whitener, _ = compute_whitener(cov, raw_for_topomap.info, verbose=False)
            meg_coils   = _concatenate_coils(
                _create_meg_coils(raw_for_topomap.info['chs'], 'accurate')
            )
            head2dev    = invert_transform(dev_to_head_trans)
            incl_idx    = np.where(include_hpis)[0]

            # Recompute pol_gofs and dist with the refined transform.
            pol_gofs_refined = np.array([
                _gof_at_fixed_pos(
                    slope[ch_i],
                    apply_trans(head2dev, hpi_orig_head[pol_i]),
                    whitener,
                    meg_coils,
                )
                for ch_i, pol_i in zip(incl_idx, tree_indices)
            ])
            # Recompute dist for the *returned* transform.
            hpi_head_refined = apply_trans(dev_to_head_trans, hpi_dev)
            dist_refined = np.linalg.norm(
                hpi_orig_head[tree_indices] - hpi_head_refined[include_hpis], axis=1
            )

            # Store both the initial and refined residuals.
            dist = dist_refined
            pol_gofs = pol_gofs_refined
        else:
            warnings.warn(
                f"Transform refinement L-BFGS-B did not converge "
                f"(status={opt_status}, success={result.success}, "
                f"x={result.x}). Retaining the initial rigid transform.",
                RuntimeWarning, stacklevel=2,
            )
            # dev_to_head_trans, dist, pol_gofs remain as computed in Stage 5/6.

    return {
        'dev_to_head_trans': dev_to_head_trans,
        'hpi_dev':    hpi_dev,
        'hpi_gofs':   hpi_gofs,
        'hpi_orig':   hpi_orig_head,
        'hpi_names':  hpi_names,
        'nasion':     nasion_head,
        'lpa':        lpa_head,
        'rpa':        rpa_head,
        'pol_info':   pol,
        'extra_pts':  extra_pts_head,
        'eeg_pts':    eeg_pts_head,
        'slope':      slope,
        'slope_ch_names': slope_ch_names,
        'raw_for_topomap': raw_for_topomap,
        'dist':         dist,
        'include_hpis': include_hpis,
        'tree_indices': tree_indices,
        'pol_gofs':     pol_gofs,
        'bads':         bads,
        'bads_fig':     bads_fig,
        'optim':        optim,
        'opt_status':    _opt_status,
        'opt_success':   opt_success,
    }

def compute_fit_diagnostics(fit):
    """Compute all derived diagnostic scalars from a :func:`fit_hpi` result.

    Centralises every calculation that was previously duplicated across
    ``check.py`` display functions (``_print_diagnostics_full`` and
    ``_fill_text_panel_full``).  ``check.py`` should call this once and
    consume the returned dict rather than recomputing anything itself.

    Parameters
    ----------
    fit : dict
        Return value of :func:`fit_hpi`.

    Returns
    -------
    dict with keys:

    ``rot_deg`` : float
        Rotation angle of the device-to-head transform in degrees.
    ``trans_mm`` : float
        Translation magnitude of the device-to-head transform in mm.
    ``mean_res_mm`` : float
        Mean per-coil residual over included coils (mm).
    ``mean_gof`` : float
        Mean per-coil dipole-fit GOF over *all* coils (0-1).
    ``optim`` : str
        Optimization method used by :func:`fit_hpi` for this result
        (``"none"`` or ``"rigid"``).
    ``intercoil_rows`` : list[tuple]
        One entry per coil pair among *included* coils.  Each tuple is
        ``(name_i, name_j, dev_dist_mm, pol_dist_mm, diff_mm)``
        where distances are in mm and names are the full channel names.
        Points are correctly matched: ``dev[k]`` and ``pol[k]`` refer to
        the same physical coil (excluded coils and unmatched polhemus
        points are omitted).
    """

    R = fit['dev_to_head_trans']['trans'][:3, :3]
    t = fit['dev_to_head_trans']['trans'][:3, 3]
    rot_deg  = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    trans_mm = float(np.linalg.norm(t) * 1000)

    dists_mm = np.array(fit['dist']) * 1000
    mean_res_mm = float(np.mean(dists_mm)) if len(dists_mm) else float('nan')

    hpi_gofs_all = np.array(fit['hpi_gofs'])
    mean_gof = float(np.mean(hpi_gofs_all)) if len(hpi_gofs_all) else float('nan')

    # Inter-coil distances — included coils only, polhemus reordered by
    # tree_indices so that dev[k] and orig[k] are the same physical coil.
    hpi_dev      = np.array(fit['hpi_dev'])
    hpi_orig     = np.array(fit['hpi_orig'])
    hpi_names    = fit['hpi_names']
    include_hpis = np.array(fit['include_hpis'])
    tree_indices = np.array(fit['tree_indices'])

    incl_idx  = np.where(include_hpis)[0]
    dev_incl  = hpi_dev[incl_idx]
    orig_incl = hpi_orig[tree_indices]   # matched polhemus points
    n = len(incl_idx)

    intercoil_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            da = float(np.linalg.norm(dev_incl[i]  - dev_incl[j])  * 1000)
            db = float(np.linalg.norm(orig_incl[i] - orig_incl[j]) * 1000)
            intercoil_rows.append((
                hpi_names[incl_idx[i]],
                hpi_names[incl_idx[j]],
                da, db, abs(da - db),
            ))

    return {
        'rot_deg':        rot_deg,
        'trans_mm':       trans_mm,
        'mean_res_mm':    mean_res_mm,
        'mean_gof':       mean_gof,
        'optim':          fit.get('optim', 'none'),
        'intercoil_rows': intercoil_rows,
        'bads':           fit.get('bads', []),
        'bads_fig':       fit.get('bads_fig', []),
    }

def apply_transform(
    datfile: 'str | mne.io.Raw',
    fit_result: dict,
    new_sfreq: float = None,
) -> mne.io.Raw:
    """
    Apply the HPI device-to-head transform to a data file.

    Loads *datfile* (or uses it directly if already a :class:`mne.io.Raw`),
    drops bad/zero-location channels, optionally resamples, embeds the
    digitisation points and ``dev_head_t`` from *fit_result*, and returns the
    modified :class:`mne.io.Raw` object.  The caller is responsible for saving
    — use :func:`save_raw` for the standard filename convention.

    Parameters
    ----------
    datfile : str or mne.io.Raw
        Path to the OPM-MEG data file to transform, or an already-loaded
        :class:`mne.io.Raw` object.
    fit_result : dict
        Result dict from :func:`fit_hpi`.  All position arrays must already
        be in head coordinates (guaranteed when produced by ``fit_hpi``).
    new_sfreq : float, optional
        Target sampling frequency.  The data is resampled only when the
        current ``sfreq`` differs from *new_sfreq*.  If ``None`` (default),
        no resampling is performed.

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

    if isinstance(datfile, mne.io.BaseRaw):
        raw = datfile if datfile.preload else datfile.load_data()
    else:
        raw = mne.io.read_raw_fif(datfile, preload=True)

    if new_sfreq is not None and new_sfreq != raw.info['sfreq']:
        raw.load_data().resample(new_sfreq)

    # Batched drop — dropping channels one at a time reallocates the full
    # preloaded data array on every call, which is costly when there are
    # many bad/zero-location channels (this pipeline can see up to ~100).
    to_drop = list(dict.fromkeys(
        list(raw.info["bads"]) + list(find_zero_location_channels(raw.info))
    ))
    if to_drop:
        raw.drop_channels(to_drop)

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
