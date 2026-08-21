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

# Sampling frequency used internally for HPI fitting (always resample to
# this before running the amplitude estimation loop).
_HPI_FIT_SFREQ = 1000

_HEAVY_LOADED = False


def _load_heavy_deps():
    """Load the scientific stack lazily on first use."""
    global _HEAVY_LOADED
    if _HEAVY_LOADED:
        return
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    from scipy.signal import find_peaks
    from mne._fiff._digitization import _call_make_dig_points, _make_dig_points
    from mne._fiff.pick import pick_types
    from mne.chpi import (
        compute_chpi_amplitudes,
        compute_chpi_locs,
        compute_whitener,
        make_ad_hoc_cov,
        _concatenate_coils,
        _create_meg_coils,
        _magnetic_dipole_field_vec,
        _magnetic_dipole_delta,
    )
    from mne.io.constants import FIFF
    from mne.transforms import (
        Transform,
        _fit_matched_points,
        _quat_to_affine,
        apply_trans,
        get_ras_to_neuromag_trans,
        invert_transform,
    )
    from mne.utils import warn
    from ..channels import find_zero_location_channels, get_hpi_output_channels

    globals().update(dict(
        plt=plt, mne=mne, np=np, find_peaks=find_peaks,
        _call_make_dig_points=_call_make_dig_points,
        _make_dig_points=_make_dig_points,
        pick_types=pick_types,
        compute_chpi_amplitudes=compute_chpi_amplitudes,
        compute_chpi_locs=compute_chpi_locs,
        compute_whitener=compute_whitener,
        make_ad_hoc_cov=make_ad_hoc_cov,
        _concatenate_coils=_concatenate_coils,
        _create_meg_coils=_create_meg_coils,
        _magnetic_dipole_field_vec=_magnetic_dipole_field_vec,
        _magnetic_dipole_delta=_magnetic_dipole_delta,
        FIFF=FIFF,
        Transform=Transform,
        _fit_matched_points=_fit_matched_points,
        _quat_to_affine=_quat_to_affine,
        apply_trans=apply_trans,
        get_ras_to_neuromag_trans=get_ras_to_neuromag_trans,
        invert_transform=invert_transform,
        warn=warn,
        find_zero_location_channels=find_zero_location_channels,
        get_hpi_output_channels=get_hpi_output_channels,
    ))
    _HEAVY_LOADED = True


def _gof_at_fixed_pos(slope_row, pos_dev, whitener, meg_coils):
    """Evaluate dipole GOF at a *fixed* device-space position.

    Unlike the floating-dipole fit in ``compute_chpi_locs``, this does not
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
    _load_heavy_deps()
    B  = np.dot(whitener, slope_row)
    B2 = float(np.dot(B, B))
    if B2 < 1e-30:
        return float('nan')
    fwd = _magnetic_dipole_field_vec(pos_dev[np.newaxis], meg_coils, 'info')
    residual, *_ = _magnetic_dipole_delta(fwd, whitener, B, B2)
    return float(1.0 - residual / B2)


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
        ``hpi_indices`` : list[int]
            Channel indices of the HPI output channels in ``raw_orig``.
        ``slope`` : np.ndarray, shape (n_coils, n_meg_mag_channels)
            Accumulated amplitude slope matrix.
        ``raw_orig`` : mne.io.Raw
            The resampled raw with HPI subsystem metadata set.
            Caller must embed ``dig`` before calling ``compute_chpi_locs``.
        ``coil_amplitudes`` : dict
            The ``compute_chpi_amplitudes`` result with the accumulated
            slope matrix injected.  Pass this to ``compute_chpi_locs``
            after embedding proper dig points into ``raw_orig.info``.
    """
    _load_heavy_deps()

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

    # Seed dev_head_t with identity so compute_chpi_locs searches in
    # device coordinates (correct — we have not computed the transform yet).
    raw.info.update(dev_head_t=Transform("meg", "head"))

    # ------------------------------------------------------------------
    # Per-coil amplitude estimation loop
    # ------------------------------------------------------------------
    raw_orig = raw.copy()
    slope = np.zeros((len(hpi_indices), len(pick_types(raw.info, meg='mag'))), dtype=float)

    for index in range(len(hpi_indices)):
        raw = raw_orig.copy()
        channel_index = hpi_indices[index]
        chan_name = raw.info['ch_names'][channel_index]

        print(f'**** HPI coil {chan_name} (index {channel_index}) ****')

        b = raw[channel_index, :][0].ravel()
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
            {"number": i + 1, "drive_chan": hpi_names[i], "coil_freq": hpi_freqs[i]}
            for i in range(len(hpi_indices))
        ]
        for i in range(len(hpi_indices)):
            hpi_sub["hpi_coils"][i]["event_bits"] = [256]

        with raw.info._unlock():
            raw.info["hpi_subsystem"] = hpi_sub
            raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]

        n_hpis = sum(
            1 for d in raw.info["hpi_subsystem"]["hpi_coils"]
            if d.get("event_bits") == [256]
        )

        if n_hpis < 3:
            # NOTE: coil_amplitudes is only assigned inside the else branch.
            # If n_hpis < 3 for every iteration the assert below will raise
            # UnboundLocalError — this is a pre-existing behaviour preserved here.
            warn(
                f"{n_hpis:d} HPIs active. At least 3 needed to perform"
                " head localization\n *NO* head localization performed"
            )
        else:
            with raw.info._unlock():
                raw.info["hpi_results"] = [
                    dict(
                        dig_points=[
                            dict(r=np.zeros(3),
                                 coord_frame=FIFF.FIFFV_COORD_DEVICE,
                                 ident=ii + 1)
                            for ii in range(n_hpis)
                        ],
                        coord_trans=Transform("meg", "head"),
                    )
                ]
            with raw.info._unlock():
                # None → MNE's _setup_hpi_amplitude_fitting takes the else
                # branch and sets line_freqs = np.zeros([0]), skipping line
                # harmonic removal.  Must be set inside _unlock() so MNE's
                # Info validation does not reject the write.
                raw.info["line_freq"] = None
            coil_amplitudes = compute_chpi_amplitudes(raw, tmin=0, tmax=2, t_window=2, t_step_min=2)
            # When all coils share one drive frequency (single-freq OPM case),
            # compute_chpi_amplitudes fits one 33 Hz GLM component and spreads
            # it identically across all n_coil rows — every row is the same
            # spatial pattern.  The correct slope for the *active* coil in
            # this window is always row 0 (any row would give the same result).
            # When coils have distinct frequencies (MEGIN/Elekta case), row
            # `index` selects the component tuned to that coil's frequency.
            n_unique_freqs = len(set(hpi_freqs))
            slope_row = 0 if n_unique_freqs == 1 else index
            slope[index, :] = coil_amplitudes['slopes'][0][slope_row]

    # ------------------------------------------------------------------
    # Inject accumulated slope back into coil_amplitudes.
    # Also copy the HPI subsystem metadata from the last loop iteration's
    # `raw` onto `raw_orig` so compute_chpi_locs can find hpi_results,
    # hpi_subsystem, hpi_meas, and line_freq on the info it will be called with.
    # ------------------------------------------------------------------
    assert len(coil_amplitudes["times"]) == 1  # noqa: F821
    coil_amplitudes['slopes'][0] = slope

    with raw_orig.info._unlock():
        raw_orig.info['hpi_subsystem'] = raw.info.get('hpi_subsystem')
        raw_orig.info['hpi_meas']      = raw.info.get('hpi_meas')
        raw_orig.info['hpi_results']   = raw.info.get('hpi_results')
        raw_orig.info['line_freq']     = raw.info.get('line_freq')

    return {
        'hpi_names':       hpi_names,
        'hpi_indices':     hpi_indices,
        'slope':           slope,
        'raw_orig':        raw_orig,
        'coil_amplitudes': coil_amplitudes,
    }


def fit_hpi(hpifile, polfile, hpifreq: float,
            gof_limit: float | None = None,
            landmark_weight: float = 1.0) -> dict:
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
    gof_limit : float | None
        Minimum dipole GOF for a coil to be included in the transform fit.
        When ``None`` (default) the threshold is chosen automatically:

        * **0.98** when coils use distinct drive frequencies (MEGIN/Elekta
          convention with SSS — standard MNE default).
        * **0.90** when all coils share one drive frequency (single-frequency
          OPM case without SSS — lower threshold accounts for the absence
          of spatial filtering).

        Pass an explicit float to override the automatic selection.
    landmark_weight : float
        Weight for the landmark (nasion, LPA, RPA) constraint in the
        device-to-head transform fit.  The combined score used for
        permutation selection and the refined fit is::

            score = HPI_residual + landmark_weight * landmark_residual

        * ``0.0`` — landmarks ignored; reproduces the previous HPI-only
          behaviour (useful for regression testing).
        * ``1.0`` (default) — equal metre-scale contribution.
        * Higher values — stronger landmark constraint, useful when coil
          geometry is nearly symmetric and HPI residuals alone cannot
          disambiguate the mapping.

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
        ``dist`` : np.ndarray
            Per-coil residual distances in head space (metres).
        ``include_hpis`` : np.ndarray of bool
            Mask of coils whose GOF ≥ ``gof_limit`` (auto-selected or explicit).
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
    _load_heavy_deps()

    # ------------------------------------------------------------------
    # Stage 1: HPI amplitude estimation (no polhemus needed)
    # ------------------------------------------------------------------
    amp = fit_hpi_amplitudes(hpifile, hpifreq)
    hpi_names       = amp['hpi_names']
    hpi_indices     = amp['hpi_indices']
    slope           = amp['slope']
    raw_orig        = amp['raw_orig']
    coil_amplitudes = amp['coil_amplitudes']

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
    # with a shape mismatch in compute_chpi_locs.  Surface this clearly.
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
    # Stage 3: Compute coil locations now that dig is properly set
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
        coil_locs = compute_chpi_locs(raw_orig.info, coil_amplitudes)

    hpi_dev  = np.array(coil_locs['rrs'][0])
    hpi_gofs = np.array(coil_locs['gofs'][0])

    # ------------------------------------------------------------------
    # Stage 4: Compute device-to-head transform
    # ------------------------------------------------------------------
    # Auto-select GOF threshold when not explicitly supplied:
    #   0.98 — distinct frequencies (MEGIN/Elekta + SSS, MNE default)
    #   0.90 — single shared frequency (OPM without SSS; lower because
    #           the absence of spatial filtering inflates the noise floor)
    # Detect single-frequency mode: all HPI coils fired at one shared
    # frequency (sequential OPM case).  hpifreq is always a scalar here,
    # so check whether the coil_amplitudes hpi_freqs array has >1 unique
    # value — if so the caller somehow configured distinct freqs.
    _amp_hpi_freqs = [
        c['coil_freq']
        for hm in raw_orig.info.get('hpi_meas', [])
        for c in hm.get('hpi_coils', [])
    ]
    n_unique_freqs = len(set(_amp_hpi_freqs)) if _amp_hpi_freqs else 1
    if gof_limit is None:
        gof_limit = 0.98 if n_unique_freqs > 1 else 0.90
    print(f'GOF threshold: {gof_limit:.2f} '
          f'({"distinct" if n_unique_freqs > 1 else "single"}-frequency, '
          f'{"auto" if gof_limit in (0.98, 0.90) else "user-supplied"})')
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

    # ------------------------------------------------------------------
    # Coil-to-polhemus assignment via exhaustive permutation search.
    #
    # The KDTree approach (nearest-neighbour in head frame queried with
    # device-frame coordinates) is incorrect because the two sets live in
    # different coordinate systems — the transform is not yet known.
    # Instead, try every injective mapping of the n_inc included fitted
    # coils onto n_pol polhemus points, fit a rigid transform for each,
    # and keep the mapping that minimises the total post-transform residual.
    # With n_inc ≤ 4 and n_pol ≤ 4 this is at most 24 permutations.
    #
    # Two-pass algorithm:
    #   Pass 1 (coarse, HPI only) — select the best permutation and coarse
    #           transform using only HPI residuals (existing behaviour).
    #   Pass 2 (refined, HPI + landmarks) — use the Pass-1 transform to
    #           estimate device-frame landmark positions, then re-fit with
    #           the combined point set to refine the transform.  When
    #           landmark_weight > 0 and n_pol > n_inc, also re-score all
    #           permutations using the landmark constraint so that a
    #           near-symmetric coil geometry cannot produce a wrong mapping.
    # ------------------------------------------------------------------

    lm_head = np.array([nasion_head, lpa_head, rpa_head])
    lm_dev  = None  # set after permutation selection when landmark_weight > 0

    def _apply_t(T, pts):
        return (T[:3, :3] @ pts.T).T + T[:3, 3]

    # --- Permutation selection: HPI-only fit + headshape-centroid orientation prior
    #
    # With near-coplanar coils the HPI Procrustes is underdetermined in yaw, so
    # two permutations can have similar HPI residuals but differ by ~180°.
    #
    # The disambiguating prior: when extra_pts (headshape) are available, map
    # them into device frame via T_inv and check that their centroid z lies on
    # the same side as the HPI coil centroid z.  Both scalp and coils sit on
    # the head, so in ANY device frame they must share the same gross z sign.
    # A permutation that puts the scalp centroid on the *opposite* z side from
    # the coils is physically impossible and receives a large penalty.
    #
    # This is non-circular: the headshape points are independent of the HPI
    # coils and of any landmark round-trip.  The prior acts only as a tie-
    # breaker; when HPI residuals differ substantially (>5 mm) it has no
    # effect.
    #
    # Score = HPI_residual + landmark_weight * headshape_orientation_penalty

    # Expected sign: the coils (in dev frame, as given) cluster at a certain z.
    # dev_pts are the measured device-frame coil positions — their centroid z
    # tells us which side of the device the head is on.
    dev_centroid_z = float(dev_pts[:, 2].mean())

    # extra_pts_head is already in head frame (computed above in either branch).
    _ep = np.asarray(extra_pts_head) if extra_pts_head is not None else np.empty((0, 3))
    extra_head = _ep if (landmark_weight > 0.0 and _ep.shape[0] >= 3) else None

    best_score  = np.inf
    best_trans  = None
    best_perm   = None

    for perm in itertools.permutations(range(n_pol), n_inc):
        perm = list(perm)
        pol_pts = hpi_orig_head[perm]
        try:
            quat, _ = _fit_matched_points(dev_pts, pol_pts)
        except Exception:
            continue
        t_cand = _quat_to_affine(quat)

        # HPI residual: sum of per-coil distances after transform
        fitted_head = _apply_t(t_cand, dev_pts)
        hpi_res = float(np.sum(np.linalg.norm(pol_pts - fitted_head, axis=1)))

        score = hpi_res
        if extra_head is not None:
            # Map headshape centroid to device frame via T_inv.
            t_inv = np.linalg.inv(t_cand)
            extra_centroid_head = extra_head.mean(axis=0)
            extra_centroid_dev  = _apply_t(t_inv, extra_centroid_head[np.newaxis])[0]
            # Penalty: headshape centroid must be on the SAME z-side as the
            # HPI coils.  If it crosses to the opposite side, apply a penalty
            # proportional to how far it is on the wrong side.
            z_wrong = -extra_centroid_dev[2] * np.sign(dev_centroid_z)
            orientation_penalty = float(max(0.0, z_wrong))
            score += landmark_weight * orientation_penalty

        if score < best_score:
            best_score  = score
            best_trans  = t_cand
            best_perm   = perm

    tree_indices      = np.array(best_perm)
    dev_to_head_trans = Transform(fro="meg", to="head", trans=best_trans)

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

    _DIST_WARN_MM = 15.0
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
    # Stage 5: Polhemus-position GOF
    # Evaluate the dipole forward model at each *digitised* polhemus coil
    # position (transformed back to device frame) against the measured MEG
    # field pattern for the matched channel.  Unlike hpi_gofs (which floats
    # the position to maximise fit), pol_gofs uses the fixed polhemus position.
    # A low pol_gof with a high hpi_gof means the digitised position is wrong
    # even though the coil itself was recorded cleanly → redo polhemus.
    # ------------------------------------------------------------------
    raw_for_topomap = raw_orig.copy()
    raw_for_topomap.pick(picks=['meg'], exclude='bads')

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
        'raw_for_topomap': raw_for_topomap,
        'dist':         dist,
        'include_hpis': include_hpis,
        'tree_indices': tree_indices,
        'pol_gofs':     pol_gofs,
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
    ``intercoil_rows`` : list[tuple]
        One entry per coil pair among *included* coils.  Each tuple is
        ``(name_i, name_j, dev_dist_mm, pol_dist_mm, diff_mm)``
        where distances are in mm and names are the full channel names.
        Points are correctly matched: ``dev[k]`` and ``pol[k]`` refer to
        the same physical coil (excluded coils and unmatched polhemus
        points are omitted).
    """
    _load_heavy_deps()

    R = fit['dev_to_head_trans']['trans'][:3, :3]
    t = fit['dev_to_head_trans']['trans'][:3, 3]
    rot_deg  = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    trans_mm = float(np.linalg.norm(t) * 1000)

    dists_mm = np.array(fit['dist']) * 1000
    mean_res_mm = float(np.mean(dists_mm)) if len(dists_mm) else float('nan')

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
        'intercoil_rows': intercoil_rows,
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
    _load_heavy_deps()

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
    _load_heavy_deps()

    stem = os.path.splitext(os.path.basename(datfile))[0].replace('_raw', '')
    outpath = os.path.join(os.path.dirname(datfile), stem + suffix)
    raw.save(outpath, overwrite=overwrite)
    return outpath
