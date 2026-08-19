"""File I/O and dialog utilities for OPM-MEG scripts."""

import json
import os
import warnings

import numpy as np


INITIAL_DIR = '/data'

def _get_tk_modules():
    import tkinter as tk
    from tkinter import filedialog, simpledialog

    return tk, filedialog, simpledialog


def write_bw_marker_file(dsName, events, chanName, fs):
    """
    Write a BrainVision Analyzer marker file (.mrk) for MEG data analysis.

    Creates a marker file that contains timing information for events in a
    dataset, formatted according to BrainVision Analyzer specifications.

    Parameters
    ----------
    dsName : str
        Path to the dataset directory where the marker file will be created.
    events : array-like
        Array of event data where each event contains timing information in
        samples.
    chanName : str
        Name of the channel or marker class to be written in the file.
    fs : float
        Sampling frequency in Hz, used to convert sample indices to seconds.

    Returns
    -------
    None
        Creates a 'MarkerFile.mrk' file in the specified dataset directory.
    """
    no_trigs = 1
    filepath = os.path.join(dsName, 'MarkerFile.mrk')

    with open(filepath, 'w') as fid:
        fid.write('PATH OF DATASET:\n')
        fid.write(f'{dsName}\n\n\n')
        fid.write('NUMBER OF MARKERS:\n')
        fid.write(f'{no_trigs}\n\n\n')

        for i in range(no_trigs):
            fid.write('CLASSGROUPID:\n')
            fid.write('3\n')
            fid.write('NAME:\n')
            fid.write(f'{chanName}\n')
            fid.write('COMMENT:\n\n')
            fid.write('COLOR:\n')
            fid.write('blue\n')
            fid.write('EDITABLE:\n')
            fid.write('Yes\n')
            fid.write('CLASSID:\n')
            fid.write(f'{i + 1}\n')
            fid.write('NUMBER OF SAMPLES:\n')
            fid.write(f'{len(events)}\n')
            fid.write('LIST OF SAMPLES:\n')
            fid.write('TRIAL NUMBER\t\tTIME FROM SYNC POINT (in seconds)\n')

            for t in range(len(events) - 1):
                fid.write(f'                  %+g\t\t\t\t               %+0.6f\n' % (0, events[t][0] / fs))

            fid.write(f'                  %+g\t\t\t\t               %+0.6f\n\n\n' % (0, events[t][-1] / fs))


def get_file(title):
    """Open a single-file dialog and return the selected path."""
    _, filedialog, _ = _get_tk_modules()
    return filedialog.askopenfilename(title=title, initialdir=INITIAL_DIR)


def get_files(title):
    """Open a multi-file dialog and return the selected paths as a tuple."""
    _, filedialog, _ = _get_tk_modules()
    return filedialog.askopenfilenames(title=title, initialdir=INITIAL_DIR)


def get_input(prompt, default):
    """Show a text-input dialog and return the entered string."""
    _, _, simpledialog = _get_tk_modules()
    return simpledialog.askstring("Input", prompt, initialvalue=default)


def get_boolean(prompt):
    """Show a yes/no dialog and return a bool.

    Returns False when the user cancels or dismisses the dialog.
    """
    tk, _, simpledialog = _get_tk_modules()
    while True:
        response = simpledialog.askstring("Input", prompt + " (y/n):", initialvalue='n')
        if response is None:  # user cancelled the dialog
            return False
        response = response.lower()
        if response in ['y', 'n']:
            return response == 'y'
        tk.messagebox.showerror("Invalid input", "Please enter 'y' or 'n'.")


def load_datafile(path: str) -> dict:
    """Probe a FIF data file and return basic metadata."""
    import mne

    raw = mne.io.read_raw_fif(path, preload=False, verbose='error')
    return {'sfreq': raw.info['sfreq'], 'path': os.path.abspath(path)}


def load_polhemus(path: 'str | mne.channels.DigMontage') -> dict:
    """Load Polhemus digitisation from JSON, FIF, or a DigMontage object.

    Accepts three input types:

    * **str ending in** ``.json`` — pylhemus-dig/1 JSON format (isotrak frame).
    * **str ending in** ``.fif`` — any FIF whose ``info['dig']`` carries the
      digitisation (head frame, as produced by TRIUX/Elekta recordings).
    * :class:`mne.channels.DigMontage` — as returned by
      ``mne.channels.read_dig_fif()``.  Coordinates are taken directly from
      ``montage.dig`` and treated as head-frame (``source='fif'``), consistent
      with how MNE stores dig points in a DigMontage.

    Parameters
    ----------
    path : str | mne.channels.DigMontage
        Source of the digitisation data.

    Returns
    -------
    dict
        Common polhemus dict with keys ``lpa``, ``nasion``, ``rpa``,
        ``hpi_orig``, ``extra_pts``, ``eeg_pts``, ``dig``, ``source``,
        and ``path``.
    """
    import mne

    # ------------------------------------------------------------------
    # Branch 1: pre-loaded DigMontage
    # ------------------------------------------------------------------
    if isinstance(path, mne.channels.DigMontage):
        montage = path
        if not montage.dig:
            raise ValueError('DigMontage contains no digitisation points')
        dig = [
            {
                'kind': int(d['kind']),
                'ident': int(d['ident']),
                'r': np.array(d['r'], dtype=float),
            }
            for d in montage.dig
        ]
        source = 'fif'   # DigMontage stores coords in head frame
        pol_path = '<DigMontage>'
    else:
        # ------------------------------------------------------------------
        # Branch 2: file path
        # ------------------------------------------------------------------
        pol_path = os.path.abspath(path)
        ext = os.path.splitext(pol_path)[1].lower()

        if ext == '.json':
            with open(pol_path, 'r', encoding='utf-8') as fid:
                pol_data = json.load(fid)
            if pol_data.get('format') != 'pylhemus-dig/1':
                raise ValueError("Unsupported polhemus JSON format; expected 'pylhemus-dig/1'")
            dig = [
                {
                    'kind': int(d['kind']),
                    'ident': int(d['ident']),
                    'r': np.array(d['r'], dtype=float),
                }
                for d in pol_data.get('dig', [])
            ]
            source = 'json'
        elif ext == '.fif':
            pol_info = mne.io.read_info(pol_path, verbose='error')
            if not pol_info['dig']:
                raise ValueError('No digitisation points found in FIF')
            dig = [
                {
                    'kind': int(d['kind']),
                    'ident': int(d['ident']),
                    'r': np.array(d['r'], dtype=float),
                }
                for d in pol_info['dig']
            ]
            source = 'fif'
        else:
            raise ValueError(f'Unsupported polhemus file extension: {ext}')

    cardinals = {d['ident']: np.array(d['r'], dtype=float) for d in dig if d['kind'] == 1}
    for ident in (1, 2, 3):
        if ident not in cardinals:
            raise ValueError(f'Missing fiducial ident {ident} in polhemus {source.upper()}')

    hpi_orig = np.array([d['r'] for d in dig if d['kind'] == 2], dtype=float)
    extra_pts = (
        np.array([d['r'] for d in dig if d['kind'] == 4], dtype=float)
        if any(d['kind'] == 4 for d in dig)
        else np.empty((0, 3), dtype=float)
    )
    eeg_pts = (
        np.array([d['r'] for d in dig if d['kind'] == 3], dtype=float)
        if any(d['kind'] == 3 for d in dig)
        else np.empty((0, 3), dtype=float)
    )

    return {
        'lpa': cardinals[1],
        'nasion': cardinals[2],
        'rpa': cardinals[3],
        'hpi_orig': hpi_orig,
        'extra_pts': extra_pts,
        'eeg_pts': eeg_pts,
        'dig': dig,
        'source': source,
        'path': pol_path,
    }


def make_debug_fif(path: str, out_path: str | None = None, keep_meg: bool | None = None) -> str:
    """Strip a FIF recording down to an anonymised, minimal debug copy.

    Suitable for both the **HPI recording** and the **polhemus FIF** inputs to
    :func:`~opm_utility_scripts.hpi._core.fit_hpi`.  Call it once for each:

    .. code-block:: python

        make_debug_fif(hpi_raw_path)        # writes alongside the source file
        make_debug_fif(polhemus_fif_path)   # same
        # or specify an explicit destination:
        make_debug_fif(hpi_raw_path, '/tmp/debug_hpi_raw.fif')

    What is removed
    ~~~~~~~~~~~~~~~
    * Subject name, date of birth, recording date (zeroed to 2000-01-01).
    * All non-MEG, non-HPI signal channels (STIM, analog/digital inputs, EEG …).
    * MEG channels from a polhemus FIF (no ``hpiout*`` channels present) unless
      *keep_meg* is explicitly ``True``.

    What is kept
    ~~~~~~~~~~~~
    * Digitisation points (``info['dig']``) — fiducials, HPI coil positions,
      headshape points — **all coordinates preserved unchanged**.
    * HPI subsystem metadata (``hpi_meas``, ``hpi_results``, ``hpi_subsystem``).
    * ``hpiout*`` MISC channels — the drive signals used by
      :func:`~opm_utility_scripts.hpi._core.fit_hpi_amplitudes`.
    * MEG channels — always kept for HPI recordings (needed for dipole fitting);
      dropped by default for polhemus FIFs (dig-only use).  Override with
      *keep_meg*.

    Parameters
    ----------
    path : str
        Source FIF recording — either the OPM HPI raw file or a polhemus
        FIF (e.g. a TRIUX recording whose ``info['dig']`` holds the
        digitisation).
    out_path : str | None
        Destination path for the anonymised copy.  When ``None`` (default)
        the output is placed next to the source file with ``_debug``
        inserted before the ``.fif`` suffix, e.g.
        ``hpipre_raw.fif`` → ``hpipre_debug_raw.fif``.
    keep_meg : bool | None
        Whether to retain MEG sensor channels.  ``None`` (default) means
        *auto*: MEG is kept unless the caller passes ``False``.  Pass
        ``False`` explicitly to drop MEG (e.g. for a polhemus-only FIF
        where you only need the dig points).

    Returns
    -------
    str
        Absolute path of the saved file.
    """
    import mne
    from mne.io.constants import FIFF

    path = os.path.abspath(path)

    # --- Derive default output path ---
    if out_path is None:
        base = os.path.basename(path)          # e.g. hpipre_raw.fif
        # Insert _debug before the first .fif occurrence
        if '_raw.fif' in base:
            base_out = base.replace('_raw.fif', '_debug_raw.fif', 1)
        else:
            stem, ext = os.path.splitext(base)
            base_out = stem + '_debug' + ext
        out_path = os.path.join(os.path.dirname(path), base_out)

    raw = mne.io.read_raw_fif(path, preload=True, verbose=False, allow_maxshield=True)

    # --- Anonymise: date → 2000-01-01, wipe subject fields completely ---
    raw.anonymize(daysback=None, keep_his=False, verbose=False)
    with raw.info._unlock():
        raw.info['subject_info'] = None

    # Auto: keep MEG unless explicitly told not to.
    # MEG channels are required for the dipole fit in fit_hpi_amplitudes;
    # pass keep_meg=False only when producing a dig-only polhemus copy.
    _keep_meg = True if keep_meg is None else bool(keep_meg)

    # --- Select channels to keep ---
    keep_kinds = set()
    if _keep_meg:
        keep_kinds.add(FIFF.FIFFV_MEG_CH)

    # Always keep HPI output channels (kind MISC, name starts with 'hpiout')
    keep_names = [
        ch['ch_name'] for ch in raw.info['chs']
        if ch['kind'] in keep_kinds
        or (ch['kind'] == FIFF.FIFFV_MISC_CH
            and ch['ch_name'].lower().startswith('hpiout'))
    ]

    if keep_names:
        raw.pick(keep_names)
        raw.load_data()
    else:
        # Polhemus FIF with keep_meg=False: no signal channels to keep.
        # Write a single zeroed stub channel so the file is valid while
        # still carrying the dig metadata.
        if raw.ch_names:
            raw.pick([raw.ch_names[0]])
            raw.load_data()
            raw._data[:] = 0.0
        # If somehow no channels exist at all, MNE still writes the dig.

    out_path = os.path.abspath(out_path)  # normalise in case caller provided relative path
    raw.save(out_path, overwrite=True, verbose=False)
    return out_path


def load_hpifile(path: str):
    """Load an HPI OPM recording and drop pre-marked bad channels."""
    import mne

    raw = mne.io.read_raw_fif(path, preload=True)
    for ch in list(raw.info['bads']):
        raw.drop_channels(ch)
    return raw


def select_best_hpi_file(hpi_files: list[str], polhemus: dict, hpifreq: float) -> tuple[str, dict]:
    """Fit all HPI candidates and return the highest-scoring path and fit."""
    from .hpi._core import fit_hpi

    best_path = None
    best_fit = None
    best_score = -np.inf
    best_raw_mean = -np.inf
    errors = []

    for path in hpi_files:
        try:
            fit = fit_hpi(path, polhemus, hpifreq)
        except Exception as exc:
            errors.append(f'{path}: {exc}')
            continue

        gofs = np.asarray(fit['hpi_gofs'], dtype=float)
        high_gofs = gofs[gofs > 0.9]
        raw_mean = float(np.mean(gofs)) if gofs.size else -np.inf

        if high_gofs.size:
            score = float(np.mean(high_gofs))
        else:
            score = raw_mean
            warnings.warn(
                f'No HPI coils exceeded GOF 0.9 for {path}; using raw mean GOF {raw_mean:.3f}'
            )

        if score > best_score or (score == best_score and raw_mean > best_raw_mean):
            best_score = score
            best_raw_mean = raw_mean
            best_path = path
            best_fit = fit

    if best_fit is None:
        error_text = '; '.join(errors) if errors else 'No candidate HPI files were provided.'
        raise RuntimeError(f'Could not fit HPI from any HPI file. {error_text}')

    return best_path, best_fit


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            'Anonymise and strip a FIF recording to a minimal debug copy. '
            'Works for both the OPM-HPI raw file and a polhemus FIF. '
            'Run once for each input file.'
        )
    )
    parser.add_argument(
        'path',
        help='Source FIF file (HPI raw or polhemus FIF).',
    )
    parser.add_argument(
        'out_path',
        nargs='?',
        default=None,
        help=(
            'Destination path for the debug copy. '
            'Defaults to the source directory with "_debug" inserted '
            'before the .fif suffix (e.g. hpipre_raw.fif -> hpipre_debug_raw.fif).'
        ),
    )
    meg_group = parser.add_mutually_exclusive_group()
    meg_group.add_argument(
        '--no-meg',
        action='store_true',
        default=False,
        help=(
            'Drop MEG sensor channels. Use for polhemus FIFs where only '
            'the dig points are needed.'
        ),
    )
    meg_group.add_argument(
        '--keep-meg',
        action='store_true',
        default=False,
        help='Explicitly keep MEG channels (default when not using --no-meg).',
    )
    args = parser.parse_args()

    keep_meg = False if args.no_meg else None  # None = auto (keep MEG)
    out = make_debug_fif(args.path, args.out_path, keep_meg=keep_meg)
    print(out)
