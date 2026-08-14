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


def load_polhemus(path: str) -> dict:
    """Load Polhemus digitisation from JSON or FIF into a common dict."""
    import mne

    path = os.path.abspath(path)
    ext = os.path.splitext(path)[1].lower()

    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as fid:
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
        pol_info = mne.io.read_info(path, verbose='error')
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
        'path': path,
    }


def load_hpifile(path: str):
    """Load an HPI OPM recording and drop pre-marked bad channels."""
    import mne

    raw = mne.io.read_raw_fif(path, preload=True)
    for ch in list(raw.info['bads']):
        raw.drop_channels(ch)
    return raw


def select_best_hpi_file(hpi_files: list[str], polhemus: dict, hpifreq: float) -> tuple[str, dict]:
    """Fit all HPI candidates and return the highest-scoring path and fit."""
    from opm_utility_scripts.hpi._core import fit_hpi

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
