"""
Unified HPI coregistration entry point.

Handles both single-file and multi-file cases through one script.
Any argument not supplied on the command line falls back to a GUI dialog,
so the script works fully non-interactively, fully interactively, or anywhere
in between.

Usage (fully non-interactive)::

    python -m opm_utility_scripts.hpi.coregister \\
        --data  /path/to/AudOdd_raw.fif /path/to/RSEO_raw.fif \\
        --hpi   /path/to/HPIBefore_raw.fif \\
        --pol   /path/to/digitisation_sub-001_20260811140000.json \\
        --freq  33 \\
        --sfreq 1000 \\
        --save --overwrite --plot

Usage (fully interactive — GUI dialogs for everything)::

    python -m opm_utility_scripts.hpi.coregister
"""

import argparse
import os
import sys
import tkinter as tk

import matplotlib.pyplot as plt
import mne
import numpy as np

from opm_utility_scripts.io import get_boolean, get_file, get_files, get_input
from opm_utility_scripts.viz import plot_hpi_alignment
from ._core import fit_hpi, apply_transform, save_raw


def _output_suffix(datfile: str, new_sfreq: float) -> str:
    """Return the output suffix, including '+ds' only when the file is resampled."""
    info = mne.io.read_info(datfile, verbose='error')
    suffix = '_proc-hpi'
    if int(new_sfreq) != int(info['sfreq']):
        suffix += '+ds'
    return suffix + '_raw.fif'


def _parse_args():
    p = argparse.ArgumentParser(
        prog='python -m opm_utility_scripts.hpi.coregister',
        description=(
            'HPI coregistration.  Any omitted argument opens a GUI dialog.'
        ),
    )
    p.add_argument(
        '--data', '-d', nargs='+', metavar='FILE',
        help='One or more data files to apply the transform to.',
    )
    p.add_argument(
        '--hpi', '-H', metavar='FILE',
        help='Raw HPI recording (e.g. HPIBefore_raw.fif).',
    )
    p.add_argument(
        '--pol', '-p', metavar='FILE',
        help='Polhemus digitisation file (.json or .fif).',
    )
    p.add_argument(
        '--freq', '-f', type=float, default=None, metavar='HZ',
        help='HPI drive frequency in Hz (default: ask).',
    )
    p.add_argument(
        '--sfreq', '-s', type=float, default=None, metavar='HZ',
        help='Target sampling frequency in Hz (default: ask).',
    )
    p.add_argument(
        '--save', action='store_true', default=False,
        help='Save the transformed file(s) to disk (default: do not save).',
    )
    p.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Overwrite existing output files (default: skip).',
    )
    p.add_argument(
        '--plot', action='store_true', default=False,
        help='Show and save the HPI alignment plot (default: no plot).',
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # ----------------------------------------------------------------
    # Resolve inputs — CLI args take priority; fall back to GUI dialogs
    # ----------------------------------------------------------------
    need_gui = not all([args.data, args.hpi, args.pol,
                        args.freq is not None, args.sfreq is not None])
    root = None
    if need_gui:
        root = tk.Tk()
        root.withdraw()

    datafiles = args.data or get_files("Select data file(s)")
    hpifile   = args.hpi  or get_file("Select HPI file")
    polfile   = args.pol  or get_file("Select Polhemus file")

    hpifreq   = args.freq  if args.freq  is not None \
                else float(get_input("HPI frequency (Hz):", "33"))
    new_sfreq = args.sfreq if args.sfreq is not None \
                else float(get_input("Downsampling frequency (Hz):", "1000"))

    # Save / overwrite / plot: if any of the three flags were given on the CLI,
    # use all CLI values directly.  Only ask interactively when a GUI is open
    # (i.e. at least one file/freq arg was missing) and no flags were provided.
    any_flags = args.save or args.overwrite or args.plot
    if any_flags:
        doSave     = args.save
        overwrite  = args.overwrite
        plotResult = args.plot
    elif root is not None:
        doSave     = get_boolean("Save result to disk? (default: no)")
        overwrite  = get_boolean("Overwrite existing files? (default: no)") if doSave else False
        plotResult = get_boolean("Plot alignment? (default: no)")
    else:
        doSave = overwrite = plotResult = False

    if root is not None:
        root.quit()
        root.destroy()

    if not datafiles or not hpifile or not polfile:
        print("ERROR: data file(s), HPI file, and Polhemus file are all required.")
        sys.exit(1)

    print(f"Data file(s): {datafiles}")
    print(f"HPI file:     {hpifile}")
    print(f"Polhemus:     {polfile}")
    print(f"Frequency:    {hpifreq} Hz")
    print(f"Target sfreq: {new_sfreq} Hz")
    print(f"Save:         {doSave}{'  (overwrite)' if overwrite else ''}")
    print(f"Plot:         {plotResult}")

    # ----------------------------------------------------------------
    # Fit HPI coils (shared across all data files)
    # ----------------------------------------------------------------
    fit = fit_hpi(hpifile, polfile, hpifreq)

    hpi_names       = fit['hpi_names']
    hpi_dev         = fit['hpi_dev']
    hpi_gofs        = fit['hpi_gofs']
    hpi_orig        = fit['hpi_orig']
    dist            = fit['dist']
    slope           = fit['slope']
    raw_for_topomap = fit['raw_for_topomap']

    # ----------------------------------------------------------------
    # Optional topomap (only shown when multiple files were selected)
    # ----------------------------------------------------------------
    if len(datafiles) > 1:
        fig_topo  = plt.figure(figsize=(13, 7))
        n_topomap = min(len(hpi_names), 4)
        for i in range(n_topomap):
            tmp = np.reshape(slope[i], (slope[i].size, 1))
            evo = mne.EvokedArray(tmp, raw_for_topomap.info)
            ax  = fig_topo.add_subplot(1, n_topomap, i + 1)
            evo.plot_topomap(0.0, ch_type='mag', size=3, res=512,
                             axes=ax, colorbar=False, show=False)
            ax.set_title(hpi_names[i], fontsize=14)
        plt.show()

    # ----------------------------------------------------------------
    # Print fit quality
    # ----------------------------------------------------------------
    print('---------------------------------------------')
    print(f"hpi_orig (head frame, mm):\n{np.round(hpi_orig * 1000, 1)}\n")
    print(f"hpi_dev  (device frame, mm):\n{np.round(hpi_dev * 1000, 1)}\n")
    print(f"mean distance = {np.mean(dist) * 1000:.1f} mm\n")
    for index, value in enumerate(hpi_gofs):
        status = 'ok' if value > 0.9 else 'not ok'
        print(f"Coil: {hpi_names[index][-3:]}, GOF: {value:.3f}, Status: {status}")
    print('---------------------------------------------')

    # ----------------------------------------------------------------
    # Apply transform; optionally save
    # ----------------------------------------------------------------
    last_outpath = None
    last_datfile = None
    last_raw_out = None

    for datfile in datafiles:
        suffix  = _output_suffix(datfile, new_sfreq)
        raw_out = apply_transform(datfile, fit, new_sfreq)

        if doSave:
            stem    = os.path.splitext(os.path.basename(datfile))[0].replace('_raw', '')
            outpath = os.path.join(os.path.dirname(datfile), stem + suffix)
            if not overwrite and os.path.exists(outpath):
                print(f"Skipped (already exists): {outpath}")
            else:
                outpath = save_raw(raw_out, datfile, suffix, overwrite=overwrite)
                print(f"Saved: {outpath}")
                last_outpath = outpath

        last_datfile = datfile
        last_raw_out = raw_out

    # ----------------------------------------------------------------
    # Optional alignment plot
    # ----------------------------------------------------------------
    if plotResult and last_raw_out is not None:
        raw_hpi_for_plot = mne.io.read_raw_fif(hpifile, preload=False, verbose='error')

        ref_path  = last_outpath if last_outpath is not None else last_datfile
        plot_stem = os.path.splitext(ref_path)[0] if ref_path else 'hpi_alignment'
        plot_path = f"{plot_stem}_hpi_alignment.png"

        fig = plot_hpi_alignment(fit, raw=raw_hpi_for_plot, show=True)
        #fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        #print(f"Alignment plot saved: {plot_path}")


if __name__ == '__main__':
    main()
