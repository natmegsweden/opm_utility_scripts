"""
Unified HPI coregistration entry point.

Handles both single-file and multi-file cases through one script:

* Select one **or many** data files with the file dialog.
* Select the HPI recording and the Polhemus file.
* ``fit_hpi`` is called once to localise the coils.
* If more than one data file is selected a topomap figure is shown.
* ``apply_transform`` is called in a loop over all selected data files.
* Saving is optional (not the default); overwrite behaviour is asked
  separately when saving is enabled.
* All output files use the suffix ``_proc-hpi+ds_raw.fif``.

Usage::

    python -m opm_utility_scripts.hpi.coregister
"""

import os
import tkinter as tk

import matplotlib.pyplot as plt
import mne
import numpy as np

from mne.transforms import apply_trans

from opm_utility_scripts.io import get_boolean, get_file, get_files, get_input
from opm_utility_scripts.viz import plot_3d, plot_hpi_alignment
from ._core import fit_hpi, apply_transform, save_raw

def _output_suffix(datfile: str, new_sfreq: float) -> str:
    """Return the output suffix, including '+ds' only when the file is resampled."""
    info = mne.io.read_info(datfile, verbose='error')
    suffix = '_proc-hpi'
    if int(new_sfreq) != int(info['sfreq']):
        suffix += '+ds'
    return suffix + '_raw.fif'


def main():
    root = tk.Tk()
    root.withdraw()

    datafiles = get_files("Select data file(s)")
    hpifile = get_file("Select HPI file")
    polfile = get_file("Select Polhemus file")

    hpifreq = float(get_input("Enter HPI frequency (Hz):", "33"))
    new_sfreq = float(get_input("Enter downsampling frequency (Hz):", "1000"))
    doSave = get_boolean("Save result to disk? (default: no)")
    overwrite = get_boolean("Overwrite existing files? (default: no)") if doSave else False
    plotResult = get_boolean("Plot alignment? (default: no)")

    root.quit()

    print(f"Data file(s): {datafiles}")
    print(f"HPI file:     {hpifile}")
    print(f"Polhemus file:{polfile}")
    print(f"Frequency:    {hpifreq} Hz")
    print(f"Save:         {doSave}{'  (overwrite)' if overwrite else ''}")
    print(f"Plot:         {plotResult}")

    # ----------------------------------------------------------------
    # Fit HPI coils (shared across all data files)
    # ----------------------------------------------------------------
    fit = fit_hpi(hpifile, polfile, hpifreq)

    hpi_names = fit['hpi_names']
    hpi_dev = fit['hpi_dev']
    hpi_gofs = fit['hpi_gofs']
    hpi_orig = fit['hpi_orig']
    dist = fit['dist']
    slope = fit['slope']
    raw_for_topomap = fit['raw_for_topomap']
    dev_to_head_trans = fit['dev_to_head_trans']

    # ----------------------------------------------------------------
    # Optional topomap (only shown when multiple files were selected)
    # ----------------------------------------------------------------
    if len(datafiles) > 1:
        fig = plt.figure(figsize=(13, 7))
        n_topomap = min(len(hpi_names), 4)
        for i in range(n_topomap):
            tmp = np.reshape(slope[i], (slope[i].size, 1))
            evo = mne.EvokedArray(tmp, raw_for_topomap.info)
            ax = fig.add_subplot(1, n_topomap, i + 1)
            evo.plot_topomap(0.0, ch_type='mag', size=3, res=512, axes=ax, colorbar=False, show=False)
            ax.set_title(hpi_names[i], fontsize=14)
        plt.show()

    # ----------------------------------------------------------------
    # Print fit quality
    # ----------------------------------------------------------------
    print('---------------------------------------------')
    print(f"hpi_orig (head frame, mm):\n{hpi_orig * 1000}\n")
    print(f"hpi_dev  (device frame, mm):\n{hpi_dev * 1000}\n")
    print(f"mean distance = {np.mean(dist) * 1000:.1f} mm\n")
    for index, value in enumerate(hpi_gofs):
        status = 'ok' if value > 0.9 else 'not ok'
        print(f"Coil: {hpi_names[index][-3:]}, GOF: {value:.3f}, Status: {status}")
    print('---------------------------------------------')

    # ----------------------------------------------------------------
    # Apply transform; optionally save
    # ----------------------------------------------------------------
    last_outpath = None
    last_raw_out = None
    last_datfile = None

    for datfile in datafiles:
        suffix = _output_suffix(datfile, new_sfreq)
        raw_out = apply_transform(datfile, fit, new_sfreq)

        if doSave:
            stem = os.path.splitext(os.path.basename(datfile))[0].replace('_raw', '')
            outpath = os.path.join(os.path.dirname(datfile), stem + suffix)
            if not overwrite and os.path.exists(outpath):
                print(f"Skipped (already exists): {outpath}")
            else:
                outpath = save_raw(raw_out, datfile, suffix, overwrite=overwrite)
                print(f"Saved: {outpath}")
                last_outpath = outpath

        last_raw_out = raw_out
        last_datfile = datfile

    # ----------------------------------------------------------------
    # Optional alignment plot
    # ----------------------------------------------------------------
    if plotResult:
        # Use the HPI raw for the sensor cloud (device-space positions).
        raw_hpi_for_plot = mne.io.read_raw_fif(hpifile, preload=False, verbose='error')

        # Derive plot filename from saved output if available, else from
        # the last data file (so the PNG lands next to the source data).
        ref_path = last_outpath if last_outpath is not None else last_datfile
        plot_stem = os.path.splitext(ref_path)[0] if ref_path else 'hpi_alignment'
        plot_filename = f"{plot_stem}_hpi_alignment.png"

        plot_hpi_alignment(
            fit,
            raw=raw_hpi_for_plot,
            show=True,
            filename=plot_filename,
        )
        print(f"Alignment plot saved: {plot_filename}")


if __name__ == '__main__':
    main()
