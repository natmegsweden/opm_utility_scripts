"""
Unified HPI coregistration entry point.

Handles both single-file and multi-file cases through one script:

* Select one **or many** data files with the file dialog.
* Select the HPI recording and the Polhemus file.
* ``fit_hpi`` is called once to localise the coils.
* If more than one data file is selected a topomap figure is shown.
* ``apply_transform`` is called in a loop over all selected data files.
* All output files use the suffix ``_proc-hpi+ds_raw.fif``.

Usage::

    python -m opm_utility_scripts.hpi.coregister
"""

import tkinter as tk

import matplotlib.pyplot as plt
import mne
import numpy as np

from mne.transforms import apply_trans

from opm_utility_scripts.io import get_boolean, get_file, get_files, get_input
from opm_utility_scripts.viz import plot_3d, plot_hpi_alignment
from ._core import fit_hpi, apply_transform, save_raw

_OUTPUT_SUFFIX = '_proc-hpi+ds_raw.fif'


def main():
    root = tk.Tk()
    root.withdraw()

    datafiles = get_files("Select data file(s)")
    hpifile = get_file("Select HPI file")
    polfile = get_file("Select Polhemus file")

    hpifreq = float(get_input("Enter HPI frequency (Hz):", "33"))
    new_sfreq = float(get_input("Enter downsampling frequency (Hz):", "1000"))
    plotResult = get_boolean("Do you want to plot the result?")

    root.quit()

    print(f"Data file(s): {datafiles}")
    print(f"HPI file:     {hpifile}")
    print(f"Polhemus file:{polfile}")
    print(f"Frequency:    {hpifreq} Hz")
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
    hpi_head = apply_trans(dev_to_head_trans, hpi_dev)

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
    print(f"hpi_orig:\n{hpi_orig}\n")
    print(f"hpi_dev:\n{hpi_dev}\n")
    print(f"mean distance = {np.mean(dist) * 1000:.1f} mm\n")
    for index, value in enumerate(hpi_gofs):
        status = 'ok' if value > 0.9 else 'not ok'
        print(f"Coil: {hpi_names[index][-3:]}, GOF: {value:.3f}, Status: {status}")
    print('---------------------------------------------')

    # ----------------------------------------------------------------
    # Apply transform to each data file, then save
    # ----------------------------------------------------------------
    import os
    last_outpath = None
    for datfile in datafiles:
        raw_out = apply_transform(datfile, fit, new_sfreq)
        outpath = save_raw(raw_out, datfile, _OUTPUT_SUFFIX)
        print(f"Saved: {outpath}")
        last_outpath = outpath

    # ----------------------------------------------------------------
    # Optional alignment plot
    # ----------------------------------------------------------------
    if plotResult:
        # Use the HPI raw for the sensor cloud (device-space positions).
        raw_hpi_for_plot = mne.io.read_raw_fif(hpifile, preload=False, verbose='error')
        plot_stem = (
            os.path.splitext(last_outpath)[0]
            if last_outpath is not None
            else os.path.splitext(hpifile)[0]
        )
        plot_hpi_alignment(
            fit,
            raw=raw_hpi_for_plot,
            show=True,
            filename=f"{plot_stem}_hpi_alignment.png",
        )


if __name__ == '__main__':
    main()
