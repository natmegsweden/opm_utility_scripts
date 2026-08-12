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

from mne._fiff.pick import pick_types
from mne.transforms import apply_trans

from opm_utility_scripts.io import get_boolean, get_file, get_files, get_input
from opm_utility_scripts.viz import plot_3d
from ._core import fit_hpi, apply_transform

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
    # Apply transform to each data file
    # ----------------------------------------------------------------
    last_outpath = None
    for datfile in datafiles:
        outpath = apply_transform(datfile, fit, new_sfreq, _OUTPUT_SUFFIX)
        print(f"Saved: {outpath}")
        last_outpath = outpath

    # ----------------------------------------------------------------
    # Optional 3-D plot (uses the last processed file)
    # ----------------------------------------------------------------
    if plotResult and last_outpath is not None:
        import mne as _mne
        raw_plot = _mne.io.read_raw_fif(last_outpath, preload=False)

        senspos = np.array([], dtype=float)
        picks = pick_types(raw_plot.info, meg='mag')
        for j in picks:
            senspos = np.append(
                senspos,
                apply_trans(dev_to_head_trans, raw_plot.info['chs'][j]['loc'][0:3])
            )
        n = int(senspos.shape[0] / 3)
        senspos = senspos.reshape((n, 3))

        senslabel = []
        for j in picks:
            idx = raw_plot.info['chs'][j]['ch_name'].find('s')
            senslabel.append(raw_plot.info['chs'][j]['ch_name'][idx:] if idx != -1 else '')

        digpts = np.array([], dtype=float)
        for j in raw_plot.info['dig']:
            digpts = np.append(digpts, j['r'])
        n = int(digpts.shape[0] / 3)
        digpts = digpts.reshape((n, 3))

        hpilabel = [str(j + 1) for j in range(len(hpi_names))]

        plot_params = {
            'senspos': senspos,
            'senslabel': senslabel,
            'hpipos': hpi_orig,
            'hpilabel': hpilabel,
            'hpipos2': hpi_head,
            'hpilabel2': hpi_names,
            'digpos': digpts,
        }
        import os
        plot_stem = os.path.splitext(last_outpath)[0]
        plot_3d(plot_params, f"{plot_stem}_hpi_plot.png")


if __name__ == '__main__':
    main()
