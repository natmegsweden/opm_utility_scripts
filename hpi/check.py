#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check an HPI recording by fitting magnetic dipoles to the detected coil
fields.

Shows goodness-of-fit values and plots the dipole locations alongside the
sensor array — helps verify quickly whether an HPI recording was successful.

Usage::

    python -m opm_utility_scripts.hpi.check
    python -m opm_utility_scripts.hpi.check --file /path/to/HPIbefore_raw.fif
    python -m opm_utility_scripts.hpi.check --freq 33

The script expects a *raw* HPI recording (e.g. ``HPIbefore_raw.fif``), not a
processed output file.  Processed files (``proc-hpi``) do not contain the
``hpiin`` drive-signal channels required for coil detection.
"""

import argparse
import sys
import tkinter as tk
from tkinter import filedialog

import math

import matplotlib.pyplot as plt
import mne
import numpy as np

from mne._fiff.pick import pick_info  # used for dipole-fit sensor info
from mne.chpi import _fit_magnetic_dipole
from mne.dipole import _make_guesses
from mne.bem import ConductorModel
from mne.forward import _concatenate_coils, _create_meg_coils, _magnetic_dipole_field_vec

from opm_utility_scripts.channels import pick_low_noise_meg_chs


def _parse_args():
    parser = argparse.ArgumentParser(
        prog='python -m opm_utility_scripts.hpi.check',
        description=(
            'Check an HPI recording by fitting magnetic dipoles to the detected '
            'coil fields and plotting the result. Pass a raw HPI file '
            '(e.g. HPIbefore_raw.fif), NOT a proc-hpi output.'
        ),
    )
    parser.add_argument(
        '--file', '-f', metavar='PATH',
        help='Path to the raw HPI .fif file. Opens a GUI dialog when omitted.',
    )
    parser.add_argument(
        '--freq', type=float, default=33.0, metavar='HZ',
        help='HPI drive frequency in Hz (default: 33).',
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    f_hpi = args.freq

    if args.file:
        hpi_file = args.file
    else:
        root = tk.Tk()
        root.withdraw()
        hpi_file = filedialog.askopenfilename(
            initialdir='/data', title='Select HPI file',
            filetypes=[('FIF files', '*.fif')],
        )

    if not hpi_file:
        print('No file selected. Exiting.')
        sys.exit(0)

    raw = mne.io.read_raw_fif(hpi_file)
    raw.info['bads'] = pick_low_noise_meg_chs(raw, n_std=2, fmax=100)
    raw.pick(picks=['meg', 'stim', 'misc'], exclude='bads')
    # Filter on the continuous raw so the signal is long enough for the FIR
    # filter (avoids distortion warnings from short epoch windows).
    raw.load_data().filter(h_freq=f_hpi + 5, l_freq=f_hpi - 5)
    epochs = (
        mne.make_fixed_length_epochs(raw, duration=0.25, preload=True)
        .resample(sfreq=200)
    )
    data = epochs.get_data(copy=False)

    mag_channels = mne.pick_types(epochs.info, meg=True)
    misc_channels = mne.pick_types(epochs.info, misc=True)

    # Collect all hpiin drive channels present in the file (up to 4).
    hpi_channels = np.array([
        i for i in misc_channels
        if 'hpiin' in epochs.info['chs'][i]['ch_name']
    ])
    n_hpi = len(hpi_channels)

    if n_hpi == 0:
        print(
            'ERROR: No active HPI drive channels (hpiin*) found in this file.\n'
            'This script requires a raw HPI recording, not a processed output.\n'
            f'File: {hpi_file}'
        )
        sys.exit(1)

    # Per-epoch, per-coil activity flag: True when the drive signal swings > 1 mV.
    hpi_trls = (
        np.max(data[:, hpi_channels, :], axis=2)
        - np.min(data[:, hpi_channels, :], axis=2)
    ) > 1e-3  # shape (n_epochs, n_hpi)

    # --- Figure 1: HPI amplitude map per coil (device-space scatter) ---
    # plot_topomap requires a head transform that doesn't exist in a raw HPI
    # recording (the transform is what we're about to compute).  Instead,
    # project sensor positions from device coordinates (loc[:3]) top-down
    # (x → right, y → up when viewed from above) and colour each sensor by
    # its signed HPI amplitude.
    #
    # Use only _bz channels so each sensor slot appears once.
    bz_mask = np.array([
        epochs.info['chs'][idx]['ch_name'].endswith('_bz')
        for idx in mag_channels
    ])
    if bz_mask.sum() == 0:          # fallback for non-standard naming
        bz_mask = np.ones(len(mag_channels), dtype=bool)
    bz_indices = mag_channels[bz_mask]   # indices into epoch channel list

    # Sensor positions in device space (metres).
    bz_pos = np.array([
        epochs.info['chs'][idx]['loc'][:3] for idx in bz_indices
    ])  # shape (n_bz, 3)

    fig1, axes1 = plt.subplots(1, n_hpi, figsize=(4 * n_hpi, 4),
                                constrained_layout=True)
    if n_hpi == 1:
        axes1 = [axes1]

    t_amp = np.zeros((n_hpi, data.shape[1]))
    for i_coil in range(n_hpi):
        trls = np.nonzero(hpi_trls[:, i_coil])[0][2:-2]  # trim first/last 2
        s = epochs.info['chs'][hpi_channels[i_coil]]['ch_name']
        print("Coil %s (%s): %d trials" % (i_coil, s, len(trls)))
        ax = axes1[i_coil]

        if trls.size == 0:
            ax.set_facecolor('#eeeeee')
            ax.text(0.5, 0.5, 'no trials', ha='center', va='center',
                    transform=ax.transAxes, color='gray', fontsize=12)
            ax.set_title(s[0:3] + s[-3:] + ' [MISSING]', fontsize=14, color='gray')
            continue

        R_mat = np.zeros((data.shape[1], trls.size))
        Theta = np.zeros((data.shape[1], trls.size))
        for i_trl in range(trls.size):
            X = np.mean(np.multiply(np.cos(2 * math.pi * f_hpi * epochs.times), data[trls[i_trl], :, :]), axis=1)
            Y = np.mean(np.multiply(np.sin(2 * math.pi * f_hpi * epochs.times), data[trls[i_trl], :, :]), axis=1)
            tmp = X + 1j * Y
            R_mat[:, i_trl] = abs(tmp)
            Theta[:, i_trl] = np.angle(tmp / tmp[hpi_channels[i_coil]])
        amp_all = np.mean(R_mat, axis=1)
        amp_all[abs(np.mean(Theta, axis=1)) > (math.pi / 2)] = \
            -amp_all[abs(np.mean(Theta, axis=1)) > (math.pi / 2)]
        t_amp[i_coil, :] = amp_all

        # Scatter: colour by amplitude at _bz channels, viewed from above (x/y plane).
        amp_bz = amp_all[bz_indices]
        clim = np.max(np.abs(amp_bz)) or 1.0
        sc = ax.scatter(bz_pos[:, 0] * 1000, bz_pos[:, 1] * 1000,
                        c=amp_bz, cmap='RdBu_r', s=40,
                        vmin=-clim, vmax=clim)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect('equal')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(s[0:3] + s[-3:], fontsize=14)

    t_amp = t_amp[:, mag_channels]

    # --- Dipole fit ---
    rrs = np.zeros((n_hpi, 3))
    gofs = np.zeros((n_hpi,))
    moms = np.zeros((n_hpi, 3))

    # Build sensor geometry once — it is the same for every coil.
    meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])
    info_raw_meg = pick_info(raw.info, meg_picks)
    meg_coils = _concatenate_coils(_create_meg_coils(info_raw_meg["chs"], "accurate"))
    cov = mne.cov.make_ad_hoc_cov(raw.info)
    whitener, _ = mne.cov.compute_whitener(cov, raw.info)

    # Use the median sensor-origin distance as the sphere radius.
    # meg_coils[0] contains coil integration points, some of which can be at
    # the origin (giving min=0 and an empty guess space after mindist pruning).
    # The median of sensor origin norms is a stable estimate of the helmet radius.
    sensor_origins = np.array([ch['loc'][:3] for ch in info_raw_meg['chs']])
    R_sphere = float(np.median(np.linalg.norm(sensor_origins, axis=1)))
    sphere = ConductorModel(layers=[dict(rad=R_sphere)], r0=np.zeros(3), is_sphere=True)
    MINDIST = 0.005  # metres — exclude guesses within 5 mm of the origin
    guesses_rr = _make_guesses(sphere, 0.01, 0.0, MINDIST)[0]["rr"]
    # _make_guesses excludes points within mindist of the sphere *surface* but
    # can still return points near the origin.  Drop those explicitly so
    # _magnetic_dipole_field_vec doesn't produce NaN rows that break the SVD.
    guesses_rr = guesses_rr[np.linalg.norm(guesses_rr, axis=1) >= MINDIST]
    fwd = _magnetic_dipole_field_vec(guesses_rr, meg_coils, 'warning')
    fwd = np.dot(fwd, whitener.T)
    fwd.shape = (guesses_rr.shape[0], 3, -1)
    # Drop any remaining NaN/Inf rows (degenerate dipole positions) before SVD.
    valid = np.isfinite(fwd).all(axis=(1, 2))
    guesses_rr = guesses_rr[valid]
    fwd = fwd[valid]
    fwd = np.linalg.svd(fwd, full_matrices=False)[2]
    guesses = dict(rr=guesses_rr, whitened_fwd_svd=fwd)

    for i_coil in range(n_hpi):
        if np.all(t_amp[i_coil, :] == 0):
            # No active trials for this coil — leave rrs/gofs at zero.
            continue
        x, gof, moment = _fit_magnetic_dipole(
            t_amp[i_coil, :], np.zeros((3,)), 'warning', whitener, meg_coils, guesses
        )
        rrs[i_coil, :] = x
        gofs[i_coil] = gof
        moms[i_coil, :] = moment

    # --- Figure 2: 3-D dipole positions vs sensor array ---
    # Convert metres → mm for readable axis labels.
    rrs_mm = rrs * 1000
    coil_pos_mm = meg_coils[0][:, :3] * 1000  # integration point positions only

    # Shared axis limits so all three views use the same scale.
    all_pts = np.vstack([rrs_mm, coil_pos_mm])
    lim = np.max(np.abs(all_pts)) * 1.1
    ax_lim = [-lim, lim]

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 5),
                                subplot_kw=dict(projection='3d'))
    views = [
        dict(title='top',   elev=90,  azim=-90, roll=0),
        dict(title='right',  elev=0,   azim=0,   roll=0),
        dict(title='front',  elev=0,   azim=90,  roll=0),
    ]
    offsets = [
        np.array([0.01,  0,     0]) * lim * 0.1,
        np.array([0,     0.01,  0]) * lim * 0.1,
        np.array([-0.01, 0,     0]) * lim * 0.1,
    ]

    active_mask = ~np.all(t_amp[:, :] == 0, axis=1)  # coils with fitted positions

    for ax, view, off in zip(axes2, views, offsets):
        ax.scatter(coil_pos_mm[:, 0], coil_pos_mm[:, 1], coil_pos_mm[:, 2],
                   color='green', alpha=0.15, marker='.', s=4)
        # Active coils: red; missing coils: grey at origin (will be labelled)
        if active_mask.any():
            ax.scatter(rrs_mm[active_mask, 0], rrs_mm[active_mask, 1], rrs_mm[active_mask, 2],
                       color='red', marker='o', s=60, zorder=5)
        for i in range(n_hpi):
            s = epochs.info['chs'][hpi_channels[i]]['ch_name']
            label = s[0:3] + s[-3:]
            if active_mask[i]:
                ax.text(rrs_mm[i, 0] + off[0], rrs_mm[i, 1] + off[1], rrs_mm[i, 2] + off[2],
                        f'{label}\n{gofs[i]:.2f}', size=9, zorder=6, color='k')
            else:
                ax.text(off[0], off[1], off[2],
                        f'{label}\n[no trials]', size=9, zorder=6, color='gray')
        ax.set_xlim(ax_lim)
        ax.set_ylim(ax_lim)
        ax.set_zlim(ax_lim)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')
        ax.view_init(elev=view['elev'], azim=view['azim'], roll=view['roll'])
        ax.set_title(view['title'])

    fig2.suptitle('HPI dipole positions (red) vs sensors (green)', fontsize=12)
    plt.tight_layout()
    plt.show()

    print('\nCoil GOF summary:')
    for i_coil in range(n_hpi):
        s = epochs.info['chs'][hpi_channels[i_coil]]['ch_name']
        if active_mask[i_coil]:
            print("  %s: gof = %.3f" % (s, gofs[i_coil]))
        else:
            print("  %s: no active trials" % s)


if __name__ == '__main__':
    main()
