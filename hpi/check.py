#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check an HPI recording by fitting magnetic dipoles to the detected coil
fields.

Shows goodness-of-fit values and plots the dipole locations alongside the
sensor array — helps verify quickly whether an HPI recording was successful.

Usage::

    python -m opm_utility_scripts.hpi.check

Note
----
``raw.info`` (not ``epochs.info``) is used for the channel lookup at the
``evo.plot_topomap`` call (line ~80 in ``main``).  This is a pre-existing
inconsistency inherited from the original script and is preserved here to
avoid behaviour changes during restructuring.
"""

import tkinter as tk
from tkinter import filedialog

import math

import matplotlib.pyplot as plt
import mne
import numpy as np

from mne._fiff.pick import pick_info
from mne.chpi import _fit_magnetic_dipole
from mne.dipole import _make_guesses
from mne.forward import _concatenate_coils, _create_meg_coils, _magnetic_dipole_field_vec

from opm_utility_scripts.channels import pick_low_noise_meg_chs
from opm_utility_scripts.viz import create_aligned_grid, rotate_points


def main():
    root = tk.Tk()
    root.withdraw()

    f_hpi = 33

    hpi_file = filedialog.askopenfilename()

    raw = mne.io.read_raw_fif(hpi_file)
    raw.info['bads'] = pick_low_noise_meg_chs(raw, n_std=2, fmax=100)
    raw.pick(picks=['meg', 'stim', 'misc'], exclude='bads')
    epochs = (
        mne.make_fixed_length_epochs(raw, duration=0.25, preload=True)
        .filter(h_freq=f_hpi + 5, l_freq=f_hpi - 5)
        .resample(sfreq=200)
    )
    data = epochs.get_data(copy=False)

    mag_channels = mne.pick_types(epochs.info, meg=True)
    misc_channels = mne.pick_types(epochs.info, misc=True)
    hpi_channels = np.zeros((6, 1), dtype=int)
    n_hpi = 0
    for i in misc_channels:
        # NOTE: raw.info used deliberately (pre-existing behaviour — see module docstring).
        if 'hpiin' in raw.info['chs'][i]['ch_name']:
            hpi_channels[n_hpi] = i
            n_hpi += 1

    hpi_trls = np.squeeze(
        (np.max(data[:, hpi_channels, :], axis=3) - np.min(data[:, hpi_channels, :], axis=3)) > 1e-3
    )
    hpi_channels = hpi_channels[np.any(hpi_trls, axis=0)]
    n_hpi = hpi_channels.size
    hpi_trls = hpi_trls[:, np.any(hpi_trls, axis=0)]

    fig = plt.figure(figsize=(13, 7))
    t_amp = np.zeros((n_hpi, data.shape[1]))
    for i_coil in range(n_hpi):
        trls = np.asarray(hpi_trls[:, i_coil]).nonzero()[0][2:-2]
        print("Coil:%s n_trls:%s" % (i_coil, trls.size))
        R = np.zeros((data.shape[1], trls.size))
        Theta = np.zeros((data.shape[1], trls.size))
        for i_trl in range(trls.size):
            X = np.mean(np.multiply(np.cos(2 * math.pi * f_hpi * epochs.times), data[trls[i_trl], :, :]), axis=1)
            Y = np.mean(np.multiply(np.sin(2 * math.pi * f_hpi * epochs.times), data[trls[i_trl], :, :]), axis=1)
            tmp = X + 1j * Y
            R[:, i_trl] = abs(tmp)
            Theta[:, i_trl] = np.angle(tmp / tmp[hpi_channels[i_coil]])
        amp = np.mean(R, axis=1)
        amp[abs(np.mean(Theta, axis=1)) > (math.pi / 2)] = -amp[abs(np.mean(Theta, axis=1)) > (math.pi / 2)]
        amp = np.reshape(amp, (amp.size, 1))
        t_amp[i_coil, :] = np.squeeze(amp)
        evo = mne.EvokedArray(amp, raw.info)
        ax = fig.add_subplot(1, n_hpi, i_coil + 1)
        evo.plot_topomap(0.0, ch_type='mag', size=3, res=512, axes=ax, colorbar=False, show=False)
        # NOTE: raw.info used here (not epochs.info) — pre-existing inconsistency.
        s = raw.info['chs'][np.squeeze(hpi_channels[i_coil])]['ch_name']
        ax.set_title(s[0:3] + s[-3:], fontsize=14)
    t_amp = t_amp[:, mag_channels]

    rrs = np.zeros((n_hpi, 3))
    gofs = np.zeros((n_hpi,))
    moms = np.zeros((n_hpi, 3))

    max_idx = t_amp.argmax(axis=1)
    for i_coil in range(n_hpi):
        loc = raw.info['chs'][mag_channels[max_idx[i_coil]]]['loc']
        grid = create_aligned_grid(loc, 0.005, 0.03, 0.02)

        meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])
        info = pick_info(raw.info, meg_picks)
        meg_coils = _concatenate_coils(_create_meg_coils(info["chs"], "accurate"))
        cov = mne.cov.make_ad_hoc_cov(raw.info)
        whitener, _ = mne.cov.compute_whitener(cov, raw.info)

        R = np.linalg.norm(meg_coils[0], axis=1).max()
        guesses = _make_guesses(dict(R=R, r0=np.zeros(3)), 0.01, 0.0, 0.01)[0]["rr"]

        fwd = _magnetic_dipole_field_vec(guesses, meg_coils, 'warning')
        fwd = np.dot(fwd, whitener.T)
        fwd.shape = (guesses.shape[0], 3, -1)
        fwd = np.linalg.svd(fwd, full_matrices=False)[2]
        guesses = dict(rr=guesses, whitened_fwd_svd=fwd)

        coil_fit = _fit_magnetic_dipole(t_amp[i_coil, :], np.zeros((3,)), 'warning', whitener, meg_coils, guesses)
        t1, t2, t3 = zip(coil_fit)
        rrs[i_coil, :] = t1[0]
        gofs[i_coil] = t2[0]
        moms[i_coil, :] = t2[0]

    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.scatter3D(rrs[:, 0], rrs[:, 1], rrs[:, 2], color="red", marker=".", linewidths=10)
    for i in range(len(rrs)):
        s = raw.info['chs'][np.squeeze(hpi_channels[i])]['ch_name']
        ax.text(rrs[i, 0] + 0.01, rrs[i, 1], rrs[i, 2], s[0:3] + s[-3:], size=10, zorder=1, color='k')
    ax.scatter3D(meg_coils[0][:, 0], meg_coils[0][:, 1], meg_coils[0][:, 2], color="green", alpha=0.1, marker=".", linewidths=0.5)
    ax.view_init(elev=90, azim=-90, roll=0)
    ax.set_title('top')

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.scatter3D(rrs[:, 0], rrs[:, 1], rrs[:, 2], color="red", marker=".", linewidths=10)
    for i in range(len(rrs)):
        s = raw.info['chs'][np.squeeze(hpi_channels[i])]['ch_name']
        ax.text(rrs[i, 0], rrs[i, 1] + 0.01, rrs[i, 2], s[0:3] + s[-3:], size=10, zorder=1, color='k')
    ax.scatter3D(meg_coils[0][:, 0], meg_coils[0][:, 1], meg_coils[0][:, 2], color="green", alpha=0.1, marker=".", linewidths=0.5)
    ax.view_init(elev=0, azim=0, roll=0)
    ax.set_title('right')

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter3D(rrs[:, 0], rrs[:, 1], rrs[:, 2], color="red", marker=".", linewidths=10)
    for i in range(len(rrs)):
        s = raw.info['chs'][np.squeeze(hpi_channels[i])]['ch_name']
        ax.text(rrs[i, 0] - 0.01, rrs[i, 1], rrs[i, 2], s[0:3] + s[-3:], size=10, zorder=1, color='k')
    ax.scatter3D(meg_coils[0][:, 0], meg_coils[0][:, 1], meg_coils[0][:, 2], color="green", alpha=0.1, marker=".", linewidths=0.5)
    ax.view_init(elev=0, azim=90, roll=0)
    ax.set_title('front')
    plt.show()

    for i_coil in range(n_hpi):
        print("Coil %s: gof = %f" % (i_coil, gofs[i_coil]))


if __name__ == '__main__':
    main()
