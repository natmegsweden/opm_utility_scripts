#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: C Pfeiffer

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import mne
import math
import scipy.signal as sig

from mne.forward import _concatenate_coils, _create_meg_coils, _magnetic_dipole_field_vec
from mne._fiff.pick import pick_info
from mne.dipole import _make_guesses
from mne.chpi import _fit_magnetic_dipole

def pick_low_noise_meg_chs(raw, n_std=2, fmax=150):
    '''Pick channels whose noise is within N std of the mean.'''
    # Adapted from Fieldline hpi script
    raw_copy = raw.copy()
    raw_copy.pick_types(meg=True);
    data = raw_copy.get_data()
    fs = raw_copy.info['sfreq']

    f, Pxx = sig.welch(data, fs=fs, nperseg=fs//2, average='median')
    Axx = np.sqrt(Pxx)

    fidx = np.where(f < fmax)[0]
    noise = np.min(Axx[:,fidx], axis=1)
    noise_u = np.mean(noise)
    noise_s = np.std(noise)

    quiet_idx = np.where(noise < noise_u + n_std*noise_s)[0]
    noisy_chs = [ n for i,n in enumerate(raw_copy.info['ch_names']) if i not in quiet_idx ]

    if len(noisy_chs) > 0:
        print(f'Discarding {len(noisy_chs)} noisy channels: {" ".join(noisy_chs)}')

    return noisy_chs

def rotate_points(points, target_vector):
    # Normalize the target vector
    target_vector = target_vector / np.linalg.norm(target_vector)
    
    # Define the original vector (0, 0, 1)
    original_vector = np.array([0, 0, 1])
    
    # Check if the original vector and target vector are the same
    if np.allclose(original_vector, target_vector):
        return points  # No rotation needed
    
    # Calculate the rotation axis (cross product of original and target vectors)
    rotation_axis = np.cross(original_vector, target_vector)
    
    # Calculate the angle between the original and target vectors
    angle = np.arccos(np.dot(original_vector, target_vector))
    
    # Normalize the rotation axis
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Create the rotation matrix using Rodrigues' rotation formula
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Rotate the points
    rotated_points = points @ R.T
    
    return rotated_points

def create_aligned_grid(loc, step_size, distXY, distZ):
    # Extract position and orientation from loc
    position = np.array(loc[:3])
    orientation_z = np.array(loc[9:])

    # Create the grid points
    x = np.arange(-distXY*2, distXY*2, step_size)
    y = np.arange(-distXY*2, distXY*2, step_size)
    z = -np.arange(0, distZ, step_size)
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Flatten the grid points
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    grid_points = grid_points[np.linalg.norm(grid_points[:,:2],axis=1)<distXY,:]
    
    # Align the grid to the given location and orientation
    aligned_grid_points = rotate_points(grid_points, orientation_z) + position
    
    return aligned_grid_points

root = tk.Tk()
root.withdraw()

f_hpi = 33

hpi_file = filedialog.askopenfilename()

raw = mne.io.read_raw_fif(hpi_file)
raw.info['bads'] = pick_low_noise_meg_chs(raw, n_std=2, fmax=100)
raw.pick(picks =['meg', 'stim', 'misc'] , exclude='bads')
epochs = mne.make_fixed_length_epochs(raw, duration=0.25, preload=True).filter(h_freq=f_hpi+5, l_freq=f_hpi-5).resample(sfreq=200)
data = epochs.get_data(copy=False)

mag_channels = mne.pick_types(epochs.info, meg=True)
misc_channels = mne.pick_types(epochs.info, misc=True)
hpi_channels = np.zeros((6,1),dtype=int)
n_hpi = 0
for i in misc_channels:
    if 'hpiin' in raw.info['chs'][i]['ch_name']:
        hpi_channels[n_hpi] = i
        n_hpi = n_hpi+1

hpi_trls = np.squeeze((np.max(data[:,hpi_channels,:],axis=3)-np.min(data[:,hpi_channels,:],axis=3))>1e-3)
hpi_channels = hpi_channels[np.any(hpi_trls,axis=0)]
n_hpi = hpi_channels.size
hpi_trls = hpi_trls[:,np.any(hpi_trls,axis=0)]

# --- Extract amplitudes ---
#fig, ax = plt.subplots(figsize=(7.5, 4.5), nrows=2, ncols=hpi_channels.size, layout="constrained")
fig= plt.figure(figsize=(13, 7))
t_amp = np.zeros((n_hpi,data.shape[1]))
for i_coil in range(n_hpi):
    trls = np.asarray(hpi_trls[:,i_coil]).nonzero()[0][2:-2]
    print("Coil:%s n_trls:%s"%(i_coil,trls.size))
    R = np.zeros((data.shape[1],trls.size))
    Theta = np.zeros((data.shape[1],trls.size))
    for i_trl in range(trls.size):
        X = np.mean(np.multiply(np.cos(2*math.pi*f_hpi*epochs.times),data[trls[i_trl],:,:]),axis=1)
        Y = np.mean(np.multiply(np.sin(2*math.pi*f_hpi*epochs.times),data[trls[i_trl],:,:]),axis=1)
        tmp = X + 1j*Y
        R[:,i_trl] = abs(tmp)
        Theta[:,i_trl] = np.angle(tmp/tmp[hpi_channels[i_coil]])
    amp = np.mean(R,axis=1)
    amp[abs(np.mean(Theta,axis=1))>(math.pi/2)] = -amp[abs(np.mean(Theta,axis=1))>(math.pi/2)]
    amp = np.reshape(amp,(amp.size,1))
    t_amp[i_coil,:] = np.squeeze(amp)
    # Plot topography
    evo = mne.EvokedArray(amp,raw.info)
    ax = fig.add_subplot(1, n_hpi, i_coil+1)
    evo.plot_topomap(0.,ch_type='mag',size=3,res=512,axes=ax,colorbar=False,show=False)
    s = raw.info['chs'][np.squeeze(hpi_channels[i_coil])]['ch_name']
    ax.set_title(s[0:3]+s[-3:], fontsize=14)    
t_amp = t_amp[:,mag_channels] #pick only magentometers
#plt.show()

rrs = np.zeros((n_hpi,3))
gofs = np.zeros((n_hpi,))
moms = np.zeros((n_hpi,3))

max_idx = t_amp.argmax(axis=1)
for i_coil in range(n_hpi):
    loc = raw.info['chs'][mag_channels[max_idx[i_coil]]]['loc'] 
    grid = create_aligned_grid(loc,0.005,0.03, 0.02)

    # --- Fit magnetic dipoles ---
    meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])    
    info = pick_info(raw.info, meg_picks)
    meg_coils = _concatenate_coils(_create_meg_coils(info["chs"], "accurate"))    
    cov = mne.cov.make_ad_hoc_cov(raw.info)
    whitener, _ = mne.cov.compute_whitener(cov, raw.info)
    
    # Make some location guesses (1 cm grid)
    R = np.linalg.norm(meg_coils[0], axis=1).max()
    guesses = _make_guesses(dict(R=R, r0=np.zeros(3)), 0.01, 0.0, 0.01)[0]["rr"]
    
    fwd = _magnetic_dipole_field_vec(guesses, meg_coils, 'warning')
    fwd = np.dot(fwd, whitener.T)
    fwd.shape = (guesses.shape[0], 3, -1)
    fwd = np.linalg.svd(fwd, full_matrices=False)[2]
    guesses = dict(rr=guesses, whitened_fwd_svd=fwd)
    
    coil_fit = _fit_magnetic_dipole(t_amp[i_coil,:], np.zeros((3,)), 'warning', whitener, meg_coils, guesses)
    t1,t2,t3 = zip(coil_fit)
    rrs[i_coil,:] = t1[0]
    gofs[i_coil] = t2[0]
    moms[i_coil,:] = t2[0]

# Plot fitted coils
fig= plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1,3,1, projection='3d')
ax.scatter3D(rrs[:,0], rrs[:,1], rrs[:,2], color = "red", marker=".", linewidths=10)
for i in range(len(rrs)):
    s = raw.info['chs'][np.squeeze(hpi_channels[i])]['ch_name']
    ax.text(rrs[i,0]+0.01,rrs[i,1],rrs[i,2],  s[0:3]+s[-3:], size=10, zorder=1, color='k') 
    #ax.text(rrs[i,0]+0.01,rrs[i,1]-0.0125,rrs[i,2],  "gof=%.3f"%(gofs[i]), size=10, zorder=1, color='k') 
ax.scatter3D(meg_coils[0][:,0], meg_coils[0][:,1], meg_coils[0][:,2], color = "green", alpha = 0.1, marker=".", linewidths=0.5)
ax.view_init(elev=90, azim=-90, roll=0)
ax.set_title('top')

ax = fig.add_subplot(1,3,2, projection='3d')
ax.scatter3D(rrs[:,0], rrs[:,1], rrs[:,2], color = "red", marker=".", linewidths=10)
for i in range(len(rrs)):
    s = raw.info['chs'][np.squeeze(hpi_channels[i])]['ch_name']
    ax.text(rrs[i,0],rrs[i,1]+0.01,rrs[i,2],  s[0:3]+s[-3:], size=10, zorder=1, color='k') 
ax.scatter3D(meg_coils[0][:,0], meg_coils[0][:,1], meg_coils[0][:,2], color = "green", alpha = 0.1, marker=".", linewidths=0.5)
ax.view_init(elev=0, azim=0, roll=0)
ax.set_title('right')

ax = fig.add_subplot(1,3,3, projection='3d')
ax.scatter3D(rrs[:,0], rrs[:,1], rrs[:,2], color = "red", marker=".", linewidths=10)
for i in range(len(rrs)):
    s = raw.info['chs'][np.squeeze(hpi_channels[i])]['ch_name']
    ax.text(rrs[i,0]-0.01,rrs[i,1],rrs[i,2],  s[0:3]+s[-3:], size=10, zorder=1, color='k') 
ax.scatter3D(meg_coils[0][:,0], meg_coils[0][:,1], meg_coils[0][:,2], color = "green", alpha = 0.1, marker=".", linewidths=0.5)
ax.view_init(elev=0, azim=90, roll=0)
ax.set_title('front')
plt.show()

for i_coil in range(n_hpi):
    print("Coil %s: gof = %f"%(i_coil,gofs[i_coil]))