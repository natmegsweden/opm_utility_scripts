# Author: C Pfeiffer (adapted from script by T Cheung)
# Last Modified Jan 28, 2025
# Function for adding dev_to_head_trans to an OPM-MEG recording
# When executing the user will be prompted to select the following inputs:
# - data file : OPM-MEG recording that the transform should be applied to
# - hpi file : OPM recording where hpi coils were activated sequentially
# - polhemus file : TRIUX recording containing a polhemus headshape with the
#                   hpi locations in head coordinates
# - hpi frequency : frequency the coils were driven at

import sys, getopt
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import mne

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Delaunay, cKDTree

from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

#from mne.io.pick import pick_types #use for older version of mne
from mne._fiff.pick import pick_types

from scipy.signal import find_peaks

#from mne.io._digitization import _call_make_dig_points #use for older version of mne
from mne._fiff._digitization import _call_make_dig_points, _make_dig_points

from mne.transforms import (
    get_ras_to_neuromag_trans,
    Transform,
    _quat_to_affine,
    _fit_matched_points,
    invert_transform,
    combine_transforms,
    apply_trans
    )

from mne.chpi import (
            _fit_coil_order_dev_head_trans,
            compute_chpi_amplitudes,
            compute_chpi_locs,
            compute_chpi_opm_locs,
            compute_whitener,
            make_ad_hoc_cov,
            _concatenate_coils,
            _create_meg_coils,
            _magnetic_dipole_field_vec,
            _magnetic_dipole_delta,
        )
from mne.io.constants import FIFF
from mne.utils import _check_fname, logger, verbose, warn

#from mne.io._digitization import _make_dig_points #use for older version of mne
from mne._fiff._digitization import _make_dig_points


def write_bw_marker_file(dsName, events, chanName, fs):
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
            fid.write(f'{i + 1}\n')  # Matlab uses 1-based indexing, Python uses 0-based
            fid.write('NUMBER OF SAMPLES:\n')
            fid.write(f'{len(events)}\n')
            fid.write('LIST OF SAMPLES:\n')
            fid.write('TRIAL NUMBER\t\tTIME FROM SYNC POINT (in seconds)\n')

            for t in range(len(events) - 1):
                fid.write(f'                  %+g\t\t\t\t               %+0.6f\n' % (0, events[t][0]/fs))

            fid.write(f'                  %+g\t\t\t\t               %+0.6f\n\n\n' % (0, events[t][-1]/fs))



def TC_findzerochans(info, tolerance=0.02):
    #tolerance default 2 cm.
    #remove channels that are inside a 2 cm sphere of the origin

    bads_fl=np.array([])
    picks = pick_types(info, meg='mag')
    lst = list(bads_fl)
    for j in picks:
        ch = info['chs'][j]
        if np.isclose(sum(ch['loc'][0:3]),0.,atol=1e-3).all():
            lst.append(ch['ch_name'])
    bads_fl = np.asarray(lst)
    print('found the following channels with locations at 0,0,0')
    print(bads_fl)
    return(bads_fl)

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
    B  = np.dot(whitener, slope_row)
    B2 = float(np.dot(B, B))
    if B2 < 1e-30:
        return float('nan')
    fwd = _magnetic_dipole_field_vec(pos_dev[np.newaxis], meg_coils, 'info')
    residual, *_ = _magnetic_dipole_delta(fwd, whitener, B, B2)
    return float(1.0 - residual / B2)

def find_bads(reffile):
    _load_heavy_deps()
    
    raw = mne.io.read_raw_fif(reffile)
    raw.load_data()
    
    #remove bad-marked channels
    for bad_chan in raw.info["bads"]:
        raw.drop_channels(bad_chan)

    #remove unlocalized channels
    bads=find_zero_location_channels(raw.info)
    for bad_chan in bads:
        raw.drop_channels(bad_chan)
          
    # Detect outliers
    picks = mne.pick_types(raw.info, meg=True, exclude='bads')
    spectrum = raw.compute_psd(picks=picks, method="welch", fmin=70, fmax=80, n_fft=5000, n_per_seg=5000)
    psds = spectrum.get_data()
    background_power = psds.mean(axis=1)

    good_idx = np.arange(len(background_power))

    for _ in range(5): #iteratively remove outliers based on z-score>3
        mean_power = np.mean(background_power[good_idx])
        std_power = np.std(background_power[good_idx])
        threshold = mean_power + 3 * std_power
        new_good_idx = np.where(background_power <= threshold)[0]
        if len(new_good_idx) == len(good_idx):
            break
        good_idx = new_good_idx

    bad_idx = np.setdiff1d(np.arange(len(background_power)), good_idx)
    ch_names = [raw.ch_names[p] for p in picks]
    bad_chs = [ch_names[idx] for idx in bad_idx]

    x = np.arange(len(background_power))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, background_power, 'ko', label='Background power')
    ax.axhline( # Threshold
        threshold,
        color='r',
        linestyle='--',
        linewidth=2,
        label=f'Threshold ({threshold:.2e})'
    )
    ax.plot( # Outliers
        bad_idx,
        background_power[bad_idx],
        'r+',
        markersize=10,
        label='Bad channels'
    )

    for idx in bad_idx:
        ax.text(
            idx,
            background_power[idx],
            ch_names[idx],
            rotation=45,
            fontsize=8,
            color='red'
        )

    ax.set_xlabel('Channel')
    ax.set_ylabel('Background PSD')
    ax.set_title(f'Bad Channel Detection Around HPI Frequency ({hpifreq} Hz)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    
    return bad_chs

def tc_plot_psd(raw):
    #hann winw
    n_fft = 1024
    psd_ylim = [1.,10000.]
    psd_xlim = [0.,500.]

    projs =0
    fig = raw.plot_psd(fmin=0,n_fft=n_fft,show=False, proj=True, dB=False ,xscale='log',winw='hann',n_jobs=-1)
    fig3 = raw.plot_psd(fmin=0,n_fft=n_fft,show=False, proj=False, dB=False ,xscale='log',winw='hann',n_jobs=-1)

    fig.suptitle('%s %d projs on hann' % (fname, projs))
    fig3.suptitle('%s projs off hann' %fname)
    fig.axes[0].set_yscale('log')
    fig3.axes[0].set_yscale('log')

    fig.axes[0].set_ylim(psd_ylim)
    fig.axes[0].set_xlim(psd_xlim)
    fig3.axes[0].set_ylim(psd_ylim)
    fig3.axes[0].set_xlim(psd_xlim)

    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
    fig3.subplots_adjust(0.1, 0.1, 0.95, 0.85)

    plt.show()

    return(fig,fig3)


def TC_get_hpiout_names(raw):
    hpi_names=list()

    #get the names of the  HPI out channels

    hpi_raw = raw.compute_psd(picks="misc")

    for name in hpi_raw.info['ch_names']:
        if 'out' in name:
            #print(name)
            hpi_names+=[name]

    hpi_indices=np.zeros(len(hpi_names),dtype=np.int64)
    i=0
    j=0
    for ch in raw.info['ch_names']:
        for hpi in hpi_names:
            if hpi in ch:
                hpi_indices[j]=i
                j=j+1
        i=i+1


    return(hpi_names,hpi_indices)

def plot_3d(senspos, senslabel, hpipos, hpilabel, hpipos2, hpilabel2, digpos):
    # Convert lists to numpy arrays
    senspos = np.array(senspos)
    senslabel = np.array(senslabel)
    hpipos = np.array(hpipos)
    hpilabel = np.array(hpilabel)

    # Convert senspos to polar coordinates (origin = center of mass)
    center_of_mass = np.mean(senspos, axis=0)
    senspos_centered = senspos - center_of_mass
    r = np.linalg.norm(senspos_centered, axis=1)
    theta = np.arccos(senspos_centered[:, 2] / r)  # polar angle
    phi = np.arctan2(senspos_centered[:, 0], senspos_centered[:, 1])  # azimuth angle with zero on y-axis
    x_proj = theta * np.cos(phi)
    y_proj = theta * np.sin(phi)
    polar_proj = np.vstack((x_proj, y_proj)).T

    # Triangulated in 2D polar space
    tri = Delaunay(polar_proj)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(senspos[:, 0], senspos[:, 1], senspos[:, 2],
                    triangles=tri.simplices, cmap='viridis', alpha=0.6,
                    edgecolor='k', linewidth=0.2)

    # Plot the sensor positions with labels
    ax.scatter(senspos[:, 0], senspos[:, 1], senspos[:, 2], color='r', s=50)
    for i in range(len(senslabel)):
        ax.text(senspos[i, 0], senspos[i, 1], senspos[i, 2], senslabel[i], color='black', fontsize=9)

    # Plot the hpi positions with labels
    ax.scatter(hpipos[:, 0], hpipos[:, 1], hpipos[:, 2], color='b', s=100)
    for i in range(len(hpilabel)):
        ax.text(hpipos[i, 0], hpipos[i, 1], hpipos[i, 2], hpilabel[i], color='blue')

    # Plot the hpi positions with labels
    ax.scatter(hpipos2[:, 0], hpipos2[:, 1], hpipos2[:, 2], color='g', s=100)
    for i in range(len(hpilabel)):
        ax.text(hpipos2[i, 0], hpipos2[i, 1], hpipos2[i, 2], hpilabel2[i], color='green')

    # Plot the hpi positions with labels
    ax.scatter(digpos[:, 0], digpos[:, 1], digpos[:, 2], color='k', s=10)

    plt.show()
    
def perturb_transform(T0, params):
    rvec = params[:3]
    tvec = params[3:]
    R = Rotation.from_rotvec(rvec).as_matrix()
    delta = np.eye(4)
    delta[:3, :3] = R
    delta[:3, 3] = tvec
    T = delta @ T0["trans"]
    return Transform(
        fro=T0["from"],
        to=T0["to"],
        trans=T,
    )

def mean_gof(params):
    trans = perturb_transform(dev_to_head_trans, params)
    head2dev = invert_transform(trans)
    gofs = np.array([
        _gof_at_fixed_pos(
            slope[ch_i],
            apply_trans(head2dev, hpi_dig[pol_i]),
            whitener,
            meg_coils,
        )
        for ch_i, pol_i in zip(incl_idx, indices)
    ])
    return gofs.mean()

def objective(params):
    return -mean_gof(params)


from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

app = QApplication.instance()
if app is None:
    app = QApplication([])

def get_file(title):
    filename, _ = QFileDialog.getOpenFileName(
        None,
        title,
        "",
        "All Files (*)",
    )
    return filename

def get_input(prompt, default):
    text, ok = QInputDialog.getText(
        None,
        "Input",
        prompt,
        text=str(default),
    )
    if not ok:
        raise RuntimeError("User cancelled input")
    return text

def get_boolean(prompt):
    while True:
        text, ok = QInputDialog.getText(
            None,
            "Input",
            f"{prompt} (y/n):",
            text="n",
        )
        if not ok:
            return False
        text = text.strip().lower()
        if text in ["y", "n"]:
            return text == "y"
        QMessageBox.critical(
            None,
            "Invalid input",
            "Please enter 'y' or 'n'.",
        )

# Get inputs
datfile = get_file("Select datafile")
hpifile = get_file("Select hpifile")
polfile = get_file("Select polhemusfile")
erfile = get_file("Select empty room file")
hpifreq = float(get_input("Enter frequency (Hz):", 33))
new_sfreq = float(get_input("Enter wnsampling frequency (Hz):", 1000))
plotResult = get_boolean("Do you want to plot the data?")
use_opt = get_boolean("Use rigid transform to optimize fit?")

# Print the results
print(f"Datafile: {datfile}")
print(f"HPIfile: {hpifile}")
print(f"Polhemusfile: {polfile}")
print(f"Frequency: {hpifreq} Hz")
print(f"wnsampling Frequency: {new_sfreq} Hz")
print(f"Plot: {plotResult}")


raw = mne.io.read_raw_fif(erfile)
raw.load_data()
#remove bad channels
for bad_chan in raw.info["bads"]:
    raw.drop_channels(bad_chan)

#remove zero channels
bads=TC_findzerochans(raw.info)
for bad_chan in bads:
    raw.drop_channels(bad_chan)
      
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.plot_psd(picks=picks, n_fft=5000, n_per_seg=5000) 

#------ Bad channels detection -----
picks = mne.pick_types(raw.info, meg=True, exclude='bads')
spectrum = raw.compute_psd(picks=picks, method="welch", fmin=hpifreq-10, fmax=hpifreq+10, n_fft=5000, n_per_seg=5000)
psds = spectrum.get_data()
freqs = spectrum.freqs
lower_band = (freqs >= hpifreq - 10) & (freqs <= hpifreq - 8) # Background bands (exclude HPI peak)
upper_band = (freqs >= hpifreq + 8) & (freqs <= hpifreq + 10)
background_power = psds[:, lower_band | upper_band].mean(axis=1)

# Detect outliers
good_idx = np.arange(len(background_power))

for _ in range(5):
    mean_power = np.mean(background_power[good_idx])
    std_power = np.std(background_power[good_idx])
    threshold = mean_power + 3 * std_power
    new_good_idx = np.where(background_power <= threshold)[0]
    if len(new_good_idx) == len(good_idx):
        break
    good_idx = new_good_idx

bad_idx = np.setdiff1d(np.arange(len(background_power)), good_idx)
ch_names = [raw.ch_names[p] for p in picks]
bad_chs = [ch_names[idx] for idx in bad_idx]

x = np.arange(len(background_power))
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(x, background_power, 'ko', label='Background power')
ax.axhline( # Threshold
    threshold,
    color='r',
    linestyle='--',
    linewidth=2,
    label=f'Threshold ({threshold:.2e})'
)
ax.plot( # Outliers
    bad_idx,
    background_power[bad_idx],
    'r+',
    markersize=10,
    label='Bad channels'
)

for idx in bad_idx:
    ax.text(
        idx,
        background_power[idx],
        ch_names[idx],
        rotation=45,
        fontsize=8,
        color='red'
    )

ax.set_xlabel('Channel')
ax.set_ylabel('Background PSD')
ax.set_title(f'Bad Channel Detection Around HPI Frequency ({hpifreq} Hz)')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

#--- Read HPI ----
raw = mne.io.read_raw_fif(hpifile)
raw.load_data()
#remove bad channels
for bad_chan in raw.info["bads"]:
    raw.drop_channels(bad_chan)

#remove zero channels
bads=TC_findzerochans(raw.info)
for bad_chan in bads:
    raw.drop_channels(bad_chan)
    
print("*** Bad channels: ***")
for ch in bad_chs:
    print(ch)

for i in bad_chs:
    if i in raw.info["ch_names"]:
        raw.drop_channels(i)

# --- HFC -------------
#projs = mne.preprocessing.compute_proj_hfc(raw.info,order=1, picks='meg',exclude='bads')
#raw.add_proj(projs)
#raw.apply_proj()

hpi_names,hpi_indices=TC_get_hpiout_names(raw)

hpi_freqs=np.zeros(len(hpi_indices))
for i in range(len(hpi_indices)):
    hpi_freqs[i]=hpifreq

#resample
if 1:
    raw.load_data().resample(1000)

picks = mne.pick_types(raw.info, meg=True, exclude='bads')
raw.plot_psd(picks=picks, n_fft=5000, n_per_seg=5000) 

#assuming file with polhemus locations of fiducials and HPIs
fname=polfile
pol_info = mne.io.read_info(fname)


digpts=np.array([],dtype=float)
lpa=pol_info['dig'][0]['r']
rpa=pol_info['dig'][2]['r']
nasion=pol_info['dig'][1]['r']

hpi=np.array([],dtype=float)
for j in pol_info['dig']:
    if j['kind']==2:    # FIFFV_POINT_HPI = 2
        hpi=np.append(hpi,j['r']) 
n=int(hpi.shape[0]/3)
hpi=hpi.reshape((n,3))
hpi_dig = hpi

dev_head_t = Transform("meg", "head", trans=None)
dev_head_t['trans']=get_ras_to_neuromag_trans(nasion, lpa, rpa) #should remain identity with the above geometry
raw.info.update(dev_head_t=dev_head_t)

info=raw.info

digpts=np.array([],dtype=float)
for j in pol_info['dig']:
    digpts=np.append(digpts,j['r'])
n=int(digpts.shape[0]/3)
digpts=digpts.reshape((n,3))

with raw.info._unlock():
    raw.info['dig'], ctf_head_t=_call_make_dig_points(nasion, lpa, rpa, hpi[0:len(hpi_indices)], digpts, convert=True)

sampling_freq = raw.info["sfreq"]

start_sample =  0
stop_sample = len(raw)
print(f'start_sample={start_sample}, stop_sample={stop_sample}')

hpi_locs = []

raw_orig = raw.copy()
n_hpis = 0
i_hpis = []
slope = np.zeros((len(hpi_indices),len(pick_types(raw.info, meg='mag'))),dtype=float)
for index in range(len(hpi_indices)):
    raw=raw_orig.copy()
    channel_index=hpi_indices[index]
    chan_name=raw.info['ch_names'][channel_index]

    print(f'********* HPI channel: {chan_name} **********')
    do_plot=False

    raw_selection = raw[channel_index, start_sample:stop_sample]
    x = raw_selection[1]
    y = raw_selection[0].T
    b = y.ravel()
    dist=round(raw.info['sfreq']/hpifreq)-2
    peaks, _ = find_peaks(b, distance=dist,height=0.0001)

    if do_plot:
        plt.plot(b)
        plt.plot(peaks, b[peaks], "x")
        plt.show()

    if len(peaks) <1 :
        print('NO PEAKS FOUND')
        continue#exit()

    winw=(peaks[-1]-peaks[0])/raw.info['sfreq']
    
    print(f'{chan_name} first point = {peaks[0]} and last point = {peaks[-1]}, time winw = {winw} s')
    #we use this winw to extract the portion of data out for the magnetic dipole fit

    minT=peaks[0]/raw.info['sfreq']
    maxT=peaks[-1]/raw.info['sfreq']

    tmin=(maxT-minT)/2.-3 + minT
    tmax=(maxT-minT)/2.+3 + minT #we extract 6 seconds worth of data

    print(f'coil on: {minT} .. {maxT} sec')

    print(f'using winw: {tmin} .. {tmax} sec')

    raw.crop(tmin=tmin,tmax=tmax)

    if do_plot:
        spectrum = raw.compute_psd(picks=hpi_indices[index],winw='hann',proj=False, )
        fig=spectrum.plot(picks='misc', amplitude=True,dB=False,)

        psd_ylim = [1.,10000.]
        psd_xlim = [0.,100.]

        fig.suptitle('%s projs off hann' % (hpi_names[index]))
        fig.axes[0].set_xlim(psd_xlim)
        fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
        plt.show()

        raw_selection2 = raw[channel_index, 0:len(raw)]
        x1 = raw_selection2[1]
        y1 = raw_selection2[0].T

        plt.plot(x1,y1)
        plt.show()

    hpi_sub = dict()

    hpi_sub["hpi_coils"] = []
    hpi_sub["hpi_coils"].append({})

    hpi_coils=[]
    hpi_coils.append({})

    drive_channels = hpi_names[0]
    key_base = "Head Localization"
    default_freqs = hpi_freqs

    # build coil structure
    hpi_coils[0]["number"] = 1
    hpi_coils[0]["drive_chan"] = drive_channels[0]
    hpi_coils[0]["coil_freq"] = default_freqs[0]

    hpi_sub["hpi_coils"][0]["event_bits"] = [256]

    with raw.info._unlock():
        raw.info["hpi_subsystem"] = hpi_sub
        raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]
        
    #****************************************************
    print('Extracting hpi amplitudes...')
    raw.info["line_freq"]=None
    coil_amplitudes = compute_chpi_amplitudes(raw, tmin=0, tmax=2, t_window=2, t_step_min=2)
    
    slope[index,:] = coil_amplitudes['slopes'][0][0]
    i_hpis.append(index)
    n_hpis+=1
    
hpi_indices = hpi_indices[i_hpis]

print('Adding hpi struct to info...')
hpi_sub = dict()
hpi_sub["hpi_coils"] = []
for _ in range(len(hpi_indices)):
    hpi_sub["hpi_coils"].append({})

hpi_coils=[]
for _ in range(len(hpi_indices)):
    hpi_coils.append({})

drive_channels = hpi_names
key_base = "Head Localization"
default_freqs = hpi_freqs
for i in range(len(hpi_indices)):
    # build coil structure
    hpi_coils[i]["number"] = i + 1
    hpi_coils[i]["drive_chan"] = drive_channels[i]
    hpi_coils[i]["coil_freq"] = default_freqs[i]
    hpi_sub["hpi_coils"][i]["event_bits"] = [256]

with raw.info._unlock():
    raw.info["hpi_subsystem"] = hpi_sub
    raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]
    raw.info["hpi_results"] = [
        dict(
            dig_points=[
                dict(
                    r=np.zeros(3),
                    coord_frame=FIFF.FIFFV_COORD_DEVICE,
                    ident=ii + 1,
                )
                for ii in range(n_hpis)
            ],
            coord_trans=Transform("meg", "head"),
        )
    ]

assert len(coil_amplitudes["times"]) == 1
coil_amplitudes['slopes'] = np.zeros((1,slope.shape[0],slope.shape[1]))
coil_amplitudes['slopes'][0] = slope

if n_hpis < 3:
    warn(
        f"{n_hpis:d} HPIs active. At least 3 needed to perform"
        "head localization\n *NO* head localization performed"
    )  

print('Fitting coils...')
coil_locs = compute_chpi_opm_locs(raw.info, coil_amplitudes)
hpi_dev = np.array(coil_locs['rrs'][0])
hpi_gofs = np.array(coil_locs['gofs'][0])

fname=datfile#args.dataset
raw = mne.io.read_raw_fif(fname)
raw.load_data()

#resample if new sampling freq different from old one
if new_sfreq != raw.info['sfreq']: 
    raw.load_data().resample(new_sfreq)

#remove bad channels
for bad_chan in raw.info["bads"]:
    raw.drop_channels(bad_chan)

#remove zero channels
bads=TC_findzerochans(raw.info)
for bad_chan in bads:
    raw.drop_channels(bad_chan)

#only use good fits
include_hpis = hpi_gofs>0.96

tree = cKDTree(hpi_dig-hpi_dig.mean(axis=0)) # shift points to centroid to avoid problems with bad coil placement
distances, indices = tree.query(hpi_dev[include_hpis]-hpi_dev[include_hpis].mean(axis=0)) # find closest points

print('Calculating transform...')
trans = _quat_to_affine(_fit_matched_points(hpi_dev[include_hpis], hpi_dig[indices])[0])
dev_to_head_trans = Transform(fro="meg", to="head", trans=trans)

print(f"hpi_dig: {hpi_dev[include_hpis]}\n")
print(f"hpi_dev: {hpi_dig[indices]}\n")
print(f"trans: {dev_to_head_trans}\n")

hpi_head = apply_trans(dev_to_head_trans, hpi_dev)
dist = np.linalg.norm(hpi_dig[indices]-hpi_head[include_hpis], axis=1)

# Evaluate fits at transformed polhemus locations
raw_for_topomap = raw_orig.copy()
raw_for_topomap.pick(picks=['meg'], exclude='bads')

cov      = make_ad_hoc_cov(raw_for_topomap.info, verbose=False)
whitener, _ = compute_whitener(cov, raw_for_topomap.info, verbose=False)
meg_coils   = _concatenate_coils(
    _create_meg_coils(raw_for_topomap.info['chs'], 'accurate')
)
head2dev    = invert_transform(dev_to_head_trans)
incl_idx    = np.where(include_hpis)[0]
pol_gofs1    = np.array([
    _gof_at_fixed_pos(
        slope[ch_i],
        apply_trans(head2dev, hpi_dig[pol_i]),
        whitener,
        meg_coils,
    )
    for ch_i, pol_i in zip(incl_idx, indices)
])
print('Fixed location GOFs:')
for i in range(len(incl_idx)):
    print(f"Coil {incl_idx[i]}: {pol_gofs1[i]}")


# Optimize transform
bounds = [
    (-np.deg2rad(5), np.deg2rad(10)),   # rx
    (-np.deg2rad(5), np.deg2rad(10)),   # ry
    (-np.deg2rad(5), np.deg2rad(10)),   # rz
    (-0.005, 0.005),                   # tx 5 mm
    (-0.005, 0.005),                   # ty
    (-0.005, 0.005),                   # tz
]
result = minimize(
    objective,
    x0=np.zeros(6),
    method="L-BFGS-B",
    bounds=bounds,
)

print(result)
opt_trans = perturb_transform(
    dev_to_head_trans,
    result.x,
)
print('*** Optimized transform ***')
print(opt_trans)
raw_for_topomap = raw_orig.copy()
raw_for_topomap.pick(picks=['meg'], exclude='bads')
cov      = make_ad_hoc_cov(raw_for_topomap.info, verbose=False)
whitener, _ = compute_whitener(cov, raw_for_topomap.info, verbose=False)
meg_coils   = _concatenate_coils(
    _create_meg_coils(raw_for_topomap.info['chs'], 'accurate')
)
head2dev    = invert_transform(opt_trans)
incl_idx    = np.where(include_hpis)[0]
pol_gofs2    = np.array([
    _gof_at_fixed_pos(
        slope[ch_i],
        apply_trans(head2dev, hpi_dig[pol_i]),
        whitener,
        meg_coils,
    )
    for ch_i, pol_i in zip(incl_idx, indices)
])
print('Fixed location GOFs (optimized):')
for i in range(len(incl_idx)):
    print(f"Coil {incl_idx[i]}: {pol_gofs2[i]}")
      
hpi_head2 = apply_trans(opt_trans, hpi_dev)
dist2 = np.linalg.norm(hpi_dig[indices]-hpi_head2[include_hpis], axis=1)

if use_opt:
    dev_to_head_trans = opt_trans

# --- Apply to recording ---------------------
print('Applying trans to recording file...')
raw.info.update(dev_head_t=dev_to_head_trans)

info=raw.info
digpts=np.array([],dtype=float)
for j in pol_info['dig']:
    digpts=np.append(digpts,j['r'])
n=int(digpts.shape[0]/3)
digpts=digpts.reshape((n,3))

with raw.info._unlock():
    raw.info['dig']=_make_dig_points(nasion, lpa, rpa, hpi_dig, digpts)

print("Path of the file..", os.path.abspath(fname))
print('File name:', os.path.basename(fname))
print('Directory Name: ', os.path.dirname(fname))

path=os.path.dirname(fname)
savename=os.path.basename(fname)
savename=os.path.splitext(savename)[0]
savename=savename.replace('_raw','')

raw.save(('%s/%s_proc-hpi+ds_meg.fif' % (path, savename)),overwrite=True)

print('---------------------------------------------')
print(f"hpi_dig: {hpi_dig}\n")
print(f"hpi_dev: {hpi_dev}\n")
print(f"order: {indices}\n")
print(f"mean distance = {np.mean(dist)*1000:.1f} mm\n")
for index, value in enumerate(hpi_gofs):
        status = 'ok' if hpi_gofs[index]>0.9 else 'not ok'
        if status == 'ok':
            print(f"Coil: {hpi_names[index][-3:]}, GOF: {value:.3f}, Dist(mm): {dist[index]*1e3:.1f}, Status: {status}, fixed-GOF: {pol_gofs1[index]:.3f}, opt-GOF: {pol_gofs2[index]:.3f}, opt-Dist(mm): {dist2[index]*1e3:.1f}")
        else:
            print(f"Coil: {hpi_names[index][-3:]}, GOF: {value:.3f}, Status: {status}")
                
print('---------------------------------------------')

if plotResult:
    senspos=np.array([],dtype=float)
    picks = pick_types(raw.info, meg='mag')
    for j in picks:
        senspos=np.append(senspos, apply_trans(dev_to_head_trans, (raw.info['chs'][j]['loc'][0:3])))
    n=int(senspos.shape[0]/3)
    senspos=senspos.reshape((n,3))

    senslabel=list()
    picks = pick_types(raw.info, meg='mag')
    for j in picks:
        index = raw.info['chs'][j]['ch_name'].find('s')
        if index != -1:
            senslabel.append(raw.info['chs'][j]['ch_name'][index:])
        else:
            senslabel.append('')

    digpts=np.array([],dtype=float)
    for j in raw.info['dig']:
        digpts=np.append(digpts,j['r']) # to account for the gap between sensor surface and cell centre
    n=int(digpts.shape[0]/3)
    digpts=digpts.reshape((n,3))
    hpilabel=list()
    for j in range(len(hpi_names)):
        hpilabel+=[str(j+1)]
   
    labels = [hpilabel[i] for i in i_hpis]
    plot_3d(senspos, senslabel, hpi_dig, labels, hpi_head, hpi_names, digpts)
