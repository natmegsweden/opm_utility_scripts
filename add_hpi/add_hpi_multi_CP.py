# Author: C Pfeiffer (adapted from script by T Cheung)
# Last Modified Jan 28, 2025
# Function for adding dev_to_head_trans to OPM-MEG recordings
# When executing the user will be prompted to select the following inputs:
# - data files : OPM-MEG recordings that the transform should be applied to
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
from scipy.spatial import Delaunay



#from mne.io.pick import pick_types #use for older version of mne
from mne._fiff.pick import pick_types

from scipy.signal import find_peaks

#from mne.io._digitization import _call_make_dig_points #use for older version of mne
from mne._fiff._digitization import _call_make_dig_points, _make_dig_points

from mne.transforms import (
    get_ras_to_neuromag_trans,
    Transform, 
    _quat_to_affine, 
    _fit_matched_points
    )

from mne.chpi import (
            _fit_coil_order_dev_head_trans,
            compute_chpi_amplitudes,
            compute_chpi_locs,
        )
from mne.io.constants import FIFF
from mne.utils import _check_fname, logger, verbose, warn
from mne.transforms import apply_trans

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

def tc_plot_psd(raw):
    #hann window
    n_fft = 1024
    psd_ylim = [1.,10000.]
    psd_xlim = [0.,500.]

    projs =0
    fig = raw.plot_psd(fmin=0,n_fft=n_fft,show=False, proj=True, dB=False ,xscale='log',window='hann',n_jobs=-1)
    fig3 = raw.plot_psd(fmin=0,n_fft=n_fft,show=False, proj=False, dB=False ,xscale='log',window='hann',n_jobs=-1)

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

'''
parser=argparse.ArgumentParser(description="sample argument parser", epilog=f"example:\n python {sys.argv[0]} -d data.fif -h hpi.fif 1 43 2 43 3 43, note bad channels should be marked and saved to fif dataset. New dataset will have bad channels deleted and resampled to 1000Hz. VERY IMPORTANT: You must use the single amplitude hpi script NOT the 2 amplitude hpi script to collect the hpi.")
parser.add_argument("-d","--dataset",help="the raw HEDscan dataset you want to update with fiducials")

parser.add_argument("--hpi", help="the hpi dataset")
parser.add_argument("--pol", help="the dataset containing polhemus locations")
parser.add_argument("--gof", type = float, help="goodness of fit")
parser.add_argument("--plot", action='store_true') 
parser.add_argument("port_freq", metavar='N freq', type=int, nargs='+', help="nasionport nasionfreq lpaport lpafreq lpaport lpafreq ... up to 6 can be specified") #1 is the first port, 2 is second port, 


args=parser.parse_args()

if args.gof is not None:
    print(f'gof = {args.gof}')
    gof=args.gof
else:
    print(f'setting gof to 0.98')
    gof=0.98

if '.fif' in args.dataset:
   print (f'adding fiducials to {args.dataset}')
else:
   print (f'invalid dataset')
   exit()

if '.fif' in args.hpi: 
   print (f'using hpi dataset {args.hpi}')
else:
   print (f'invalid hpi dataset')
   exit()
   '''
   
import tkinter as tk
from tkinter import simpledialog, filedialog

def get_files(title):
    return filedialog.askopenfilenames(title=title)

def get_file(title):
    return filedialog.askopenfilename(title=title)

def get_input(prompt, default):
    return simpledialog.askstring("Input", prompt, initialvalue=default)

def get_boolean(prompt):
    while True:
        response = simpledialog.askstring("Input", prompt + " (y/n):", initialvalue='n').lower()
        if response in ['y', 'n']:
            return response == 'y'
        else:
            tk.messagebox.showerror("Invalid input", "Please enter 'y' or 'n'.")

# Create the main window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Get filenames
datafiles = get_files("Select datafile")
hpifile = get_file("Select hpifile")
polfile = get_file("Select polhemusfile")

# Get frequency and order
hpifreq = float(get_input("Enter frequency (Hz):", "33"))
hpiorder = np.array([int(x) for x in get_input("Enter order (comma-separated):", "1, 4, 2, 3").split(',')], dtype=int)

# Get boolean input for plot
plotResult = get_boolean("Do you want to plot the data?")

# Print the results
print(f"Datafile: {datafiles}")
print(f"HPIfile: {hpifile}")
print(f"Polhemusfile: {polfile}")
print(f"Frequency: {hpifreq} Hz")
print(f"Order: {hpiorder}")
print(f"Plot: {plotResult}")

fname=hpifile#args.hpi

raw = mne.io.read_raw_fif(fname)
raw.load_data()
#remove bad channels
for bad_chan in raw.info["bads"]:
    raw.drop_channels(bad_chan)


#remove zero channels
bads=TC_findzerochans(raw.info)
for bad_chan in bads:
    raw.drop_channels(bad_chan)

hpi_names_orig,hpi_indices_orig=TC_get_hpiout_names(raw)

hpi_freq=33.0
hpi_freqs_orig=np.zeros(len(hpi_indices_orig))
for i in range(len(hpi_indices_orig)):
    hpi_freqs_orig[i]=hpi_freq

n=len(hpiorder)
print(hpiorder)
print(len(hpiorder))

print(f'n={n}')

if n == len(hpi_indices_orig):
    hpi_freqs=hpi_freqs_orig
else:
    hpi_freqs=np.zeros(n)

hpi_indices=np.zeros(n,dtype=np.int64)
hpi_names=list()

j=0
for i in range(n):
    hpi_indices[i]=hpi_indices_orig[hpiorder[i]-1]
    hpi_names+=[hpi_names_orig[hpiorder[i]-1]]
    hpi_freqs[i]=hpifreq

print(hpi_indices)
print(hpi_names)
print(hpi_freqs)

#resample

if 1:
    new_sfreq=1000
    raw.load_data().resample(new_sfreq)
    

#Add default cardinals and 3 hpi's with the same locations as the cardinals
#Add any additional hpi's attached with default locations
fname=polfile#args.pol
pol_info = mne.io.read_info(fname)


digpts=np.array([],dtype=float)
lpa=pol_info['dig'][0]['r']
rpa=pol_info['dig'][2]['r']
nasion=pol_info['dig'][1]['r']

hpi=np.array([],dtype=float)
for j in pol_info['dig']:
    if j['kind']==2:    # FIFFV_POINT_HPI = 2
        hpi=np.append(hpi,j['r']) # to account for the gap between sensor surface and cell centre
n=int(hpi.shape[0]/3)
hpi=hpi.reshape((n,3))
hpi_orig = hpi

dev_head_t = Transform("meg", "head", trans=None) 
dev_head_t['trans']=get_ras_to_neuromag_trans(nasion, lpa, rpa) #should remain identity with the above geometry
raw.info.update(dev_head_t=dev_head_t)

info=raw.info

digpts=np.array([],dtype=float)
for j in pol_info['dig']:
    digpts=np.append(digpts,j['r']) # to account for the gap between sensor surface and cell centre
n=int(digpts.shape[0]/3)
digpts=digpts.reshape((n,3))

with raw.info._unlock(): 
    raw.info['dig'], ctf_head_t=_call_make_dig_points(nasion, lpa, rpa, hpi[0:len(hpi_indices)], digpts, convert=True)

sampling_freq = raw.info["sfreq"]

start_sample =  0
stop_sample = len(raw)
print(f'start_sample={start_sample}, stop_sample={stop_sample}')

hpi_locs = []

dist_limit = 0.005

raw_orig = raw.copy()
print(hpi_indices)
slope = np.zeros((len(hpi_indices),len(pick_types(raw.info, meg='mag'))),dtype=float)
for index in range(len(hpi_indices)):
    raw=raw_orig.copy()
    channel_index=hpi_indices[index]
    print(index)
    print(channel_index)
    chan_name=raw.info['ch_names'][channel_index]
    
    print(f'*********HPI channel we want to localize {chan_name}**********')
    print(f'channel_index = {channel_index}')
    print(f'hpi_indices[index] = {hpi_indices[index]}')

    do_plot=False
    
    raw_selection = raw[channel_index, start_sample:stop_sample]
    x = raw_selection[1]
    y = raw_selection[0].T
    b = y.ravel()
    dist=round(raw.info['sfreq']/hpi_freq)-2
    peaks, _ = find_peaks(b, distance=dist,height=0.0001)
    
    if do_plot:
        plt.plot(b)
        plt.plot(peaks, b[peaks], "x")
        plt.show()

    if len(peaks) <1 :
        print('*****************************************')
        print('***********Error no peaks found**********')
        exit()

    window=(peaks[-1]-peaks[0])/raw.info['sfreq']

    print(f'{chan_name} first point = {peaks[0]} and last point = {peaks[-1]}, time window = {window} s')

    #we use this window to extract the portion of data out for the magnetic dipole fit
    if 0:
        raw.filter(l_freq=3, h_freq=55)
        raw = raw.notch_filter(freqs=freqs)

    minT=peaks[0]/raw.info['sfreq']
    maxT=peaks[-1]/raw.info['sfreq']

    tmin=(maxT-minT)/2.-1+minT
    tmax=(maxT-minT)/2.+1+minT #we extract 2 seconds worth of data

    print(f'min time = {minT}, max time = {maxT}')

    print(f'tmin window = {tmin}, t max window = {tmax}')

    raw.crop(tmin=tmin,tmax=tmax) 

   

    if do_plot:

        spectrum = raw.compute_psd(picks=hpi_indices[index],window='hann',proj=False, )
        fig=spectrum.plot(picks='misc', amplitude=True,dB=False,)

        psd_ylim = [1.,10000.]
        psd_xlim = [0.,100.]

        fig.suptitle('%s projs off hann' % (hpi_names[index]))
        fig.axes[0].set_xlim(psd_xlim)
        fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
        plt.show()

        raw_selection2 = raw[channel_index, 0:len(raw)]
        print(f'cropped time window length = {len(raw)}')
        x1 = raw_selection2[1]
        y1 = raw_selection2[0].T
   
        plt.plot(x1,y1)
        plt.show()
    
    
    print('************* add hpi struct to info ***********')

    hpi_sub = dict()

    # Don't know what event_channel is don't think we have it HPIs are either
    # always on or always off.
    # hpi_sub['event_channel'] = ???

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
        print(hpi_coils[i]["drive_chan"])
        hpi_coils[i]["coil_freq"] = default_freqs[i]

        # check if coil is on
        #if header_info[key_base % ("Channel", i + 1)] == "OFF":
        #        hpi_sub["hpi_coils"][i]["event_bits"] = [0]
        #else:
        hpi_sub["hpi_coils"][i]["event_bits"] = [256]

        # read in digitized points if supplied
        #if pos_fname is not None:
        #        info["dig"] = _read_pos(pos_fname)
        #else:
        #        info["dig"] = []

        #info._unlocked = False
        #info._update_redundant()
    
    with raw.info._unlock():  
        raw.info["hpi_subsystem"] = hpi_sub
        raw.info["hpi_meas"] = [{"hpi_coils": hpi_coils}]

    #****************************************************
    print('************* localize hpi *******************')
    n_hpis = 0

    info=raw.info

    for d in info["hpi_subsystem"]["hpi_coils"]:
        if d["event_bits"] == [256]:
            n_hpis += 1
    if n_hpis < 3:
        warn(
            f"{n_hpis:d} HPIs active. At least 3 needed to perform"
            "head localization\n *NO* head localization performed"
        )
    else:
        # Localized HPIs using 2000 milliseconds of data.
        with info._unlock():
            info["hpi_results"] = [
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
        raw.info["line_freq"]=None
        
        coil_amplitudes = compute_chpi_amplitudes(raw, tmin=0, tmax=2, t_window=2, t_step_min=2)
        slope[index,:] = coil_amplitudes['slopes'][0][index]
        
        '''
        assert len(coil_amplitudes["times"]) == 1
        coil_locs = compute_chpi_locs(raw.info, coil_amplitudes)
    
        with info._unlock():
            info["hpi_results"] = None
        
        hpi_g = coil_locs["gofs"][0]
        hpi_dev = coil_locs["rrs"][0]

        hpi_locs.append(hpi_dev)
        
        # fill in hpi_results
        hpi_result = dict()

        # add HPI points in device coords...
        dig = []
        for idx, point in enumerate(hpi_dev):
            dig.append(
                {
                    "r": point,
                    "ident": idx + 1,
                    "kind": FIFF.FIFFV_POINT_HPI,
                    "coord_frame": FIFF.FIFFV_COORD_DEVICE,
                }
            )
        hpi_result["dig_points"] = dig

        # attach Transform
        hpi_result["coord_trans"] = raw.info["dev_head_t"]
        print(f"transfrom={raw.info['dev_head_t']}")

        # 1 based indexing
        hpi_result["order"] = np.arange(len(hpi_indices)) + 1
        hpi_result["used"] = np.arange(len(hpi_indices)) + 1
        hpi_result["dist_limit"] = dist_limit
        hpi_result["good_limit"] = 0.98

        print(f'hpi_result={hpi_result}')
        # Warn for large discrepancies between digitized and fit
        # cHPI locations
        if hpi_result["dist_limit"] > 0.005:
            warn(
                "Large difference between digitized geometry"
                " and HPI geometry. Max coil to coil difference"
                f" is {100.0 * tmp_dists.max():0.2f} cm\n"
                "beware of *POOR* head localization"
            )

        # store it
        with raw.info._unlock():
            raw.info["hpi_results"] = [hpi_result]
    
    if index==0:
        nasion_dev=raw.info['hpi_results'][0]['dig_points'][0]['r']
        na_raw=raw.copy()
        print('generated na_raw ')
        hpi=[nasion_dev]
        gof_na=hpi_g
    if index==1:
        lpa_dev=raw.info['hpi_results'][0]['dig_points'][1]['r']
        le_raw=raw.copy()
        print('generated le_raw ')
        hpi=[nasion_dev,lpa_dev]
        gof_le=hpi_g
    if index==2:
        rpa_dev=raw.info['hpi_results'][0]['dig_points'][2]['r']
        re_raw=raw.copy()
        print('generated re_raw ')
        hpi=[nasion_dev,lpa_dev,rpa_dev]
        gof_re=hpi_g
    if len(hpi_indices) > 3:
        if index==3:
            in_dev=raw.info['hpi_results'][0]['dig_points'][3]['r']
            in_raw=raw.copy()
            print('generated in_raw ')
            hpi=[in_dev] #meters
            hpi=[nasion_dev,lpa_dev,rpa_dev,in_dev]
            gof_in=hpi_g
    if len(hpi_indices) > 4:
        if index==4:
            cz_dev=raw.info['hpi_results'][0]['dig_points'][4]['r']
            cz_raw=raw.copy()
            print('generated cz_raw ') 
            hpi=[in_dev,cz_dev] #meters
            hpi=[nasion_dev,lpa_dev,rpa_dev,in_dev,cz_dev]
            gof_cz=hpi_g
      '''
#********

assert len(coil_amplitudes["times"]) == 1
#%%
rawcopy = raw.copy();
rawcopy.pick(picks =['meg'] , exclude='bads')
fig= plt.figure(figsize=(13, 7))
for i in range(4):
    tmp = np.reshape(slope[i],(slope[i].size,1))
    evo = mne.EvokedArray(tmp,rawcopy.info)
    ax = fig.add_subplot(1, 4, i+1)
    evo.plot_topomap(0.,ch_type='mag',size=3,res=512,axes=ax,colorbar=False,show=False)
    ax.set_title(hpi_names[i], fontsize=14)   

#%%
coil_amplitudes['slopes'][0]= slope
coil_locs = compute_chpi_locs(raw.info, coil_amplitudes)
hpi_dev = np.array(coil_locs['rrs'][0])
hpi_gofs = np.array(coil_locs['gofs'][0])

# Calculate transform
trans = _quat_to_affine(_fit_matched_points(hpi_dev, hpi_orig)[0])
dev_to_head_trans = Transform(fro="meg", to="head", trans=trans)

# Apply trans to hpi_dev
hpi_head = apply_trans(dev_to_head_trans, hpi_dev)
dist = np.linalg.norm(hpi_orig-hpi_head, axis=1)

print('***** Apply trans to files *******************************************')

for i in range(len(datafiles)):
    fname=datafiles[i]
    raw = mne.io.read_raw_fif(fname)
    raw.load_data()
    
    new_sfreq = raw.info['sfreq']
    
    #resample 
    #Note for brainwave, it seems 5000Hz sample rate seems to produce unstable results
    #it is not related to filtering - tested and does not change the results
    #For now always resample to 1000Hz
    #I think it is specific to the beamformer and not other parts of the program
    
    if 1:
        new_sfreq=1000
        raw.load_data().resample(new_sfreq)
        
    #remove bad channels
    for bad_chan in raw.info["bads"]:
        raw.drop_channels(bad_chan)
    
    #remove zero channels
    bads=TC_findzerochans(raw.info)
    for bad_chan in bads:
        raw.drop_channels(bad_chan)
    
    #add the cardinals 
    
    print(dev_to_head_trans)
    raw.info.update(dev_head_t=dev_to_head_trans)
    
    info=raw.info
    digpts=np.array([],dtype=float)
    for j in pol_info['dig']:
        digpts=np.append(digpts,j['r']) 
    n=int(digpts.shape[0]/3)
    digpts=digpts.reshape((n,3))
    
    with raw.info._unlock(): 
        raw.info['dig']=_make_dig_points(nasion, lpa, rpa, hpi_orig, digpts)
    
    path ='/Users/teresa/data/windowshare/20240719/sub-HX'
    savename='test'
    
    print("Path of the file..", os.path.abspath(fname))
    print('File name:', os.path.basename(fname))
    print('Directory Name: ', os.path.dirname(fname))
    
    path=os.path.dirname(fname)
    savename=os.path.basename(fname)
    savename=os.path.splitext(savename)[0]
    savename=savename.replace('_raw','')
    
    raw.save(('%s/%s_CP_hpi_raw.fif' % (path, savename)),overwrite=True)

print('---------------------------------------------')
print(f"hpi_orig: {hpi_orig}\n")
print(f"hpi_dev: {hpi_dev}\n")
for index, value in enumerate(dist):
        status = 'ok' if hpi_gofs[index]<0.9 else 'not ok'
        print(f"Coil: {hpi_names[index][-3:]}, Distance: {(value*1e3):.2f} mm, GOF: {hpi_gofs[index]:.4f}, Status: {status}")
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
    
    plot_3d(senspos, senslabel, hpi_orig, hpilabel, hpi_head, hpi_names, digpts)
