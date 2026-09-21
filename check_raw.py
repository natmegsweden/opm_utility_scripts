import mne

file = '/data/CAPSI/raw/sub-1398/260917/hedscan/HPIbefore_raw.fif'
raw = mne.io.read_raw_fif(file, preload=True)

raw.info

twindow = 5
tmax = raw.times[-1]
tmin = tmax - twindow

noise = raw.copy().crop(tmin=tmin, tmax=tmax)