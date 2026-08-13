#!/usr/bin/env python3
"""
Step-by-step diagnostic for HPI fitting.

Runs the same pipeline as fit_hpi but pauses at each stage and prints/plots
intermediate results so you can pinpoint where the fit breaks down.

Usage::

    python -m opm_utility_scripts.hpi.diagnose \\
        --hpi  /data/<proj>/raw/<sub>/<ses>/hedscan/HPIBefore_raw.fif \\
        --pol  /data/<proj>/raw/<sub>/<ses>/polhemus/digitisation_*.json \\
        --freq 33
"""

import argparse
import sys
import warnings

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne._fiff._digitization import _call_make_dig_points
from mne._fiff.pick import pick_info, pick_types
from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs
from mne.io.constants import FIFF
from mne.transforms import (
    Transform,
    _fit_matched_points,
    _quat_to_affine,
    apply_trans,
    get_ras_to_neuromag_trans,
)
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from opm_utility_scripts.channels import find_zero_location_channels, get_hpi_output_channels
from opm_utility_scripts.io import load_polhemus

_HPI_FIT_SFREQ = 1000
SEP = '─' * 72


def _sep(title=''):
    if title:
        pad = max(0, 72 - len(title) - 2)
        print(f'\n{"─" * (pad // 2)} {title} {"─" * (pad - pad // 2)}\n')
    else:
        print(f'\n{SEP}\n')


# ---------------------------------------------------------------------------
# Step 1 — Polhemus
# ---------------------------------------------------------------------------

def step1_polhemus(pol_path):
    _sep('STEP 1 — Polhemus file')
    print(f'File: {pol_path}')

    pol = load_polhemus(pol_path)
    lpa, nasion, rpa = pol['lpa'], pol['nasion'], pol['rpa']
    hpi_orig = pol['hpi_orig']

    print(f'\nFiducials in isotrak (mm):')
    print(f'  LPA:    {lpa * 1000}')
    print(f'  Nasion: {nasion * 1000}')
    print(f'  RPA:    {rpa * 1000}')
    print(f'\nHPI coils in isotrak (mm):  [{len(hpi_orig)} coils]')
    for i, h in enumerate(hpi_orig):
        print(f'  coil {i}: {h * 1000}')

    isotrak_to_head = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    hpi_orig_head = apply_trans(isotrak_to_head, hpi_orig)
    fids_head = apply_trans(isotrak_to_head, np.array([lpa, nasion, rpa]))

    print(f'\nFiducials in head frame (mm) — should be on-axis:')
    print(f'  LPA:    {fids_head[0] * 1000}  (expect [-x, ~0, ~0])')
    print(f'  Nasion: {fids_head[1] * 1000}  (expect [~0, +y, ~0])')
    print(f'  RPA:    {fids_head[2] * 1000}  (expect [+x, ~0, ~0])')

    print(f'\nHPI coils in head frame (mm):')
    for i, h in enumerate(hpi_orig_head):
        dist = np.linalg.norm(h)
        print(f'  coil {i}: {h * 1000}  |r|={dist * 1000:.1f} mm')

    inter = np.linalg.norm(hpi_orig_head[:, None] - hpi_orig_head[None, :], axis=2)
    np.fill_diagonal(inter, np.nan)
    print(f'\nInter-coil distances (mm):')
    for i in range(len(hpi_orig_head)):
        row = [f'{inter[i,j]*1000:.1f}' if not np.isnan(inter[i,j]) else '  — '
               for j in range(len(hpi_orig_head))]
        print(f'  coil {i}: {" ".join(row)}')

    extra = pol['extra_pts']
    print(f'\nHeadshape points: {len(extra)}')

    return pol, isotrak_to_head, hpi_orig_head


# ---------------------------------------------------------------------------
# Step 2 — HPI raw file
# ---------------------------------------------------------------------------

def step2_raw(hpi_path, hpifreq):
    _sep('STEP 2 — HPI raw file')
    print(f'File: {hpi_path}')

    raw = mne.io.read_raw_fif(hpi_path, preload=True, verbose='error')
    print(f'\nSampling rate: {raw.info["sfreq"]} Hz')
    print(f'Duration: {raw.times[-1]:.1f} s')
    print(f'Total channels: {len(raw.ch_names)}')
    print(f'Pre-marked bads: {raw.info["bads"]}')

    for ch in list(raw.info['bads']):
        raw.drop_channels(ch)

    zero_bads = find_zero_location_channels(raw.info)
    print(f'Zero-location channels (dropped): {zero_bads}')
    for ch in zero_bads:
        raw.drop_channels(ch)

    hpi_names, hpi_indices = get_hpi_output_channels(raw)
    print(f'\nHPI output channels: {hpi_names}')
    print(f'HPI output indices:  {hpi_indices}')

    misc_chs = [raw.info['ch_names'][i] for i in mne.pick_types(raw.info, misc=True)]
    hpiin_chs = [n for n in misc_chs if 'hpiin' in n]
    print(f'HPI drive (hpiin) channels: {hpiin_chs}')

    meg_count = len(mne.pick_types(raw.info, meg=True))
    print(f'MEG channels remaining: {meg_count}')

    raw.load_data().resample(_HPI_FIT_SFREQ)
    print(f'Resampled to {_HPI_FIT_SFREQ} Hz')

    return raw, hpi_names, hpi_indices


# ---------------------------------------------------------------------------
# Step 3 — Peak detection & amplitude windows
# ---------------------------------------------------------------------------

def step3_peaks(raw, hpi_indices, hpi_names, hpifreq):
    _sep('STEP 3 — Drive signal peak detection')

    peak_dist = round(raw.info['sfreq'] / hpifreq) - 2
    print(f'Expected peak separation: {peak_dist} samples  '
          f'({peak_dist / raw.info["sfreq"] * 1000:.1f} ms)')

    n_coils = len(hpi_indices)
    fig, axes = plt.subplots(n_coils, 1, figsize=(14, 2.5 * n_coils), sharex=False)
    if n_coils == 1:
        axes = [axes]

    windows = []   # (tmin, tmax) per coil
    for idx, (ch_idx, name) in enumerate(zip(hpi_indices, hpi_names)):
        b = raw[ch_idx, :][0].ravel()
        t = raw.times
        peaks, props = find_peaks(b, distance=peak_dist, height=0.0001)
        print(f'\n  {name}:  {len(peaks)} peaks found')

        if len(peaks) == 0:
            windows.append(None)
            axes[idx].plot(t, b)
            axes[idx].set_title(f'{name} — NO PEAKS FOUND', color='red')
            continue

        minT = peaks[0] / raw.info['sfreq']
        maxT = peaks[-1] / raw.info['sfreq']
        tmin = (maxT - minT) / 2.0 - 1 + minT
        tmax = (maxT - minT) / 2.0 + 1 + minT
        windows.append((tmin, tmax))
        print(f'    first peak: {minT:.3f} s  last peak: {maxT:.3f} s')
        print(f'    fitting window: {tmin:.3f} – {tmax:.3f} s  '
              f'({(tmax - tmin):.2f} s)')

        axes[idx].plot(t, b, lw=0.6)
        axes[idx].plot(t[peaks], b[peaks], 'r.', ms=4)
        axes[idx].axvspan(tmin, tmax, alpha=0.15, color='green', label='fit window')
        axes[idx].set_title(name)
        axes[idx].set_ylabel('V')

    axes[-1].set_xlabel('time (s)')
    fig.suptitle('Step 3 — HPI drive signals with detected peaks and fit windows')
    plt.tight_layout()
    plt.show()

    return windows


# ---------------------------------------------------------------------------
# Step 4 — Amplitude estimation & GOF
# ---------------------------------------------------------------------------

def step4_amplitudes(raw, hpi_indices, hpi_names, hpi_orig_head,
                     pol, isotrak_to_head, windows, hpifreq):
    _sep('STEP 4 — Amplitude estimation (compute_chpi_amplitudes/locs)')

    n_coils = len(hpi_indices)
    hpi_freqs = np.full(n_coils, hpifreq)

    slope = np.zeros((n_coils, len(pick_types(raw.info, meg='mag'))), dtype=float)
    raw_orig = raw.copy()
    last_coil_amplitudes = None

    for index in range(n_coils):
        raw_c = raw_orig.copy()
        win = windows[index]
        if win is None:
            print(f'  {hpi_names[index]}: skipped (no peaks)')
            continue

        raw_c.crop(tmin=win[0], tmax=win[1])

        hpi_sub = {'hpi_coils': [{'event_bits': [256]} for _ in range(n_coils)]}
        hpi_coils_info = [
            {'number': i + 1, 'drive_chan': hpi_names[i], 'coil_freq': hpi_freqs[i]}
            for i in range(n_coils)
        ]
        with raw_c.info._unlock():
            raw_c.info['hpi_subsystem'] = hpi_sub
            raw_c.info['hpi_meas'] = [{'hpi_coils': hpi_coils_info}]
            raw_c.info['hpi_results'] = [dict(
                dig_points=[
                    dict(r=np.zeros(3),
                         coord_frame=FIFF.FIFFV_COORD_DEVICE,
                         ident=ii + 1)
                    for ii in range(n_coils)
                ],
                coord_trans=Transform('meg', 'head'),
            )]
            raw_c.info['line_freq'] = None

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ca = compute_chpi_amplitudes(raw_c, tmin=0, tmax=2,
                                         t_window=2, t_step_min=2)
        slope[index, :] = ca['slopes'][0][index]
        last_coil_amplitudes = ca
        print(f'  {hpi_names[index]}: amplitude estimated ok')

    if last_coil_amplitudes is None:
        print('ERROR: no coil amplitudes estimated — cannot continue')
        return None, None, None

    last_coil_amplitudes['slopes'][0] = slope

    # Need dig + dev_head_t set before compute_chpi_locs
    nasion = pol['nasion']
    lpa = pol['lpa']
    rpa = pol['rpa']

    with raw_orig.info._unlock():
        raw_orig.info['dig'], _ = _call_make_dig_points(
            nasion, lpa, rpa,
            pol['hpi_orig'][0:n_coils],
            pol['extra_pts'],
            convert=True,
        )
    raw_orig.info.update(dev_head_t=Transform('meg', 'head'))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        coil_locs = compute_chpi_locs(raw_orig.info, last_coil_amplitudes)

    hpi_dev = np.array(coil_locs['rrs'][0])
    hpi_gofs = np.array(coil_locs['gofs'][0])

    print(f'\n  Fitted HPI positions in device coordinates (mm) and GOF:')
    for i, (pos, gof) in enumerate(zip(hpi_dev, hpi_gofs)):
        flag = '  OK' if gof > 0.9 else '  POOR'
        dist = np.linalg.norm(pos)
        print(f'    coil {i} ({hpi_names[i]}): '
              f'{pos * 1000}  |r|={dist * 1000:.1f} mm  GOF={gof:.3f}{flag}')

    return hpi_dev, hpi_gofs, raw_orig


# ---------------------------------------------------------------------------
# Step 5 — Matching & transform
# ---------------------------------------------------------------------------

def step5_matching(hpi_dev, hpi_gofs, hpi_orig_head, hpi_names):
    _sep('STEP 5 — KDTree matching and transform')

    include = hpi_gofs > 0.9
    print(f'Coils with GOF > 0.9: {include.sum()} of {len(include)}')
    if include.sum() < 3:
        print('WARNING: fewer than 3 coils with GOF > 0.9 — '
              'transform may be unreliable')

    tree = cKDTree(hpi_orig_head)
    distances, tree_indices = tree.query(hpi_dev[include])

    print(f'\nMatching (device-space coil → nearest polhemus coil in head frame):')
    for k, (dev_i, pol_i) in enumerate(
            zip(np.where(include)[0], tree_indices)):
        print(f'  device coil {dev_i} ({hpi_names[dev_i]}) '
              f'→ polhemus coil {pol_i}  '
              f'(naive distance before fit: {distances[k]*1000:.1f} mm)')

    if include.sum() < 2:
        print('ERROR: need at least 2 included coils to fit transform')
        return None

    trans = _quat_to_affine(
        _fit_matched_points(hpi_dev[include], hpi_orig_head[tree_indices])[0]
    )
    dev_to_head = Transform(fro='meg', to='head', trans=trans)

    hpi_head_fitted = apply_trans(dev_to_head, hpi_dev)

    print(f'\nTransform matrix (device → head):')
    print(dev_to_head['trans'])

    print(f'\nResidual distances after transform (mm):')
    for k, (dev_i, pol_i) in enumerate(
            zip(np.where(include)[0], tree_indices)):
        res = np.linalg.norm(
            hpi_orig_head[pol_i] - hpi_head_fitted[dev_i]
        ) * 1000
        flag = '  OK' if res < 5 else '  LARGE'
        print(f'  coil {dev_i}: {res:.2f} mm{flag}')

    # Also check excluded coils
    excluded = np.where(~include)[0]
    if len(excluded):
        print(f'\nExcluded coils (GOF ≤ 0.9) transformed positions vs polhemus:')
        for dev_i in excluded:
            pos_head = hpi_head_fitted[dev_i]
            dists_to_all = np.linalg.norm(hpi_orig_head - pos_head, axis=1) * 1000
            nearest = np.argmin(dists_to_all)
            print(f'  coil {dev_i} ({hpi_names[dev_i]}) '
                  f'GOF={hpi_gofs[dev_i]:.3f} → '
                  f'nearest polhemus coil {nearest}  '
                  f'dist={dists_to_all[nearest]:.1f} mm')

    return dev_to_head


# ---------------------------------------------------------------------------
# Step 6 — 3-D summary plot
# ---------------------------------------------------------------------------

def step6_plot(hpi_dev, hpi_gofs, hpi_orig_head, hpi_names,
               pol, dev_to_head, raw):
    _sep('STEP 6 — 3-D summary plot')

    # Sensor positions
    meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])
    sensor_pos = np.array([raw.info['chs'][i]['loc'][:3] for i in meg_picks]) * 1000

    # HPI in device space (mm)
    hpi_dev_mm = hpi_dev * 1000

    # Polhemus coils in head space (mm) — these are the targets
    hpi_pol_mm = hpi_orig_head * 1000

    # Fitted coil positions transformed to head space
    if dev_to_head is not None:
        hpi_fitted_head_mm = apply_trans(dev_to_head, hpi_dev) * 1000
    else:
        hpi_fitted_head_mm = None

    # Headshape in head space
    extra = apply_trans(
        get_ras_to_neuromag_trans(pol['nasion'], pol['lpa'], pol['rpa']),
        pol['extra_pts']
    ) * 1000 if len(pol['extra_pts']) else None

    views = [
        dict(title='top',   elev=90,  azim=-90),
        dict(title='right', elev=0,   azim=0),
        dict(title='front', elev=0,   azim=90),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(projection='3d'))

    for ax, v in zip(axes, views):
        # Sensors (device space)
        ax.scatter(sensor_pos[:, 0], sensor_pos[:, 1], sensor_pos[:, 2],
                   c='lightgreen', s=3, alpha=0.2, label='sensors (dev)')

        # Polhemus coils (head space) — targets, shown as stars
        ax.scatter(hpi_pol_mm[:, 0], hpi_pol_mm[:, 1], hpi_pol_mm[:, 2],
                   c='blue', s=120, marker='*', zorder=6, label='polhemus (head)')

        # Fitted coils in device space — crosses
        for i, (pos, gof) in enumerate(zip(hpi_dev_mm, hpi_gofs)):
            color = 'red' if gof > 0.9 else 'orange'
            ax.scatter(*pos, c=color, s=60, marker='x', zorder=5)
            ax.text(pos[0], pos[1], pos[2],
                    f'{hpi_names[i][-3:]}\nGOF={gof:.2f}',
                    size=7, color=color)

        # Transformed positions (head space) — circles
        if hpi_fitted_head_mm is not None:
            ax.scatter(hpi_fitted_head_mm[:, 0],
                       hpi_fitted_head_mm[:, 1],
                       hpi_fitted_head_mm[:, 2],
                       c='red', s=60, marker='o', zorder=5,
                       label='fitted (head)', alpha=0.7)
            # Lines connecting matched pairs
            include = hpi_gofs > 0.9
            tree = cKDTree(hpi_orig_head)
            _, tree_indices = tree.query(hpi_dev[include])
            for k, (dev_i, pol_i) in enumerate(
                    zip(np.where(include)[0], tree_indices)):
                p0 = hpi_fitted_head_mm[dev_i]
                p1 = hpi_pol_mm[pol_i]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                        'k--', lw=0.8, alpha=0.5)

        if extra is not None:
            ax.scatter(extra[:, 0], extra[:, 1], extra[:, 2],
                       c='gray', s=2, alpha=0.3)

        ax.view_init(elev=v['elev'], azim=v['azim'])
        ax.set_title(v['title'])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_zlabel('z (mm)')

    axes[0].legend(loc='upper left', fontsize=7)
    fig.suptitle(
        'Step 6 — 3-D summary\n'
        'green=sensors(dev)  blue★=polhemus(head)  '
        'red×=fitted(dev)  red●=fitted(head)\n'
        'dashed lines = matched pairs',
        fontsize=9
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='python -m opm_utility_scripts.hpi.diagnose',
        description='Step-by-step HPI fit diagnostic.'
    )
    parser.add_argument('--hpi', '-H', required=True,
                        help='Path to raw HPI .fif file (e.g. HPIBefore_raw.fif)')
    parser.add_argument('--pol', '-p', required=True,
                        help='Path to polhemus file (.json or .fif)')
    parser.add_argument('--freq', '-f', type=float, default=33.0,
                        help='HPI drive frequency in Hz (default: 33)')
    parser.add_argument('--step', '-s', type=int, default=0,
                        help='Run only up to this step (0 = all steps)')
    args = parser.parse_args()

    pol, isotrak_to_head, hpi_orig_head = step1_polhemus(args.pol)
    if args.step == 1:
        return

    raw, hpi_names, hpi_indices = step2_raw(args.hpi, args.freq)
    if args.step == 2:
        return

    windows = step3_peaks(raw, hpi_indices, hpi_names, args.freq)
    if args.step == 3:
        return

    hpi_dev, hpi_gofs, raw_setup = step4_amplitudes(
        raw, hpi_indices, hpi_names, hpi_orig_head,
        pol, isotrak_to_head, windows, args.freq
    )
    if hpi_dev is None or args.step == 4:
        return

    dev_to_head = step5_matching(hpi_dev, hpi_gofs, hpi_orig_head, hpi_names)
    if args.step == 5:
        return

    step6_plot(hpi_dev, hpi_gofs, hpi_orig_head, hpi_names,
               pol, dev_to_head, raw)


if __name__ == '__main__':
    main()
