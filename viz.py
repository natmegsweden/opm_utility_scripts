"""Visualisation utilities for OPM-MEG data."""

import os

import matplotlib.pyplot as plt
import mne
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial import Delaunay


def plot_psd(raw):
    """
    Generate power spectral density plots for MEG data quality assessment.

    Creates dual PSD plots (with/without projections) using Hann windowing
    for frequency domain analysis of raw MEG data.

    Args:
        raw (mne.io.Raw): Raw MEG data object

    Returns:
        tuple: (fig_proj_on, fig_proj_off) - matplotlib figure objects

    Side Effects:
        - Displays PSD plots with logarithmic scaling
        - Shows frequency range 0-500 Hz, power range 1-10000
    """
    fname = os.path.basename(raw.filenames[0])
    n_fft = 1024
    psd_ylim = [1.0, 10000.0]
    psd_xlim = [0.0, 500.0]

    projs = 0
    fig = raw.plot_psd(fmin=0, n_fft=n_fft, show=False, proj=True, dB=False, xscale='log', window='hann', n_jobs=-1)
    fig3 = raw.plot_psd(fmin=0, n_fft=n_fft, show=False, proj=False, dB=False, xscale='log', window='hann', n_jobs=-1)

    fig.suptitle('%s %d projs on hann' % (fname, projs))
    fig3.suptitle('%s projs off hann' % fname)
    fig.axes[0].set_yscale('log')
    fig3.axes[0].set_yscale('log')

    fig.axes[0].set_ylim(psd_ylim)
    fig.axes[0].set_xlim(psd_xlim)
    fig3.axes[0].set_ylim(psd_ylim)
    fig3.axes[0].set_xlim(psd_xlim)

    fig.subplots_adjust(0.1, 0.1, 0.95, 0.85)
    fig3.subplots_adjust(0.1, 0.1, 0.95, 0.85)

    plt.show()

    return fig, fig3


def plot_3d(plot_params: dict, filename: str):
    """
    Create 3D visualisation of sensor array, HPI coils, and digitisation points.

    Generates a static 3D plot saved to *filename* showing:
    - MEG sensor positions with a triangulated mesh surface
    - HPI coil locations in device coordinates (blue)
    - HPI coil locations in head coordinates (green)
    - Digitisation points (black)

    Args:
        plot_params (dict): Keys: senspos, senslabel, hpipos, hpilabel,
            hpipos2, hpilabel2, digpos.
        filename (str): Output image path (PNG recommended).

    Returns:
        None
    """
    senspos = plot_params.get('senspos', None)
    senslabel = plot_params.get('senslabel', None)
    hpipos = plot_params.get('hpipos', None)
    hpilabel = plot_params.get('hpilabel', None)
    hpipos2 = plot_params.get('hpipos2', None)
    hpilabel2 = plot_params.get('hpilabel2', None)
    digpos = plot_params.get('digpos', None)

    senspos = np.array(senspos)
    senslabel = np.array(senslabel)
    hpipos = np.array(hpipos)
    hpilabel = np.array(hpilabel)

    center_of_mass = np.mean(senspos, axis=0)
    senspos_centered = senspos - center_of_mass
    r = np.linalg.norm(senspos_centered, axis=1)
    theta = np.arccos(senspos_centered[:, 2] / r)
    phi = np.arctan2(senspos_centered[:, 0], senspos_centered[:, 1])
    x_proj = theta * np.cos(phi)
    y_proj = theta * np.sin(phi)
    polar_proj = np.vstack((x_proj, y_proj)).T

    tri = Delaunay(polar_proj)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_trisurf(
        senspos[:, 0], senspos[:, 1], senspos[:, 2],
        triangles=tri.simplices, cmap='viridis', alpha=0.6,
        edgecolor='k', linewidth=0.2,
    )

    ax.scatter(senspos[:, 0], senspos[:, 1], senspos[:, 2], color='r', s=50)
    for i in range(len(senslabel)):
        ax.text(senspos[i, 0], senspos[i, 1], senspos[i, 2], senslabel[i], color='black', fontsize=9)

    ax.scatter(hpipos[:, 0], hpipos[:, 1], hpipos[:, 2], color='b', s=100)
    for i in range(len(hpilabel)):
        ax.text(hpipos[i, 0], hpipos[i, 1], hpipos[i, 2], hpilabel[i], color='blue')

    ax.scatter(hpipos2[:, 0], hpipos2[:, 1], hpipos2[:, 2], color='g', s=100)
    for i in range(len(hpilabel)):
        ax.text(hpipos2[i, 0], hpipos2[i, 1], hpipos2[i, 2], hpilabel2[i], color='green')

    ax.scatter(digpos[:, 0], digpos[:, 1], digpos[:, 2], color='k', s=10)

    ax.view_init(elev=10, azim=20)
    fig.savefig(filename, dpi=300, bbox_inches='tight')


def plot_hpi_alignment(fit: dict, raw=None, show: bool = True, filename: str = None):
    """
    Plot the alignment between fitted HPI coil positions and Polhemus positions.

    Both sets are shown in **head coordinates** in three orthogonal views
    (top / right / front).  Connecting lines between matched pairs are labelled
    with the residual distance in mm so it is immediately obvious which coil
    is off and by how much.

    Args:
        fit (dict):
            Result dict from :func:`opm_utility_scripts.hpi._core.fit_hpi`.
            Required keys: ``hpi_dev``, ``hpi_gofs``, ``hpi_orig``,
            ``hpi_names``, ``dev_to_head_trans``, ``include_hpis``,
            ``tree_indices``, ``nasion``, ``lpa``, ``rpa``,
            ``extra_pts``.  All position arrays must be in **head**
            coordinates (metres) — this is the case for all fit dicts
            returned by ``fit_hpi``.
        raw (mne.io.Raw | None):
            The HPI raw object (or any raw with MEG channels in device
            space).  When provided, sensor positions are transformed to
            head space and plotted as a grey reference cloud.
        show (bool):
            Call ``plt.show()`` after building the figure (default True).
        filename (str | None):
            When given, save the figure to this path (PNG recommended)
            in addition to showing it.

    Returns:
        matplotlib.figure.Figure
    """
    from mne.transforms import apply_trans
    import mne as _mne

    hpi_dev          = np.array(fit['hpi_dev'])        # device frame
    hpi_gofs         = np.array(fit['hpi_gofs'])
    hpi_orig         = np.array(fit['hpi_orig'])       # head frame (Polhemus)
    hpi_names        = fit['hpi_names']
    dev_to_head      = fit['dev_to_head_trans']
    include_hpis     = np.array(fit['include_hpis'])   # bool mask
    tree_indices     = np.array(fit['tree_indices'])   # polhemus index per included coil
    nasion           = np.array(fit['nasion'])          # head frame
    lpa              = np.array(fit['lpa'])
    rpa              = np.array(fit['rpa'])
    extra_pts        = np.array(fit['extra_pts'])      # head frame

    # Fitted coil positions in head frame (mm)
    hpi_fitted_head  = apply_trans(dev_to_head, hpi_dev) * 1000
    hpi_pol_head     = hpi_orig * 1000                  # Polhemus targets (mm)
    extra_mm         = extra_pts * 1000 if len(extra_pts) else None

    # Fiducials (mm)
    fids = {
        'LPA':    lpa    * 1000,
        'Nasion': nasion * 1000,
        'RPA':    rpa    * 1000,
    }

    # Sensor cloud in head frame (mm)
    sensor_head_mm = None
    if raw is not None:
        meg_picks = _mne.pick_types(raw.info, meg=True, exclude=[])
        sensor_dev = np.array([raw.info['chs'][i]['loc'][:3] for i in meg_picks])
        sensor_head_mm = apply_trans(dev_to_head, sensor_dev) * 1000

    # ------------------------------------------------------------------ #
    views = [
        dict(title='top',   elev=90,  azim=-90),
        dict(title='right', elev=0,   azim=0),
        dict(title='front', elev=0,   azim=90),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                             subplot_kw=dict(projection='3d'))

    for ax, v in zip(axes, views):

        # Sensor cloud
        if sensor_head_mm is not None:
            ax.scatter(sensor_head_mm[:, 0],
                       sensor_head_mm[:, 1],
                       sensor_head_mm[:, 2],
                       c='lightgray', s=4, alpha=0.25, zorder=1)

        # Headshape
        if extra_mm is not None and len(extra_mm):
            ax.scatter(extra_mm[:, 0], extra_mm[:, 1], extra_mm[:, 2],
                       c='gainsboro', s=3, alpha=0.3, zorder=1)

        # Fiducials
        fid_colors = {'LPA': 'darkorange', 'Nasion': 'limegreen', 'RPA': 'darkorange'}
        for fname_f, fpos in fids.items():
            ax.scatter(*fpos, c=fid_colors[fname_f], s=60, marker='^',
                       zorder=5, depthshade=False)
            ax.text(fpos[0], fpos[1], fpos[2], f' {fname_f}',
                    fontsize=7, color=fid_colors[fname_f], zorder=6)

        # Polhemus coil targets — blue stars
        ax.scatter(hpi_pol_head[:, 0], hpi_pol_head[:, 1], hpi_pol_head[:, 2],
                   c='royalblue', s=120, marker='*', zorder=7,
                   depthshade=False, label='Polhemus (target)')

        # Fitted coil positions — circles, colour-coded by GOF
        for i, (pos, gof) in enumerate(zip(hpi_fitted_head, hpi_gofs)):
            color = 'red' if gof > 0.9 else 'orange'
            ax.scatter(*pos, c=color, s=80, marker='o', zorder=7,
                       depthshade=False)
            short = hpi_names[i][-3:] if len(hpi_names[i]) >= 3 else hpi_names[i]
            ax.text(pos[0], pos[1], pos[2],
                    f' {short}\n GOF={gof:.2f}',
                    fontsize=7, color=color, zorder=8)

        # Connecting lines for matched pairs, labelled with residual distance
        for k, (dev_i, pol_i) in enumerate(
                zip(np.where(include_hpis)[0], tree_indices)):
            p_fit = hpi_fitted_head[dev_i]
            p_pol = hpi_pol_head[pol_i]
            dist_mm = np.linalg.norm(p_fit - p_pol)
            mid = (p_fit + p_pol) / 2

            color = 'green' if dist_mm < 5 else ('orange' if dist_mm < 10 else 'red')
            ax.plot([p_fit[0], p_pol[0]],
                    [p_fit[1], p_pol[1]],
                    [p_fit[2], p_pol[2]],
                    color=color, lw=1.2, linestyle='--', zorder=6)
            ax.text(mid[0], mid[1], mid[2],
                    f'{dist_mm:.1f} mm',
                    fontsize=7, color=color, zorder=9,
                    ha='center', va='bottom')

        ax.view_init(elev=v['elev'], azim=v['azim'])
        ax.set_title(v['title'], fontsize=10)
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        ax.set_zlabel('z (mm)', fontsize=8)

    # Shared legend on first axis
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='royalblue',
               markersize=10, label='Polhemus target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=8, label='Fitted (GOF > 0.9)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markersize=8, label='Fitted (GOF ≤ 0.9)'),
        Line2D([0], [0], linestyle='--', color='green',
               label='Match < 5 mm'),
        Line2D([0], [0], linestyle='--', color='orange',
               label='Match 5–10 mm'),
        Line2D([0], [0], linestyle='--', color='red',
               label='Match > 10 mm'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper left',
                   fontsize=7, framealpha=0.7)

    # Summary title
    included_dists = []
    for k, (dev_i, pol_i) in enumerate(
            zip(np.where(include_hpis)[0], tree_indices)):
        included_dists.append(
            np.linalg.norm(hpi_fitted_head[dev_i] - hpi_pol_head[pol_i])
        )
    mean_dist = np.mean(included_dists) if included_dists else float('nan')
    n_good = int(include_hpis.sum())
    n_total = len(hpi_gofs)
    fig.suptitle(
        f'HPI alignment — {n_good}/{n_total} coils included (GOF > 0.9)  '
        f'|  mean residual = {mean_dist:.1f} mm',
        fontsize=11,
    )

    plt.tight_layout()

    if filename:
        fig.savefig(filename, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def rotate_points(points, target_vector):
    """
    Rotate *points* so that the local z-axis aligns with *target_vector*.

    Uses Rodrigues' rotation formula.

    Args:
        points (np.ndarray): (N, 3) array of points to rotate.
        target_vector (np.ndarray): 3-element target direction vector.

    Returns:
        np.ndarray: Rotated points, same shape as input.
    """
    target_vector = target_vector / np.linalg.norm(target_vector)
    original_vector = np.array([0, 0, 1])

    if np.allclose(original_vector, target_vector):
        return points

    rotation_axis = np.cross(original_vector, target_vector)
    angle = np.arccos(np.dot(original_vector, target_vector))
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0],
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    return points @ R.T


def create_aligned_grid(loc, step_size, distXY, distZ):
    """
    Build a disc-shaped grid of points aligned to the sensor normal direction.

    Args:
        loc (array-like): 12-element sensor location vector (MNE convention).
            Positions are ``loc[:3]``; z-orientation is ``loc[9:]``.
        step_size (float): Grid point spacing (metres).
        distXY (float): Disc radius (metres).
        distZ (float): Depth of the grid below the sensor surface (metres).

    Returns:
        np.ndarray: (M, 3) array of grid points in device coordinates.
    """
    position = np.array(loc[:3])
    orientation_z = np.array(loc[9:])

    x = np.arange(-distXY * 2, distXY * 2, step_size)
    y = np.arange(-distXY * 2, distXY * 2, step_size)
    z = -np.arange(0, distZ, step_size)

    X, Y, Z = np.meshgrid(x, y, z)
    grid_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    grid_points = grid_points[np.linalg.norm(grid_points[:, :2], axis=1) < distXY, :]

    return rotate_points(grid_points, orientation_z) + position
