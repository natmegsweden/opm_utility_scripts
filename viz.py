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
