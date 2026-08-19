# Author: C Pfeiffer (adapted from script by T Cheung)
# Last Modified Jan 28, 2025
# Function for displaying OPM-MEG helmetscan in 2D
#
# When executed the user is prompted to select a CSV file containing the
# results from the helmet scan.  Empty slots are marked red, occupied slots
# are marked green and labelled with the name of the channel the sensor is
# connected to.

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio

# Default mat file lives alongside this script.
_DEFAULT_MAT = Path(__file__).parent / 'hedscan_layout.mat'


def select_file():
    root = tk.Tk()
    root.withdraw()
    default_path = "/home/administrator/.local/share/HEDscan"
    file_path = filedialog.askopenfilename(initialdir=default_path, filetypes=[("CSV files", "*locations.csv")])
    if not file_path.endswith('_helmetscan_locations.csv'):
        print("ERROR: Wrong file type. The file should end with '_helmetscan_locations.csv'.")
        sys.exit()
    return file_path


def plot_helmetscan(csv_path=None, mat_file_path=None):
    """
    Plot a 2D helmetscan layout.

    Parameters
    ----------
    csv_path : str, optional
        Path to the ``_helmetscan_locations.csv`` file.  Opens a file dialog
        if not provided.
    mat_file_path : str or Path, optional
        Path to the ``hedscan_layout.mat`` file.  Defaults to the copy
        bundled alongside this script.
    """
    if csv_path is None:
        csv_path = select_file()

    if mat_file_path is None:
        mat_file_path = _DEFAULT_MAT

    sensor_df = pd.read_csv(csv_path)
    datestring = os.path.basename(csv_path).split('_')[0:2]

    mat_data = sio.loadmat(mat_file_path)
    positions = mat_data['layout']['pos'][0, 0]
    labels = mat_data['layout']['label'][0, 0]
    outlines = mat_data['layout']['outline'][0, 0][:]

    labels = [label[0] for label in labels.flatten()]

    plt.figure(figsize=(10, 8))

    for outline in outlines:
        for line in outline:
            x_points, y_points = zip(*line)
            plt.plot(x_points, y_points, 'k-')

    for i, label in enumerate(labels):
        x, y = positions[i]
        match = sensor_df.iloc[0:, 2].str.contains(label[:4])
        if match.any():
            sensor_name = sensor_df.iloc[0:, :][match].iloc[0, 0]
            sensor_name_disp = sensor_name.split('_')[0]
            plt.plot(x, y, 'go', markersize=10)
            plt.text(x, y + 0.08, sensor_name_disp, fontsize=9, ha='center')
        else:
            plt.plot(x, y, 'ro', markersize=10)

    plt.title(f'Helmetscan - {datestring[0]}-{datestring[1]}')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    plot_helmetscan()
