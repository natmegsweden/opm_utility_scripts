import tkinter as tk
from tkinter import filedialog
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
import os

# Function to open a file dialog and select the CSV file
def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return file_path

# Load the CSV file containing sensor definitions
csv_path = select_file()
sensor_df = pd.read_csv(csv_path)

# Extract the datestring from the filename
datestring = os.path.basename(csv_path).split('_')[0:2]

# Hard-wired path to the .mat file containing 2D MEG sensor layout
mat_file_path = '/Users/christophpfeiffer/src/plot_helmetscan/hedscan_layout.mat'

# Load the .mat file
mat_data = sio.loadmat(mat_file_path)
positions = mat_data['layout']['pos'][0,0]  # Assuming the variable name in the .mat file is 'positions'
labels = mat_data['layout']['label'][0,0]  # Assuming the variable name in the .mat file is 'labels'
outlines = mat_data['layout']['outline'][0,0][:]  # Assuming the variable name in the .mat file is 'outline'

# Convert labels to a list of strings
labels = [label[0] for label in labels.flatten()]

# Create a figure for plotting
plt.figure(figsize=(10, 8))

for outline in outlines:
    for line in outline:
        x_points, y_points = zip(*line)
        plt.plot(x_points, y_points, 'k-')  # Plot outline with black lines

# Plot green filled circles for matching sensors and red filled circles for non-matching sensors
for i, label in enumerate(labels):
    x, y = positions[i]
    match = sensor_df.iloc[1:, 2].str.contains(label[:4])
    if match.any():
        sensor_name = sensor_df.iloc[1:, :][match].iloc[0, 0]
        sensor_name_disp = sensor_name.split('_')[0]
        plt.plot(x, y, 'go', markersize=10)  # Green filled circle
        plt.text(x, y+0.08, sensor_name_disp, fontsize=9, ha='center')
    else:
        plt.plot(x, y, 'ro', markersize=10)  # Red filled circle

# Set plot title and labels
plt.title(f'Helmetscan - {datestring[0]}-{datestring[1]}')

# Remove axes
plt.axis('off')

# Show plot
plt.show()



# mat_file_path = '/Users/christophpfeiffer/src/plot_helmetscan/fieldlinebeta2bz_helmet.mat'