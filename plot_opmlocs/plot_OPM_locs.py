#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:20:07 2024

@author: christophpfeiffer
"""
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import scipy.io

root = tk.Tk()
root.withdraw()

csv_file = filedialog.askopenfilename()

#%%
# Read the CSV file
df = pd.read_csv('/Users/christophpfeiffer/src/plot_opmlocs/20240902_153021_helmetscan_locations.csv')

# Extract the columns
channel = df['Channel'].to_numpy()
coil_name = df['Coil Name'].to_numpy()
sensors = df[['sensor_x', 'sensor_y', 'sensor_z']].to_numpy()

#%%
tmp = scipy.io.loadmat('/Users/christophpfeiffer/src/plot_opmlocs/fieldlinebeta2bz_helmet.mat')
layout = tmp['layout']