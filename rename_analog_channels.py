#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 13:21:07 2026

@author: christophpfeiffer
"""
import mne
import json
from os.path import isfile

def rename_channels(path, mapping_file, newpath):
    # Read data
    raw = mne.io.read_raw(path,preload=False)
    
    # Read mapping file
    with open(mapping_file) as json_file:
        mapping = json.load(json_file)
        
    # Rename 
    for oldname,new in mapping.items():
        for chi,ch in enumerate(raw.info["chs"]):
            if ch["ch_name"] == oldname:
                ch["ch_name"] = new["newname"] #Rename
                ch["kind"] = new["type"]["kind"] #Change kind
                ch["unit"] = new["type"]["unit"] # Change unit
                
        for chi,ch_name in enumerate(raw.info["ch_names"]):
            #print(oldname)
            if ch_name == oldname:
                raw.info["ch_names"][chi] = new["newname"] #Rename
    
    raw.load_data()
    raw.save(newpath,overwrite=True)
    print('done!')
        
def args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Rename analog channels in a raw FIF file.")
    parser.add_argument('--file', type=str, help="Path to the raw (fif) file")
    parser.add_argument('--newfile', type=str, help="Path to save new raw (fif) file. Otherwise file will be overwritten.")
    parser.add_argument('--map', type=str, help="Select analog channel mapping file")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parser()
    if args.file:
        fif_path = args.file
    if not fif_path or not isfile(fif_path):
        print("Invalid or missing file path. Please provide a valid raw (fif) file.")
    if not args.map:
        args.map = 'analog_channel_mapping.json'
    else:
        print(args.map)
    if args.newfile:
        newfile = args.newfile
    else:
        newfile = fif_path
        print('Overwriting raw file')
        
    rename_channels(fif_path, args.map, newfile)
