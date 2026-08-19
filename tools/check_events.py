# Author: C Pfeiffer

import mne
import numpy as np
from collections import defaultdict
import re
from os.path import isfile

from opm_utility_scripts.io import get_file


def extract_events_from_raw(raw, stim_channel='di1'):

    if stim_channel in raw.ch_names:
        print("Channel 'stim' found. Extracting events...")
        events = mne.find_events(raw, stim_channel=stim_channel, shortest_event=1)
    else:
        print("Channel not found. Searching for 'ai' channels...")
        print(raw.ch_names)
        ai_channels = [ch for ch in raw.ch_names if ch.startswith('ai')]
        print(ai_channels)
        if not ai_channels:
            raise ValueError("No trigger channels found in the data.")

        print(f"Found 'ai' channels: {ai_channels}")

        ai_data = raw.copy().pick_channels(ai_channels).get_data()
        ai_values = ai_data > 2.5

        suffixes = [int(ch[2:]) for ch in ai_channels]
        sorted_indices = np.argsort(suffixes)
        ai_values_sorted = ai_values[sorted_indices]

        bit_values = 2 ** np.arange(len(ai_values_sorted))
        combined_values = np.dot(ai_values_sorted.T, bit_values)

        changes = np.diff(combined_values, prepend=combined_values[0])
        event_onsets = np.where(changes > 0)[0]

        valid_onsets = []
        valid_codes = []
        for onset in event_onsets:
            current_code = combined_values[onset]
            duration = 1
            for i in range(onset + 1, len(combined_values)):
                if combined_values[i] == current_code:
                    duration += 1
                else:
                    break
            if duration >= 10:
                valid_onsets.append(onset)
                valid_codes.append(current_code)

        events = np.column_stack((valid_onsets, np.zeros(len(valid_onsets), dtype=int), valid_codes))

    return events


def check_events(fif_path, stim_channel):
    raw = mne.io.read_raw_fif(fif_path, preload=False, allow_maxshield=True)

    events = extract_events_from_raw(raw, stim_channel=stim_channel)

    event_codes = events[:, 2]
    event_times = events[:, 0] / raw.info['sfreq']

    print("Sequence of event codes:")

    summary = defaultdict(lambda: {'count': 0, 'to_prev': [], 'to_next': []})

    durations_to_prev = np.diff(event_times, prepend=np.nan)
    durations_to_next = np.diff(event_times, append=np.nan)

    for i, code in enumerate(event_codes):
        summary[code]['count'] += 1
        if i > 0:
            summary[code]['to_prev'].append(durations_to_prev[i])
        if i < len(event_codes) - 1:
            summary[code]['to_next'].append(durations_to_next[i])

    print("\nSummary per event code:")
    for code, stats in summary.items():
        to_prev = np.array(stats['to_prev'])
        to_next = np.array(stats['to_next'])

        median_prev = np.nanmedian(to_prev) if len(to_prev) > 0 else np.nan
        median_next = np.nanmedian(to_next) if len(to_next) > 0 else np.nan

        max_prev = np.percentile(to_prev[to_prev <= 2 * median_prev], 95) if len(to_prev) > 0 else 0
        max_next = np.percentile(to_next[to_next <= 2 * median_next], 95) if len(to_next) > 0 else 0
        min_prev = np.percentile(to_prev[to_prev <= 2 * median_prev], 5) if len(to_prev) > 0 else 0
        min_next = np.percentile(to_next[to_next <= 2 * median_next], 5) if len(to_next) > 0 else 0

        print(f"Code {hex(code)}: n={stats['count']}; ITI_post = {median_next:.3f}s ({min_next:.3f}-{max_next:.3f}s); ITI_pre = {median_prev:.3f}s ({min_prev:.3f}-{max_prev:.3f}s)")

    all_to_prev = durations_to_prev[1:]
    all_to_next = durations_to_next[:-1]

    median_all_prev = np.nanmedian(all_to_prev)
    median_all_next = np.nanmedian(all_to_next)

    max_all_prev = np.nanmax(all_to_prev[all_to_prev <= 2 * median_all_prev])
    max_all_next = np.nanmax(all_to_next[all_to_next <= 2 * median_all_next])

    print(f"All Events: n={len(event_codes)}; ITI = {median_all_next:.3f}s ({np.nanmin(all_to_next):.3f} - {max_all_next:.3f}s)")


def _args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="Check events in a FIF file.")
    parser.add_argument('--file', type=str, help="Path to the .fif file")
    parser.add_argument('--stim', type=str, help="Select stim channel")
    return parser.parse_args()


if __name__ == "__main__":
    args = _args_parser()
    if args.file:
        fif_path = args.file
    else:
        fif_path = get_file("Select FIF file")
    if not fif_path or not isfile(fif_path):
        print("Invalid file path. Please provide a valid .fif file.")
    else:
        stim = args.stim if args.stim else 'di38'
        if args.stim:
            print(stim)
        check_events(fif_path, stim_channel=stim)
