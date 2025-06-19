
import mne
import numpy as np
from collections import defaultdict
import re

def extract_events_from_raw(raw):
    if 'di38' in raw.ch_names:
        print("Channel 'di38' found. Extracting events...")
        events = mne.find_events(raw, stim_channel='di38')
    else:
        print("Channel 'di38' not found. Searching for 'ai' channels...")
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

        bit_values = 2 ** np.arange(len(ai_values))
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


def check_events(fif_path):
    # Load the raw data
    raw = mne.io.read_raw_fif(fif_path, preload=False)
    
    # Pick only the trigger channel 'di38'
    #raw.pick_channels(['di38'])
    # Find events from the trigger channel
    #events = mne.find_events(raw, stim_channel='di38', shortest_event=1, verbose=False)

    events = extract_events_from_raw(raw)

    # Extract event codes and times
    event_codes = events[:, 2]
    event_times = events[:, 0] / raw.info['sfreq']  # convert sample indices to time in seconds

    # Print the sequence of event codes
    print("Sequence of event codes:")
    print(event_codes.tolist())

    # Initialize summary dictionary
    summary = defaultdict(lambda: {'count': 0, 'to_prev': [], 'to_next': []})

    # Calculate durations to previous and next events (any type)
    durations_to_prev = np.diff(event_times, prepend=np.nan)
    durations_to_next = np.diff(event_times, append=np.nan)

    # Populate summary for each event code
    for i, code in enumerate(event_codes):
        summary[code]['count'] += 1
        if i > 0:
            summary[code]['to_prev'].append(durations_to_prev[i])
        if i < len(event_codes) - 1:
            summary[code]['to_next'].append(durations_to_next[i])

    # Print summary for each unique event code
    print("\nSummary per event code:")
    for code, stats in summary.items():
        to_prev = np.array(stats['to_prev'])
        to_next = np.array(stats['to_next'])

        # Compute medians
        median_prev = np.nanmedian(to_prev) if len(to_prev) > 0 else np.nan
        median_next = np.nanmedian(to_next) if len(to_next) > 0 else np.nan

        # Filter out durations > 2 * median for max calculation
        max_prev = np.nanmax(to_prev[to_prev <= 2 * median_prev]) if len(to_prev) > 0 else np.nan
        max_next = np.nanmax(to_next[to_next <= 2 * median_next]) if len(to_next) > 0 else np.nan

        #print(f"Event Code: {code}")
        #print(f"  Count: {stats['count']}")
        print(f"Code {code}: n={stats['count']}; ITI_post = {median_next:.3f}s ({np.nanmin(to_next):.3f}-{max_next:.3f}s); ITI_pre = {median_prev:.3f}s ({np.nanmin(to_prev):.3f}-{max_prev:.3f}s)")

    # Combined summary for all events
    all_to_prev = durations_to_prev[1:]  # exclude first NaN
    all_to_next = durations_to_next[:-1]  # exclude last NaN

    median_all_prev = np.nanmedian(all_to_prev)
    median_all_next = np.nanmedian(all_to_next)

    max_all_prev = np.nanmax(all_to_prev[all_to_prev <= 2 * median_all_prev])
    max_all_next = np.nanmax(all_to_next[all_to_next <= 2 * median_all_next])

    #print("\nAll Events:")
    #print(f"  Total Events: {len(event_codes)}")
    #print(f"  To Previous - Median: {median_all_prev:.3f}s, Max: {max_all_prev:.3f}s, Min: {np.nanmin(all_to_prev):.3f}s")
    print(f"All Events: n={len(event_codes)}; ITI = {median_all_next:.3f}s ({np.nanmin(all_to_next):.3f} - {max_all_next:.3f}s)")
    
# Example usage:
#check_events('/Volumes/dataarchvie/21099_opm/MEG/NatMEG_0953/241104/osmeg/PhalangesOPM_raw.fif')

check_events('/Volumes/dataarchvie/CHOP/MEG/SBIRA27/241127/20241127_135607_sub-SBIRA27_file-VarITINoWire_raw.fif')
