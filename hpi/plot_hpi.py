#!/usr/bin/env python3
"""Run the HPI checker against the newest files matching supplied globs.

Example::

    python -m opm_utility_scripts.hpi.plot_hpi \
        --hpi-glob '/data/sub-01/**/HPI*raw.fif' \
        --pol-glob '/data/sub-01/**/digitisation*.json' \
        --localization-grid medium

Additional options are passed through to :mod:`opm_utility_scripts.hpi.check`.
"""

import argparse
import glob
import os
import re
import sys

from . import check


# Hardcoded exclude pattern: a "-proc" entity in the filename denotes an
# already processed/derivative recording (e.g. a BIDS-style "proc-sss" /
# "proc-tsss" entity, typically written as "..._proc-sss_raw.fif") and must
# never be picked up as the source HPI/Polhemus file. The separator in front
# of "proc" is usually "_" (BIDS entity separator) rather than a literal
# "-", so both are matched here — a plain substring check for "-proc" alone
# misses the common "_proc-..." naming actually seen in practice.
_EXCLUDE_RE = re.compile(r'[_-]proc', re.IGNORECASE)


def _is_excluded(path):
    return _EXCLUDE_RE.search(os.path.basename(path)) is not None


def _case_insensitive_pattern(pattern):
    """Translate a glob pattern into a case-insensitive equivalent.

    ``glob`` has no built-in case-insensitivity flag, so each alphabetic
    character is rewritten as a ``[aA]``-style bracket expression, which
    ``glob``/``fnmatch`` already supports.
    """
    return ''.join(
        f'[{char.lower()}{char.upper()}]' if char.isalpha() else char
        for char in pattern
    )


def _sorted_matches(pattern, extensions=None):
    """Return every file matching ``pattern``, newest first.

    Parameters
    ----------
    extensions : tuple[str] or None
        If given, matches are filtered to these extensions (case-insensitive)
        before sorting. Without this filter, a glob that happens to also
        catch an unrelated sidecar file (e.g. a ``.json`` metadata file
        sitting next to the ``.fif`` recording) can "win" the newest-file
        comparison simply because it was written later, even though it is
        the wrong file type for this argument.
    """
    pattern = os.path.expanduser(pattern)
    ci_pattern = _case_insensitive_pattern(pattern)
    matches = [path for path in glob.glob(ci_pattern, recursive=True)
               if os.path.isfile(path)]
    if extensions is not None:
        matches = [path for path in matches
                   if path.lower().endswith(tuple(ext.lower() for ext in extensions))]
    matches = [path for path in matches if not _is_excluded(path)]
    if not matches:
        raise FileNotFoundError(
            f"No files match pattern: {pattern} "
            f"(filenames with a '-proc'/'_proc' entity are excluded)"
        )

    def created_at(path):
        stat = os.stat(path)
        return getattr(stat, 'st_birthtime', stat.st_mtime)

    return sorted(matches, key=created_at, reverse=True)


def _newest_match(pattern, extensions=None):
    """Return the single newest file matching ``pattern``.

    See :func:`_sorted_matches` for the ``extensions`` filter rationale.
    """
    return _sorted_matches(pattern, extensions=extensions)[0]


def _validate_hpi_channels(path):
    """Fail fast if the selected file has no active HPI channels.

    Checks for channels whose name starts with ``hpiin`` or ``hpiout``
    (case-insensitive) that also carry an actual signal (non-flat data),
    mirroring the active-channel criterion used by
    :func:`opm_utility_scripts.channels.get_hpi_output_channels`. Without
    this secondary check, picking the wrong raw file (e.g. a resting-state
    or empty-room recording with no HPI coils driven) fails deep inside the
    fitting pipeline with a far less actionable error.
    """
    import mne

    info = mne.io.read_info(path, verbose='error')
    candidates = [name for name in info['ch_names']
                 if name.lower().startswith(('hpiin', 'hpiout'))]
    if not candidates:
        raise ValueError(
            f'No HPI channels (hpiin*/hpiout*) found in {path}. '
            'This does not look like an HPI recording.'
        )

    raw = mne.io.read_raw_fif(path, preload=False, verbose='error')
    non_empty = [name for name in candidates
                if raw.get_data(picks=[name]).var() > 1e-25]
    if not non_empty:
        raise ValueError(
            f'HPI channel(s) found in {path} ({", ".join(candidates)}) but all '
            'are empty/flat (no signal). Check the HPI coil drive and recording.'
        )


# Substring of the ValueError message raised by fit_hpi_amplitudes (in
# ``_core.py``) when no HPI coil's drive signal has a detectable peak. On
# this specific failure we retry with the next-newest HPI candidate rather
# than giving up, since it usually just means the wrong/unfinished file was
# picked up (e.g. HPI not yet activated in that particular recording).
_NO_PEAKS_MSG = 'No HPI coils had detectable peaks'


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Select the newest matching HPI/Polhemus files and plot their check.'
    )
    parser.add_argument('--hpi-glob', required=True,
                        help='Glob for HPI FIF files (supports ** recursive matching).')
    parser.add_argument('--pol-glob', default=None,
                        help='Optional glob for Polhemus files; omit for HPI-only mode.')
    parser.add_argument('--localization-grid',
                        choices=['coarse', 'medium', 'fine', 'mne_stock'],
                    default='medium',
                        help=('Dipole OPM search grid to use for localization. '
                        'Default: medium.'))
    args, check_args = parser.parse_known_args(argv)

    # This wrapper intentionally fixes frequency handling to check defaults.
    # Reject legacy --freq passthrough explicitly for clarity.
    if '--freq' in check_args or any(arg.startswith('--freq=') for arg in check_args):
        parser.error('--freq is not supported by plot-hpi; use opmutil check/coregister if you need custom frequency.')

    try:
        hpi_candidates = _sorted_matches(args.hpi_glob, extensions=('.fif',))
        pol_file = None
        if args.pol_glob:
            pol_file = _newest_match(args.pol_glob, extensions=('.fif', '.json'))
    except FileNotFoundError as exc:
        parser.error(str(exc))

    # Suppress the bad-channel diagnostic figure in this convenience
    # wrapper unless the user explicitly selects a policy.
    has_bad_channel_policy = any(
        arg == '--bad-channel-policy' or arg.startswith('--bad-channel-policy=')
        for arg in check_args
    )

    if args.pol_glob:
        print(f'Selected Polhemus file: {pol_file}')

    for hpi_file in hpi_candidates:
        try:
            _validate_hpi_channels(hpi_file)
        except ValueError as exc:
            print(f'Skipping {hpi_file}: {exc}', file=sys.stderr)
            continue

        forwarded = ['--hpi', hpi_file]
        if pol_file:
            forwarded.extend(['--pol', pol_file])
        forwarded.extend(['--localization-grid', args.localization_grid])
        if not has_bad_channel_policy:
            forwarded.extend(['--bad-channel-policy', 'none'])

        print(f'Selected HPI file: {hpi_file}')
        try:
            check.main([*forwarded, *check_args], raise_on_fit_error=True)
            return
        except ValueError as exc:
            if _NO_PEAKS_MSG not in str(exc):
                raise
            print(f'No detectable HPI peaks in {hpi_file}; trying next file.',
                 file=sys.stderr)
            continue

    parser.error(
        f'No HPI file matching {args.hpi_glob!r} produced detectable HPI peaks.'
    )


if __name__ == '__main__':
    main()
