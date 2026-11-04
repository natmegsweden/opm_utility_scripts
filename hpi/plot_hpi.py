#!/usr/bin/env python3
"""Run the HPI checker against the newest files matching supplied globs.

Example::

    python -m opm_utility_scripts.hpi.plot_hpi \
        --hpi-glob '/data/sub-01/**/HPI*raw.fif' \
        --pol-glob '/data/sub-01/**/digitisation*.json' --freq 33

Additional options are passed through to :mod:`opm_utility_scripts.hpi.check`.
"""

import argparse
import glob
import os
import sys

from . import check


def _newest_match(pattern):
    pattern = os.path.expanduser(pattern)
    matches = [path for path in glob.glob(pattern, recursive=True)
               if os.path.isfile(path)]
    if not matches:
        raise FileNotFoundError(f'No files match pattern: {pattern}')

    def created_at(path):
        stat = os.stat(path)
        return getattr(stat, 'st_birthtime', stat.st_mtime)

    return max(matches, key=created_at)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Select the newest matching HPI/Polhemus files and plot their check.'
    )
    parser.add_argument('--hpi-glob', required=True,
                        help='Glob for HPI FIF files (supports ** recursive matching).')
    parser.add_argument('--pol-glob',
                        help='Optional glob for Polhemus files; omit for HPI-only mode.')
    args, check_args = parser.parse_known_args(argv)

    try:
        hpi_file = _newest_match(args.hpi_glob)
        forwarded = ['--hpi', hpi_file]
        if args.pol_glob:
            pol_file = _newest_match(args.pol_glob)
            forwarded.extend(['--pol', pol_file])
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f'Selected HPI file: {hpi_file}')
    if args.pol_glob:
        print(f'Selected Polhemus file: {pol_file}')
    check.main([*forwarded, *check_args])


if __name__ == '__main__':
    main()
