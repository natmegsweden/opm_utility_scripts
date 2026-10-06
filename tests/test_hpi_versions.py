#!/usr/bin/env python3
"""Run comparable HPI fits using the documented v0.1.0/0.2.0/0.3.0 mappings.

This is an executable comparison utility, not an automated unit test. The
three configurations use the current sequential fitter with historical
settings where the current API exposes an equivalent control; they do not
reproduce every historical implementation detail.

Example::

    python tests/test_hpi_versions.py \
        --data subject_raw.fif resting_raw.fif \
        --hpi HPIBefore_raw.fif --pol digitisation.json \
        --reffile resting_reference_raw.fif \
        --freq 33 --sfreq 1000 --output-dir version_comparison

The reference file is optional. v0.1.0-like settings intentionally disable
noise detection, so the reference is used only by the v0.2.0/v0.3.0-like runs.
Each successful run writes transformed FIFs, JSON fit sidecars, one alignment
plot, and a GOF/residual comparison plot.
"""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import numpy as np

# Permit direct execution from the repository checkout without requiring an
# editable install or a manually configured PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opm_utility_scripts.hpi import apply_transform, fit_hpi
from opm_utility_scripts.hpi._core import write_settings_json
from opm_utility_scripts.viz import plot_hpi_alignment


VERSION_SETTINGS = {
    'v0.1.0': {
        'bad_channel_policy': 'none',
        'activation_window_s': 2.0,
        'gof_limit': 0.9,
        'gof_comparison': 'strict',
        'matching_strategy': 'coordinate_nearest',
        'unique_matches': False,
        'optim': 'none',
    },
    'v0.2.0': {
        'bad_channel_policy': 'auto',
        'activation_window_s': 2.0,
        'gof_limit': 0.95,
        'gof_comparison': 'inclusive',
        'matching_strategy': 'centroid_nearest',
        'unique_matches': False,
        'optim': 'rigid_gof',
    },
    'v0.3.0': {
        'bad_channel_policy': 'auto',
        'activation_window_s': 2.0,
        'gof_limit': 0.95,
        'gof_comparison': 'inclusive',
        'matching_strategy': 'centroid_nearest',
        'unique_matches': True,
        'optim': 'rigid_gof',
    },
}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', required=True, metavar='FILE',
                        help='Data FIF file(s) to transform in each run.')
    parser.add_argument('--hpi', required=True, metavar='FILE',
                        help='HPI recording shared by all runs.')
    parser.add_argument('--pol', required=True, metavar='FILE',
                        help='Polhemus digitisation file shared by all runs.')
    parser.add_argument('--reffile', default=None, metavar='FILE',
                        help='Optional reference recording shared by runs using automatic noise detection.')
    parser.add_argument('--freq', type=float, default=33.0, metavar='HZ',
                        help='HPI drive frequency (default: 33).')
    parser.add_argument('--sfreq', type=float, default=None, metavar='HZ',
                        help='Optional output sampling frequency (default: retain source rate).')
    parser.add_argument('--output-dir', type=Path, default=Path('hpi_version_comparison'),
                        metavar='DIR', help='Directory for version-specific outputs.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace existing comparison outputs (default: refuse).')
    return parser.parse_args(argv)


def _output_name(datafile, version, sfreq):
    datafile = Path(datafile)
    stem = datafile.stem
    if stem.endswith('_raw'):
        stem = stem[:-4]
    suffix = '_proc-hpi'
    if sfreq is not None and int(sfreq) != int(mne.io.read_info(datafile, verbose='error')['sfreq']):
        suffix += '+ds'
    return f'{stem}_{version}{suffix}_raw.fif'


def _save_comparison_plot(results, output_path):
    versions = list(results)
    coil_names = list(dict.fromkeys(
        name for version in versions for name in results[version]['fit']['hpi_names']))
    x = np.arange(len(coil_names))
    width = 0.8 / len(versions)
    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(coil_names) * 1.5), 8),
                             sharex=True, constrained_layout=True)

    for version_i, version in enumerate(versions):
        fit = results[version]['fit']
        coil_indices = {name: i for i, name in enumerate(fit['hpi_names'])}
        gofs = np.full(len(coil_names), np.nan)
        residuals = np.full(len(coil_names), np.nan)
        for x_i, name in enumerate(coil_names):
            coil_i = coil_indices.get(name)
            if coil_i is not None and coil_i < len(fit['hpi_gofs']):
                gofs[x_i] = fit['hpi_gofs'][coil_i]
        for matched_i, coil_i in enumerate(np.flatnonzero(fit['include_hpis'])):
            if matched_i < len(fit['dist']) and coil_i < len(fit['hpi_names']):
                residuals[coil_names.index(fit['hpi_names'][coil_i])] = fit['dist'][matched_i] * 1000.0
        gof_offset = (version_i - (len(versions) - 1) / 2) * width
        axes[0].bar(x + gof_offset, gofs, width, label=version)
        axes[1].bar(x + gof_offset, residuals, width, label=version)

    axes[0].set_ylabel('Dipole GOF')
    axes[0].set_ylim(bottom=0)
    axes[0].set_title('HPI fit comparison across historical settings mappings')
    axes[1].set_ylabel('Post-fit residual (mm)')
    axes[1].set_xticks(x, coil_names, rotation=30, ha='right')
    axes[1].set_xlabel('HPI coil')
    for ax in axes:
        ax.grid(axis='y', alpha=0.25)
        ax.legend()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def _check_output_collisions(args):
    """Refuse to replace any planned output unless explicitly requested."""
    planned = [args.output_dir / 'version_fit_comparison.png']
    for version in VERSION_SETTINGS:
        version_dir = args.output_dir / version
        planned.append(version_dir / f'{version}_alignment.png')
        for datafile in args.data:
            output_path = version_dir / _output_name(datafile, version, args.sfreq)
            planned.extend((output_path, version_dir / f'hpi_{output_path.stem}.json'))

    collisions = [path for path in planned if path.exists()]
    if collisions and not args.overwrite:
        paths = '\n'.join(f'  {path}' for path in collisions)
        raise FileExistsError(
            f'Comparison outputs already exist; pass --overwrite to replace them:\n{paths}')


def main(argv=None):
    args = _parse_args(argv)
    try:
        _check_output_collisions(args)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_hpi_for_plot = mne.io.read_raw_fif(args.hpi, preload=False, verbose='error')
    results = {}
    errors = []

    for version, settings in VERSION_SETTINGS.items():
        version_dir = args.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        print(f'\n=== {version} settings ===')
        print(settings)
        try:
            fit = fit_hpi(args.hpi, args.pol, args.freq, reffile=args.reffile,
                          **settings)
            results[version] = {'fit': fit, 'outputs': []}
            fig = plot_hpi_alignment(fit, raw=raw_hpi_for_plot, show=False)
            plot_path = version_dir / f'{version}_alignment.png'
            fig.savefig(plot_path, dpi=180, bbox_inches='tight')
            plt.close(fig)
            print(f'Saved plot:  {plot_path}')
            for datafile in args.data:
                raw_out = apply_transform(datafile, fit, new_sfreq=args.sfreq)
                output_path = version_dir / _output_name(datafile, version, args.sfreq)
                raw_out.save(output_path, overwrite=args.overwrite, verbose='error')
                sidecar_path = version_dir / f'hpi_{output_path.stem}.json'
                write_settings_json(sidecar_path, fit, hpifile=args.hpi,
                                    polfile=args.pol, reffile=args.reffile,
                                    datafile=datafile, output_file=output_path)
                results[version]['outputs'].append(output_path)
                print(f'Saved data:  {output_path}')
                print(f'Saved fit:   {sidecar_path}')
        except Exception as exc:
            errors.append((version, exc))
            print(f'FAILED {version}: {type(exc).__name__}: {exc}', file=sys.stderr)

    if results:
        comparison_path = args.output_dir / 'version_fit_comparison.png'
        _save_comparison_plot(results, comparison_path)
        print(f'\nSaved comparison plot: {comparison_path}')
    else:
        plt.close('all')

    if errors:
        print('\nOne or more settings runs failed:', file=sys.stderr)
        for version, exc in errors:
            print(f'  {version}: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
