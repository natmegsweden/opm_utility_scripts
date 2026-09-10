#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check an HPI recording.

Usage::

    python -m opm_utility_scripts.hpi.check [--hpi HPIbefore_raw.fif] [--pol digitisation.json] [--freq 33]

- Without ``--pol``: HPI-only mode — calls ``fit_hpi_amplitudes()`` from
  ``_core.py``, shows GOF and dipole positions in device space.
- With ``--pol``: Full coregistration mode — calls ``fit_hpi()`` from
  ``_core.py`` (same pipeline as ``coregister``), shows polhemus targets,
  residuals, inter-coil distances, and transform summary in head space.

Both modes use the same calculation engine in ``_core.py``.
"""

import argparse
import sys
import warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Thresholds from MNE-Python defaults (mne/chpi.py: compute_head_pos,
# _get_hpi_initial_fit) and MEGIN/Elekta MaxFilter convention, consistent
# with Zetter et al. 2019 (doi:10.1038/s41598-019-41763-4) and Tierney et
# al. 2021 (doi:10.1016/j.neuroimage.2021.118091).
_GOF_ACCEPT     = 0.98  # per-coil fit GOF — MNE default gof_limit / good_limit
_POL_GOF_ACCEPT = 0.90  # polhemus-position GOF threshold — lower than _GOF_ACCEPT
                        # because _gof_at_fixed_pos uses raw slopes without the
                        # SSS-like external interference projection, so values
                        # naturally run ~0.05 below the floating-dipole GOF
_DIST_ACCEPT    = 5.0   # per-coil residual (mm) — MNE default dist_limit 0.005 m
_MIN_COILS      = 3     # minimum coils passing both criteria


def _gof_color(g):
    if g >= _GOF_ACCEPT:
        return 'green'
    elif g >= 0.8:
        return 'darkorange'
    return 'red'


def _is_nasion_coil(ch_name):
    """Return True if a channel name identifies it as a nasion landmark coil.

    The FieldLine system lets operators name HPI coil slots freely.  A coil
    placed at the nasion is sometimes labelled 'Nasion' (e.g. 'hpiout_Nasion').
    These coils sit on a bony landmark rather than on scalp, which affects
    the dipole fit geometry and polhemus GOF; they should not trigger a REDO
    recommendation on their own.
    """
    return 'nasion' in ch_name.lower()


def _recommendation(hpi_gofs, dists_mm=None, include_hpis=None,
                    pol_gofs=None, hpi_names=None):
    """Evaluate HPI fit quality and return separate verdicts for each failure mode.

    There are two independent causes of a poor HPI result:

    * **HPI recording quality** — measured by ``hpi_gofs`` (goodness-of-fit of
      the single-dipole model to MEG sensor data).  A poor GOF means the coil
      signal was not cleanly captured; the remedy is to **redo the HPI
      recording** (subject may have moved, coil placement or drive issue).

    * **Polhemus registration quality** — measured by ``dists_mm`` (distance
      between the fitted coil position in device space, transformed to head
      space, and the digitised coil position).  A large residual means the
      polhemus digitisation does not match the fitted positions; the remedy is
      to **redo the polhemus digitisation**.

    Thresholds follow MNE-Python ``compute_head_pos`` defaults
    (``gof_limit=0.98``, ``dist_limit=0.005`` m), which originate from the
    MEGIN/Elekta MaxFilter convention and are consistent with Zetter et al.
    2019 (doi:10.1038/s41598-019-41763-4) and Tierney et al. 2021
    (doi:10.1016/j.neuroimage.2021.118091).

    Parameters
    ----------
    hpi_gofs : array-like
        Per-coil GOF from ``compute_chpi_locs`` (0–1).
    dists_mm : array-like or None
        Per-coil residuals in mm for coils in ``include_hpis``.
        ``None`` in HPI-only mode.
    include_hpis : array-like of bool or None
        Mask of coils whose GOF ≥ ``_GOF_ACCEPT``. ``None`` in HPI-only mode.
    pol_gofs : array-like or None
        GOF of the polhemus-digitised position against MEG sensor data for
        each included coil (from ``_gof_at_fixed_pos`` in ``_core.py``).
        ``None`` in HPI-only mode.
    hpi_names : list[str] or None
        Channel names for all coils (same length as ``hpi_gofs``).
        When provided, coils identified as nasion-landmark coils (via
        ``_is_nasion_coil``) are excluded from the verdict logic — they
        are reported in the GOF table but do not trigger REDO recommendations.

    Returns
    -------
    hpi_verdict  : str   — 'OK', 'REDO HPI', or 'POOR'
    hpi_color    : str
    hpi_reasons  : list[str]
    pol_verdict  : str   — 'OK', 'REDO POLHEMUS', or 'POOR' (or None in HPI-only)
    pol_color    : str   (or None)
    pol_reasons  : list[str] (or None)
    """
    hpi_gofs = np.asarray(hpi_gofs)
    n_coils  = len(hpi_gofs)

    # Build nasion mask — coils on bony landmarks are excluded from verdicts.
    if hpi_names is not None and len(hpi_names) == n_coils:
        nasion_mask = np.array([_is_nasion_coil(n) for n in hpi_names])
    else:
        nasion_mask = np.zeros(n_coils, dtype=bool)

    # ------------------------------------------------------------------ #
    # Part 1 — HPI recording quality (GOF from MEG sensor data)          #
    # Nasion-labelled coils are noted but excluded from the verdict.      #
    # ------------------------------------------------------------------ #
    hpi_reasons = []

    # Evaluate only non-nasion coils for the verdict.
    scoreable    = ~nasion_mask
    poor_gof_all = hpi_gofs < _GOF_ACCEPT
    poor_gof     = poor_gof_all & scoreable

    if include_hpis is not None:
        # n_good = coils that passed GOF and are not nasion
        n_good = int(np.sum(include_hpis & scoreable))
    else:
        n_good = int(np.sum((hpi_gofs >= _GOF_ACCEPT) & scoreable))

    if poor_gof.any():
        hpi_reasons.append(
            f'{poor_gof.sum()} coil(s) have GOF < {_GOF_ACCEPT} '
            f'— dipole model does not fit the MEG data well'
        )
    if nasion_mask.any() and poor_gof_all[nasion_mask].any():
        hpi_reasons.append(
            f'Nasion coil(s) also have low GOF '
            f'(noted but not used for verdict — landmark placement expected)'
        )
    if n_good < _MIN_COILS:
        hpi_reasons.append(
            f'Only {n_good} non-nasion coil(s) pass GOF threshold '
            f'(need ≥ {_MIN_COILS} for a valid transform)'
        )

    if not hpi_reasons:
        hpi_verdict, hpi_color = 'OK', 'green'
        hpi_reasons = [f'All coils GOF ≥ {_GOF_ACCEPT} — HPI recording is good']
    elif n_good < _MIN_COILS:
        hpi_verdict, hpi_color = 'POOR', 'red'
        hpi_reasons.append('→ Redo HPI recording (check coil drive and subject movement)')
    else:
        hpi_verdict, hpi_color = 'REDO HPI', 'darkorange'
        hpi_reasons.append('→ Consider redoing HPI recording (subject movement or coil issue)')

    # ------------------------------------------------------------------ #
    # Part 2 — Polhemus registration quality                             #
    # Two independent signals:                                           #
    #   pol_gofs  — dipole GOF at the digitised position against MEG    #
    #               data.  Low = digitised position is wrong.           #
    #   dists_mm  — geometric distance between fitted and digitised pos. #
    #               Large = positions don't agree.                      #
    # ------------------------------------------------------------------ #
    if dists_mm is None or len(dists_mm) == 0:
        return hpi_verdict, hpi_color, hpi_reasons, None, None, None

    dists_mm    = np.asarray(dists_mm)
    pol_reasons = []

    # Nasion mask for the *included* coils (subset of all coils).
    if include_hpis is not None and hpi_names is not None:
        incl_names    = [hpi_names[i] for i in np.where(include_hpis)[0]]
        nasion_incl   = np.array([_is_nasion_coil(n) for n in incl_names])
        scoreable_pol = ~nasion_incl
    else:
        scoreable_pol = np.ones(len(dists_mm), dtype=bool)

    large_all = dists_mm >= _DIST_ACCEPT
    large     = large_all & scoreable_pol
    # Mean over non-nasion included coils only.
    scored_dists = dists_mm[scoreable_pol]
    mean_res     = float(np.mean(scored_dists)) if len(scored_dists) else float('nan')

    if pol_gofs is not None and len(pol_gofs):
        pol_gofs  = np.asarray(pol_gofs)
        finite    = np.isfinite(pol_gofs)
        poor_pgof = finite & scoreable_pol & (pol_gofs < _POL_GOF_ACCEPT)
        if poor_pgof.any():
            pol_reasons.append(
                f'{poor_pgof.sum()} coil(s) have polhemus-position GOF < {_POL_GOF_ACCEPT} '
                f'— digitised position does not match the MEG field pattern'
            )

    if large.any():
        pol_reasons.append(
            f'{large.sum()} coil(s) have residual ≥ {_DIST_ACCEPT:.0f} mm '
            f'— fitted and digitised positions disagree'
        )
    if mean_res >= _DIST_ACCEPT:
        pol_reasons.append(
            f'Mean residual {mean_res:.1f} mm ≥ {_DIST_ACCEPT:.0f} mm'
        )

    if not pol_reasons:
        pol_verdict, pol_color = 'OK', 'green'
        pol_reasons = [
            f'All polhemus-position GOFs ≥ {_POL_GOF_ACCEPT} and '
            f'residuals < {_DIST_ACCEPT:.0f} mm — polhemus registration is good'
        ]
    elif mean_res >= _DIST_ACCEPT * 2 or (
        pol_gofs is not None
        and np.any((pol_gofs < 0.5) & scoreable_pol & np.isfinite(pol_gofs))
    ):
        pol_verdict, pol_color = 'POOR', 'red'
        pol_reasons.append('→ Redo polhemus digitisation (fiducial placement or stylus error)')
    else:
        pol_verdict, pol_color = 'REDO POLHEMUS', 'darkorange'
        pol_reasons.append('→ Consider redoing polhemus digitisation (coil position mismatch)')

    return hpi_verdict, hpi_color, hpi_reasons, pol_verdict, pol_color, pol_reasons


def _short_name(ch_name):
    """Strip the 'hpiout' / 'hpiin' prefix, leaving just the coil identifier.

    Examples: 'hpiout2' -> '2', 'hpiout_Nasion' -> 'Nasion', 'hpiin4' -> '4'.
    Falls back to the full name if no recognised prefix is present.
    """
    for prefix in ('hpiout', 'hpiin'):
        if ch_name.startswith(prefix):
            suffix = ch_name[len(prefix):]
            return suffix.lstrip('_') or ch_name
    return ch_name


def _sep(title=''):
    SEP = '─' * 72
    if title:
        pad = max(0, 72 - len(title) - 2)
        print(f'\n{"─" * (pad // 2)} {title} {"─" * (pad - pad // 2)}\n')
    else:
        print(f'\n{SEP}\n')


def _render_text(ax, lines):
    """Render a list of (text, color) into an axis('off') panel."""
    y = 0.97
    step = min(0.055, 0.97 / max(len(lines), 1))
    for text, color in lines:
        ax.text(0.02, y, text, transform=ax.transAxes,
                fontsize=8, color=color, fontfamily='monospace',
                va='top', ha='left')
        y -= step


def _resolve_hpi_only(amp):
    """Call compute_chpi_locs on the amplitude result with a dummy dig.

    ``fit_hpi_amplitudes`` stops before ``compute_chpi_locs`` because that
    function requires properly set isotrak dig points to determine the number
    of polhemus coils.  In HPI-only mode we have no polhemus, so we inject a
    dummy dig sized to match the number of HPI output channels, which prevents
    the shape mismatch in ``_get_hpi_initial_fit``.

    Returns the amplitude dict augmented with ``hpi_dev`` and ``hpi_gofs``.
    """
    raw_orig        = amp['raw_orig']
    coil_amplitudes = amp['coil_amplitudes']
    hpi_indices     = amp['hpi_indices']
    n_hpi           = len(hpi_indices)

    # Inject a dummy dig with the correct number of coils so
    # compute_chpi_locs does not hit a shape mismatch.
    # _get_hpi_initial_fit requires coord_frame == FIFFV_COORD_HEAD (4) for HPI
    # dig points — FIFFV_COORD_DEVICE raises "cHPI coordinate frame incorrect".
    with raw_orig.info._unlock():
        raw_orig.info['dig'] = [
            dict(r=np.zeros(3),
                 coord_frame=FIFF.FIFFV_COORD_HEAD,
                 ident=ii + 1,
                 kind=FIFF.FIFFV_POINT_HPI)
            for ii in range(n_hpi)
        ]

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        coil_locs = compute_chpi_locs(raw_orig.info, coil_amplitudes)

    return {
        **amp,
        'hpi_dev':  np.array(coil_locs['rrs'][0]),
        'hpi_gofs': np.array(coil_locs['gofs'][0]),
    }


def _load_heavy_deps():
    """Import scientific stack lazily so --help returns without loading MNE."""
    import matplotlib.pyplot as plt
    import mne
    import numpy as np
    from mne.chpi import compute_chpi_locs
    from mne.io.constants import FIFF
    from mne.transforms import apply_trans, Transform
    from ._core import fit_hpi_amplitudes, fit_hpi, compute_fit_diagnostics
    from ..viz import plot_hpi_raw_channels

    g = globals()
    g.update(dict(
        plt=plt, mne=mne, np=np,
        compute_chpi_locs=compute_chpi_locs,
        FIFF=FIFF, apply_trans=apply_trans, Transform=Transform,
        fit_hpi_amplitudes=fit_hpi_amplitudes, fit_hpi=fit_hpi,
        compute_fit_diagnostics=compute_fit_diagnostics,
        plot_hpi_raw_channels=plot_hpi_raw_channels,
    ))


def _parse_args():
    parser = argparse.ArgumentParser(
        prog='python -m opm_utility_scripts.hpi.check',
        description=(
            'Check HPI recording quality and/or polhemus digitisation.\n\n'
            '  --hpi only       : HPI-only mode — GOF, sensor count, '
            'device-frame inter-coil distances.\n'
            '  --pol only       : Polhemus-only mode — dig point summary, '
            'head-frame inter-coil distances.\n'
            '  --hpi and --pol  : Full coregistration — brief by default '
            '(GOF + residuals + recommendations).\n'
            '                     Add --detailed for all diagnostics.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--hpi', '-f', metavar='PATH',
                        help='Path to the raw HPI .fif file.')
    parser.add_argument('--pol', '-p', metavar='PATH',
                        help='Path to Polhemus file (.fif or .json). '
                             'Without --hpi: polhemus-only mode.')
    parser.add_argument('--freq', type=float, default=33.0, metavar='HZ',
                        help='HPI drive frequency in Hz (default: 33).')
    parser.add_argument('--gof', type=float, default=None, metavar='THRESH',
                        help=(
                            'Minimum dipole GOF to include a coil in the transform '
                            '(default: auto — 0.98 for distinct-frequency systems, '
                            '0.90 for single-frequency OPM systems).'
                        ))
    parser.add_argument('--detailed', action='store_true',
                        help='Show full diagnostics in --hpi + --pol mode '
                             '(sensor count, inter-coil distance table, pol-GOFs).')
    parser.add_argument('--optimization', choices=['none', 'rigid'],
                        default='none',metavar='METHOD',
                        help=(
                            'Optimization method applied after the initial HPI→Polhemus '
                            'coregistration. '
                            '"none": no refinement (default). '
                            '"rigid": refine with rigid transform of polhemus '
                            'locations by minimizing the summed dipole RV.'
                        ))
    return parser.parse_args()


# ---------------------------------------------------------------------------
# HPI-only mode  (fit_hpi_amplitudes)
# ---------------------------------------------------------------------------

def _build_figure_hpi_only(amp):
    """2-panel figure: device-space sensor scatter + dipole positions."""
    hpi_dev   = np.array(amp['hpi_dev'])
    hpi_gofs  = np.array(amp['hpi_gofs'])
    hpi_names = amp['hpi_names']
    raw       = amp['raw_orig']
    n_hpi     = len(hpi_names)

    # Sensor positions in device space for background scatter.
    # Deduplicate by slot: keep one channel per unique location (prefer _bz).
    meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])
    bz_mask = np.array([
        '_bz' in raw.info['chs'][i]['ch_name'] for i in meg_picks
    ])
    if bz_mask.sum() == 0:
        bz_mask = np.ones(len(meg_picks), dtype=bool)
    bz_pos_mm = np.array([raw.info['chs'][meg_picks[i]]['loc'][:3]
                           for i in range(len(meg_picks)) if bz_mask[i]]) * 1000

    fig = plt.figure(figsize=(14, 7), constrained_layout=True, facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax_3d = fig.add_subplot(gs[0], projection='3d')
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')

    ax_3d.scatter(bz_pos_mm[:, 0], bz_pos_mm[:, 1], bz_pos_mm[:, 2],
                  c='#e07b39', alpha=0.6, marker=(4, 0, 45), s=18)

    rrs_mm = hpi_dev * 1000
    for i in range(n_hpi):
        short = _short_name(hpi_names[i])
        gof = hpi_gofs[i]
        color = _gof_color(gof)
        pos = rrs_mm[i]
        if not np.allclose(pos, 0):
            ax_3d.scatter(*pos, c=color, s=80, marker='o', zorder=7, depthshade=False)
            ax_3d.text(pos[0], pos[1], pos[2],
                       f' {short}\nGOF={gof:.2f}', fontsize=8, color=color, zorder=8)
        else:
            ax_3d.text(0, 0, 0, f'{short}\n[no fit]', fontsize=8, color='gray', zorder=8)

    ax_3d.view_init(elev=70, azim=-60)
    ax_3d.set_xlabel('x (mm)', fontsize=8)
    ax_3d.set_ylabel('y (mm)', fontsize=8)
    ax_3d.set_zlabel('z (mm)', fontsize=8)
    ax_3d.grid(False)
    ax_3d.set_facecolor('white')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('white')
    ax_3d.yaxis.pane.set_edgecolor('white')
    ax_3d.zaxis.pane.set_edgecolor('white')
    ax_3d.set_title('HPI dipole positions vs sensors — device space', fontsize=9)

    lines = []
    def add(t, c='black'): lines.append((t, c))
    add('HPI Check (HPI-only mode)', 'black')
    add('─' * 40, '#888888')
    add('Per-coil GOF:', 'black')
    for i in range(n_hpi):
        gof = hpi_gofs[i]
        add(f'  {hpi_names[i]}: {gof:.3f}', _gof_color(gof))

    hpi_v, hpi_c, hpi_r, *_ = _recommendation(hpi_gofs, hpi_names=hpi_names)
    add('─' * 40, '#888888')
    add(f'HPI recording: {hpi_v}', hpi_c)
    for r in hpi_r:
        add(f'  {r}', hpi_c)

    _render_text(ax_text, lines)
    return fig


def _print_diagnostics_hpi_only(amp):
    hpi_gofs  = np.array(amp['hpi_gofs'])
    hpi_names = amp['hpi_names']
    hpi_dev   = np.array(amp['hpi_dev'])
    raw       = amp['raw_orig']

    n_sensors = len(mne.pick_types(raw.info, meg=True, exclude=[]))
    n_bads    = len(raw.info.get('bads', []))

    _sep('HPI Check — HPI-only mode')
    print(f'  MEG sensors used in fit: {n_sensors}'
          + (f'  (excluded bads: {n_bads})' if n_bads else ''))
    print(f'  Active HPI coils: {len(hpi_names)}')
    print()
    print(f'  {"Coil":<28} {"GOF":>6}  {"x (mm)":>8} {"y (mm)":>8} {"z (mm)":>8}')
    print('  ' + '─' * 62)
    for name, gof, pos in zip(hpi_names, hpi_gofs, hpi_dev):
        flag = 'OK' if gof >= _GOF_ACCEPT else ('MARGINAL' if gof >= 0.8 else 'POOR')
        p = pos * 1000
        print(f'  {name:<28} {gof:6.3f}  {p[0]:8.1f} {p[1]:8.1f} {p[2]:8.1f}  [{flag}]')

    if len(hpi_dev) >= 2:
        _sep('Inter-coil distances — device frame (mm)')
        shorts = [_short_name(n) for n in hpi_names]
        n = len(hpi_names)
        print(f'  {"Pair":<28}  {"dist (mm)":>10}')
        print('  ' + '─' * 40)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(hpi_dev[i] - hpi_dev[j]) * 1000
                print(f'  {shorts[i]}-{shorts[j]:<24}  {d:10.1f}')

    hpi_v, _, hpi_r, *_ = _recommendation(hpi_gofs, hpi_names=hpi_names)
    _sep(f'HPI recording: {hpi_v}')
    for r in hpi_r:
        print(f'  {r}')


# ---------------------------------------------------------------------------
# Polhemus-only mode  (--pol without --hpi)
# ---------------------------------------------------------------------------

def _print_diagnostics_pol_only(pol):
    """Print a summary of a polhemus digitisation file."""
    from ..io import load_polhemus

    _sep('Polhemus digitisation summary')

    hpi_orig  = np.array(pol['hpi_orig'])   # head frame, metres
    n_hpi     = len(hpi_orig)
    extra_pts = pol.get('extra_pts')
    n_extra   = len(extra_pts) if extra_pts is not None else 0

    fids = {}
    for key, label in [('nasion', 'Nasion'), ('lpa', 'LPA'), ('rpa', 'RPA')]:
        v = pol.get(key)
        if v is not None:
            fids[label] = np.asarray(v) * 1000  # mm

    print(f'  Source:           {pol.get("path", pol.get("source", "?"))}')
    print(f'  Frame:            {pol.get("source", "?")}')
    print(f'  HPI dig points:   {n_hpi}')
    print(f'  Headshape points: {n_extra}')
    print()

    if fids:
        print(f'  Fiducials (mm, head frame):')
        for label, pos in fids.items():
            print(f'    {label:<8}  x={pos[0]:7.1f}  y={pos[1]:7.1f}  z={pos[2]:7.1f}')
        print()

    if n_hpi > 0:
        print(f'  HPI coil positions (mm, head frame):')
        print(f'    {"#":<4}  {"x (mm)":>8} {"y (mm)":>8} {"z (mm)":>8}')
        print('    ' + '─' * 32)
        for i, pos in enumerate(hpi_orig * 1000):
            print(f'    {i+1:<4}  {pos[0]:8.1f} {pos[1]:8.1f} {pos[2]:8.1f}')
        print()

    if n_hpi >= 2:
        _sep('Inter-coil distances — polhemus head frame (mm)')
        names = [f'HPI#{i+1}' for i in range(n_hpi)]
        shorts = names
        print(f'  {"Pair":<20}  {"dist (mm)":>10}')
        print('  ' + '─' * 32)
        for i in range(n_hpi):
            for j in range(i + 1, n_hpi):
                d = np.linalg.norm(hpi_orig[i] - hpi_orig[j]) * 1000
                print(f'  {shorts[i]}-{shorts[j]:<16}  {d:10.1f}')


def _build_figure_pol_only(pol):
    """3-D scatter of polhemus dig points in head frame."""
    hpi_orig  = np.array(pol['hpi_orig']) * 1000   # mm
    extra_pts = pol.get('extra_pts')
    n_hpi     = len(hpi_orig)

    fig = plt.figure(figsize=(10, 7), constrained_layout=True, facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')

    if extra_pts is not None and len(extra_pts):
        ep = np.asarray(extra_pts) * 1000
        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2],
                   c='#666666', s=4, alpha=0.4, label='Headshape')

    fid_colors = {'LPA': 'darkorange', 'Nasion': 'limegreen', 'RPA': 'darkorange'}
    for label, key in [('LPA', 'lpa'), ('Nasion', 'nasion'), ('RPA', 'rpa')]:
        v = pol.get(key)
        if v is not None:
            fp = np.asarray(v) * 1000
            ax.scatter(*fp, c=fid_colors[label], s=80, marker='^',
                       zorder=5, depthshade=False)
            ax.text(fp[0], fp[1], fp[2], f' {label}',
                    fontsize=8, color=fid_colors[label])

    for i, pos in enumerate(hpi_orig):
        ax.scatter(*pos, c='royalblue', s=120, marker='*',
                   zorder=7, depthshade=False)
        ax.text(pos[0], pos[1], pos[2], f' HPI#{i+1}',
                fontsize=9, color='royalblue', zorder=8)

    # Draw inter-coil lines
    for i in range(n_hpi):
        for j in range(i + 1, n_hpi):
            d = np.linalg.norm(hpi_orig[i] - hpi_orig[j])
            mid = (hpi_orig[i] + hpi_orig[j]) / 2
            ax.plot([hpi_orig[i, 0], hpi_orig[j, 0]],
                    [hpi_orig[i, 1], hpi_orig[j, 1]],
                    [hpi_orig[i, 2], hpi_orig[j, 2]],
                    color='royalblue', lw=0.8, linestyle='--', alpha=0.5)
            ax.text(mid[0], mid[1], mid[2], f'{d:.0f} mm',
                    fontsize=7, color='royalblue', ha='center')

    ax.view_init(elev=70, azim=-60)
    ax.set_xlabel('x (mm)', fontsize=8)
    ax.set_ylabel('y (mm)', fontsize=8)
    ax.set_zlabel('z (mm)', fontsize=8)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    path = pol.get('path', '')
    ax.set_title(f'Polhemus digitisation — head frame\n{path}', fontsize=9)
    return fig


# ---------------------------------------------------------------------------
# Full coregistration mode  (fit_hpi)
# ---------------------------------------------------------------------------

def _build_figure_full(fit, detailed=False, diag=None):
    """2-panel figure: head-space alignment matching coregister's plot_hpi_alignment."""
    hpi_dev      = np.array(fit['hpi_dev'])
    hpi_gofs     = np.array(fit['hpi_gofs'])
    hpi_orig     = np.array(fit['hpi_orig'])      # head frame (Polhemus targets)
    hpi_names    = fit['hpi_names']
    dev_to_head  = fit['dev_to_head_trans']
    include_hpis = np.array(fit['include_hpis'])
    tree_indices = np.array(fit['tree_indices'])
    raw          = fit['raw_for_topomap']

    hpi_fitted_head_mm = apply_trans(dev_to_head, hpi_dev) * 1000
    hpi_pol_mm         = hpi_orig * 1000

    fig = plt.figure(figsize=(14, 7), constrained_layout=True, facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax_3d = fig.add_subplot(gs[0], projection='3d')
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis('off')

    # Sensor cloud in head space — deduplicate by slot (prefer _bz).
    meg_picks = mne.pick_types(raw.info, meg=True, exclude=[])
    bz_mask = np.array(['_bz' in raw.info['chs'][i]['ch_name'] for i in meg_picks])
    if bz_mask.sum() == 0:
        bz_mask = np.ones(len(meg_picks), dtype=bool)
    sensor_dev = np.array([raw.info['chs'][meg_picks[i]]['loc'][:3]
                            for i in range(len(meg_picks)) if bz_mask[i]])
    sensor_head_mm = apply_trans(dev_to_head, sensor_dev) * 1000
    ax_3d.scatter(sensor_head_mm[:, 0], sensor_head_mm[:, 1], sensor_head_mm[:, 2],
                  c='#e07b39', s=18, alpha=0.6, marker=(4, 0, 45), zorder=1)

    # Headshape
    extra_pts = fit.get('extra_pts')
    if extra_pts is not None and len(extra_pts):
        extra_mm = np.asarray(extra_pts) * 1000
        ax_3d.scatter(extra_mm[:, 0], extra_mm[:, 1], extra_mm[:, 2],
                      c='#666666', s=3, alpha=0.4, zorder=1)

    # Fiducials
    fid_colors = {'LPA': 'darkorange', 'Nasion': 'limegreen', 'RPA': 'darkorange'}
    for label, key in [('LPA', 'lpa'), ('Nasion', 'nasion'), ('RPA', 'rpa')]:
        fpos = np.array(fit[key]) * 1000
        ax_3d.scatter(*fpos, c=fid_colors[label], s=60, marker='^',
                      zorder=5, depthshade=False)
        ax_3d.text(fpos[0], fpos[1], fpos[2], f' {label}',
                   fontsize=7, color=fid_colors[label], zorder=6)

    # Polhemus targets — blue stars
    ax_3d.scatter(hpi_pol_mm[:, 0], hpi_pol_mm[:, 1], hpi_pol_mm[:, 2],
                  c='royalblue', s=120, marker='*', zorder=7,
                  depthshade=False, label='Polhemus target')

    # Fitted coil positions — colour-coded by GOF
    for i, (pos, gof) in enumerate(zip(hpi_fitted_head_mm, hpi_gofs)):
        color = _gof_color(gof)
        ax_3d.scatter(*pos, c=color, s=80, marker='o', zorder=7, depthshade=False)
        short = _short_name(hpi_names[i])
        ax_3d.text(pos[0], pos[1], pos[2],
                   f' {short}\nGOF={gof:.2f}', fontsize=7, color=color, zorder=8)

    # Connecting lines — colour-coded by residual (mirrors plot_hpi_alignment)
    for k, (dev_i, pol_i) in enumerate(zip(np.where(include_hpis)[0], tree_indices)):
        p_fit = hpi_fitted_head_mm[dev_i]
        p_pol = hpi_pol_mm[pol_i]
        dist_mm = np.linalg.norm(p_fit - p_pol)
        mid = (p_fit + p_pol) / 2
        lcolor = 'green' if dist_mm < 5 else ('darkorange' if dist_mm < 10 else 'red')
        ax_3d.plot([p_fit[0], p_pol[0]], [p_fit[1], p_pol[1]], [p_fit[2], p_pol[2]],
                   color=lcolor, lw=1.2, linestyle='--', zorder=6)
        ax_3d.text(mid[0], mid[1], mid[2], f'{dist_mm:.1f} mm',
                   fontsize=7, color=lcolor, zorder=9, ha='center', va='bottom')

    ax_3d.view_init(elev=70, azim=-60)
    ax_3d.set_xlabel('x (mm)', fontsize=8)
    ax_3d.set_ylabel('y (mm)', fontsize=8)
    ax_3d.set_zlabel('z (mm)', fontsize=8)
    ax_3d.grid(False)
    ax_3d.set_facecolor('white')
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    ax_3d.xaxis.pane.set_edgecolor('white')
    ax_3d.yaxis.pane.set_edgecolor('white')
    ax_3d.zaxis.pane.set_edgecolor('white')
    ax_3d.set_title('HPI fitted (●) vs Polhemus target (★) — head space', fontsize=9)

    _fill_text_panel_full(ax_text, fit, detailed=detailed, diag=diag)
    return fig


def _fill_text_panel_full(ax, fit, detailed=False, diag=None):
    """Render the text panel for the full coregistration figure.

    Parameters
    ----------
    diag : dict or None
        Pre-computed diagnostics from ``compute_fit_diagnostics(fit)``.
        Pass this to avoid recomputing when already available; if ``None``
        it is computed here.
    """
    hpi_gofs     = np.array(fit['hpi_gofs'])
    hpi_names    = fit['hpi_names']
    include_hpis = np.array(fit['include_hpis'])
    dists_mm     = np.array(fit['dist']) * 1000

    if diag is None:
        diag = compute_fit_diagnostics(fit)

    lines = []
    def add(t, c='black'): lines.append((t, c))

    add('HPI Check (full coregistration)', 'black')
    add('─' * 40, '#888888')

    # Sensor count (detailed only)
    if detailed:
        raw = fit.get('raw_for_topomap')
        if raw is not None:
            n_sensors = len(mne.pick_types(raw.info, meg=True, exclude=[]))
            add(f'MEG sensors in fit: {n_sensors}', '#444444')

    add('Per-coil GOF:', 'black')
    for name, gof in zip(hpi_names, hpi_gofs):
        add(f'  {name}: {gof:.3f}', _gof_color(gof))

    pol_gofs_fig = np.asarray(fit.get('pol_gofs', []))
    has_pgof_fig = len(pol_gofs_fig) == len(np.where(include_hpis)[0])

    add('─' * 40, '#888888')
    add('Residuals (fitted vs Polhemus):', 'black')
    for k, dev_i in enumerate(np.where(include_hpis)[0]):
        short = _short_name(hpi_names[dev_i])
        d = dists_mm[k]
        color = 'green' if d < _DIST_ACCEPT else ('darkorange' if d < 10 else 'red')
        pgof_str = f'  pgof={pol_gofs_fig[k]:.3f}' if (has_pgof_fig and detailed) else ''
        add(f'  {short}: {d:.2f} mm{pgof_str}', color)
    for dev_i in np.where(~include_hpis)[0]:
        short = _short_name(hpi_names[dev_i])
        add(f'  {short}: excl. (GOF<{_GOF_ACCEPT})', 'gray')

    add('─' * 40, '#888888')
    add(f'Mean residual: {diag["mean_res_mm"]:.2f} mm',
        'green' if diag['mean_res_mm'] < _DIST_ACCEPT
        else ('darkorange' if diag['mean_res_mm'] < 10 else 'red'))

    add('─' * 40, '#888888')
    add('Transform summary:', 'black')
    add(f'  Rotation:    {diag["rot_deg"]:.1f}°', 'black')
    add(f'  Translation: {diag["trans_mm"]:.1f} mm', 'black')

    # Inter-coil distance consistency (detailed only) — from compute_fit_diagnostics
    if detailed and diag['intercoil_rows']:
        add('─' * 40, '#888888')
        add('Inter-coil dist dev/pol (mm):', 'black')
        add(f'  {"pair":<14} {"dev":>7} {"pol":>7} {"diff":>7}', '#555555')
        for name_i, name_j, da, db, diff in diag['intercoil_rows']:
            col = 'black' if diff < 5 else ('darkorange' if diff < 15 else 'red')
            si, sj = _short_name(name_i), _short_name(name_j)
            add(f'  {si}-{sj:<10} {da:7.1f} {db:7.1f} {diff:7.1f}', col)

    pol_gofs     = np.asarray(fit.get('pol_gofs', []))
    has_pol_gofs = len(pol_gofs) == len(np.where(include_hpis)[0])

    hpi_v, hpi_c, hpi_r, pol_v, pol_c, pol_r = _recommendation(
        hpi_gofs, dists_mm, include_hpis,
        pol_gofs if has_pol_gofs else None,
        hpi_names=hpi_names,
    )
    add('─' * 40, '#888888')
    add(f'HPI recording: {hpi_v}', hpi_c)
    for r in hpi_r:
        add(f'  {r}', hpi_c)
    add(f'Polhemus: {pol_v}', pol_c)
    for r in pol_r:
        add(f'  {r}', pol_c)

    _render_text(ax, lines)



def _print_diagnostics_full(fit, detailed=False, diag=None):
    hpi_gofs     = np.array(fit['hpi_gofs'])
    hpi_names    = fit['hpi_names']
    include_hpis = np.array(fit['include_hpis'])
    dists_mm     = np.array(fit['dist']) * 1000
    pol_gofs     = np.array(fit.get('pol_gofs', []))
    has_pol_gofs = len(pol_gofs) == len(np.where(include_hpis)[0])

    # All derived scalars and inter-coil data from the single authoritative source.
    if diag is None:
        diag = compute_fit_diagnostics(fit)

    _sep('HPI Check — full coregistration')

    if detailed:
        raw = fit.get('raw_for_topomap')
        if raw is not None:
            n_sensors = len(mne.pick_types(raw.info, meg=True, exclude=[]))
            n_bads    = len(raw.info.get('bads', []))
            print(f'  MEG sensors used in fit: {n_sensors}'
                  + (f'  (excluded bads: {n_bads})' if n_bads else ''))

    # Always: per-coil GOF
    print(f'  {"Coil":<28} {"GOF":>6}')
    print('  ' + '─' * 36)
    for name, gof in zip(hpi_names, hpi_gofs):
        flag = 'OK' if gof >= _GOF_ACCEPT else ('MARGINAL' if gof >= 0.8 else 'POOR')
        print(f'  {name:<28} {gof:6.3f}  [{flag}]')

    # Always: residuals
    _sep('Residuals (fitted vs Polhemus)')
    for k, dev_i in enumerate(np.where(include_hpis)[0]):
        d    = dists_mm[k]
        flag = 'OK' if d < _DIST_ACCEPT else ('LARGE' if d < 10 else 'VERY LARGE')
        pgof_str = (f'  pol_GOF={pol_gofs[k]:.3f}' if has_pol_gofs and detailed else '')
        print(f'  {hpi_names[dev_i]}: {d:.2f} mm  [{flag}]{pgof_str}')
    for dev_i in np.where(~include_hpis)[0]:
        g = hpi_gofs[dev_i]
        print(f'  {hpi_names[dev_i]}: excluded (GOF = {g:.3f} < {_GOF_ACCEPT})')

    print(f'\n  Mean residual: {diag["mean_res_mm"]:.2f} mm')
    print(f'  Transform: rotation {diag["rot_deg"]:.1f}°, translation {diag["trans_mm"]:.1f} mm')

    # Detailed only: inter-coil distance consistency — from compute_fit_diagnostics
    if detailed and diag['intercoil_rows']:
        _sep('Inter-coil distances: device frame vs polhemus (mm)')
        print('  (* >5 mm diff, ** >15 mm diff — large diff = fit/polhemus mismatch)')
        print(f'  {"Pair":<28}  {"device":>8}  {"polhemus":>8}  {"diff":>8}')
        print('  ' + '─' * 56)
        for name_i, name_j, da, db, diff in diag['intercoil_rows']:
            flag = '' if diff < 5 else (' *' if diff < 15 else ' **')
            si, sj = _short_name(name_i), _short_name(name_j)
            print(f'  {si}-{sj:<24}  {da:8.1f}  {db:8.1f}  {diff:8.1f}{flag}')

    # Always: recommendations
    hpi_v, _, hpi_r, pol_v, _, pol_r = _recommendation(
        hpi_gofs, dists_mm, include_hpis,
        pol_gofs if len(pol_gofs) else None,
        hpi_names=hpi_names,
    )
    _sep(f'HPI recording: {hpi_v}')
    for r in hpi_r:
        print(f'  {r}')
    _sep(f'Polhemus registration: {pol_v}')
    for r in pol_r:
        print(f'  {r}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    _load_heavy_deps()

    detailed = args.detailed

    hpi_file = args.hpi or None
    pol_file = args.pol or None

    # ------------------------------------------------------------------ #
    # Mode dispatch                                                        #
    #                                                                      #
    #   --hpi only       → HPI-only: GOF, sensor count, device distances  #
    #   --pol only       → Polhemus-only: dig summary, head distances      #
    #   --hpi + --pol    → Full coregistration (brief / --detailed)        #
    #   neither          → open GUI dialogs                                #
    # ------------------------------------------------------------------ #

    if hpi_file is None and pol_file is None:
        # GUI — ask for HPI first, then optionally polhemus
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        hpi_file = filedialog.askopenfilename(
            initialdir='/data', title='Select HPI file (cancel = polhemus-only)',
            filetypes=[('FIF files', '*.fif'), ('All files', '*')],
        ) or None
        root2 = tk.Tk(); root2.withdraw()
        pol_file = filedialog.askopenfilename(
            initialdir='/data',
            title='Select Polhemus file (cancel = HPI-only mode)',
            filetypes=[('JSON files', '*.json'), ('FIF files', '*.fif'), ('All files', '*')],
        ) or None
        if hpi_file is None and pol_file is None:
            print('No file selected. Exiting.')
            sys.exit(0)

    if hpi_file and pol_file:
        # Full coregistration
        try:
            fit = fit_hpi(hpi_file, pol_file, args.freq, gof_limit=args.gof, optim=args.optimization)
            diag = compute_fit_diagnostics(fit)
            _print_diagnostics_full(fit, detailed=detailed, diag=diag)
            _build_figure_full(fit, detailed=detailed, diag=diag)
        except ValueError as exc:
            print(
                f'\n[ERROR] HPI fitting failed: {exc}\n'
                '  → Falling back to raw channel plot.\n',
                file=sys.stderr,
            )
            plot_hpi_raw_channels(hpi_file, hpifreq=args.freq, show=False)

    elif hpi_file:
        # HPI-only — full detail always
        try:
            amp = fit_hpi_amplitudes(hpi_file, args.freq)
            amp = _resolve_hpi_only(amp)
            _print_diagnostics_hpi_only(amp)
            _build_figure_hpi_only(amp)
        except ValueError as exc:
            print(
                f'\n[ERROR] HPI fitting failed: {exc}\n'
                '  → Falling back to raw channel plot.\n',
                file=sys.stderr,
            )
            plot_hpi_raw_channels(hpi_file, hpifreq=args.freq, show=False)

    else:
        # Polhemus-only — no HPI fitting needed
        from ..io import load_polhemus
        pol = load_polhemus(pol_file)
        _print_diagnostics_pol_only(pol)
        _build_figure_pol_only(pol)

    plt.show()


if __name__ == '__main__':
    main()
