"""Shared behavior options for the HPI command-line entry points."""


def add_input_options(parser, *, hpi_short, hpi_help, pol_help,
                      freq_default, freq_short=None):
    """Add HPI, digitisation, and fit-input arguments shared by both commands."""
    parser.add_argument('--hpi', hpi_short, metavar='PATH', help=hpi_help)
    parser.add_argument('--pol', '-p', metavar='PATH', help=pol_help)
    freq_default_text = 'ask' if freq_default is None else f'{freq_default:g}'
    freq_flags = ('--freq', freq_short) if freq_short else ('--freq',)
    parser.add_argument(*freq_flags, type=float, default=freq_default, metavar='HZ',
                        help=f'HPI drive frequency in Hz (default: {freq_default_text}).')
    parser.add_argument('--reffile', '-r', metavar='PATH', default=None,
                        help='Optional reference recording for noisy-channel detection; '
                        'otherwise a clean HPI tail is used when available.')
    parser.add_argument('--gof', type=float, default=0.95, metavar='THRESH',
                        help='Minimum dipole GOF for including a coil in the '
                        'device-to-head transform fit (default: 0.95).')


def add_fit_options(parser):
    parser.add_argument('--bad-channel-policy', choices=['auto', 'reference', 'none'],
                        default='auto', help='Noise detection: reference or clean tail (auto), '
                        'required reference, or none. Explicit bads are always excluded.')
    parser.add_argument('--activation-window-s', type=float, default=2.0,
                        help='Actual fit duration in seconds, centered on activation midpoint (default: 2).')
    parser.add_argument('--localization-grid', choices=['coarse', 'medium', 'fine', 'mne_stock'],
                        default='fine', help='Dipole OPM search grid (default: fine). '
                        'mne_stock matches MNE 1.12.1 guesses: minimum integration-point '
                        'radius, 10 mm spacing, 5 mm clearance, zero exclusion; same localizer.')
    parser.add_argument('--gof-comparison', choices=['inclusive', 'strict'], default='inclusive',
                        help='Coil inclusion uses >= (inclusive) or > (strict) the GOF limit.')
    matching = parser.add_mutually_exclusive_group()
    matching.add_argument('--matching-strategy', choices=['centroid_nearest', 'coordinate_nearest'],
                          default=None, help='Nearest-target matching after centering (default) or on raw coordinates.')
    matching.add_argument('--no-center-matching', dest='matching_strategy',
                          action='store_const', const='coordinate_nearest',
                          help='Compatibility alias for --matching-strategy coordinate_nearest.')
    parser.add_argument('--allow-repeated-matches', dest='unique_matches', action='store_false',
                        default=True, help='Allow repeated nearest targets; degenerate fits still raise.')
    parser.add_argument('--optimization', choices=['none', 'rigid_gof', 'rigid'], default='none',
                        help='Use no refinement or bounded field-GOF rigid refinement; '
                        'rigid is an alias for rigid_gof, which requires at least four accepted coils.')
    parser.add_argument('--settings-json', nargs='?', const='', default=None, metavar='DIR',
                        help='Write one JSON sidecar per data output (default beside output, named hpi_<output-stem>.json); optionally choose a sidecar directory.')


def fit_options(args):
    return dict(bad_channel_policy=args.bad_channel_policy,
                activation_window_s=args.activation_window_s, gof_comparison=args.gof_comparison,
                matching_strategy=args.matching_strategy, unique_matches=args.unique_matches,
                localization_grid=args.localization_grid,
                optim=args.optimization, settings_json=args.settings_json)
