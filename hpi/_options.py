"""Shared behavior options for the HPI command-line entry points."""


def add_fit_options(parser):
    parser.add_argument('--bad-channel-policy', choices=['auto', 'reference', 'none'],
                        default='auto', help='Noise detection: reference or clean tail (auto), '
                        'required reference, or none. Explicit bads are always excluded.')
    parser.add_argument('--activation-window-s', type=float, default=2.0,
                        help='Actual fit duration in seconds, centered on activation midpoint (default: 2).')
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
    parser.add_argument('--optimization', choices=['none', 'rigid_gof', 'rigid'], default='rigid_gof',
                        help='Bounded field-GOF rigid refinement (default), or none; rigid is a compatibility alias.')
    parser.add_argument('--settings-json', nargs='?', const='', default=None, metavar='DIR',
                        help='Write one JSON sidecar per data output (default beside output, named hpi_<output-stem>.json); optionally choose a sidecar directory.')


def fit_options(args):
    return dict(bad_channel_policy=args.bad_channel_policy,
                activation_window_s=args.activation_window_s, gof_comparison=args.gof_comparison,
                matching_strategy=args.matching_strategy, unique_matches=args.unique_matches,
                optim=args.optimization, settings_json=args.settings_json)
