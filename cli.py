"""Command-line entry point for opm_utility_scripts."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
import runpy
import sys


_COMMANDS = {
    'coregister': {
        'module': 'opm_utility_scripts.hpi.coregister',
        'help': 'Run HPI coregistration.',
    },
    'check': {
        'module': 'opm_utility_scripts.hpi.check',
        'help': 'Run HPI quality checks.',
    },
}


try:
    _VERSION = version('opm-utility-scripts')
except PackageNotFoundError:
    _VERSION = '0.1.0'


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='opmutil',
        description='Run OPM utility scripts through a single executable.',
        epilog='Commands: ' + ', '.join(f'{name} ({meta["help"]})' for name, meta in _COMMANDS.items()),
    )
    parser.add_argument(
        'command',
        choices=sorted(_COMMANDS),
        help='Utility command to run.',
    )
    parser.add_argument(
        'args',
        nargs=argparse.REMAINDER,
        help='Arguments passed through to the selected command.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'opmutil {_VERSION}',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    namespace = parser.parse_args(argv)

    module_name = _COMMANDS[namespace.command]['module']
    sys.argv = [f'opmutil {namespace.command}', *namespace.args]
    runpy.run_module(module_name, run_name='__main__', alter_sys=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
