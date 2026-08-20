#!/usr/bin/env python3
"""Compatibility wrapper for the ``opmutil`` console script.

Preferred usage after installation::

    opmutil coregister
    opmutil check --pol ...

Running ``python run_hpi.py ...`` is still supported for local development.
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent

# Insert the project parent only when running from the source tree.
if (_here / 'pyproject.toml').exists():
    _pkg_root = str(_here.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)

from opm_utility_scripts.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
