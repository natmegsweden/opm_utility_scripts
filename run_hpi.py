#!/usr/bin/env python3
"""
Run HPI scripts from inside the opm_utility_scripts directory.

Usage (from inside opm_utility_scripts/):

    python run_hpi.py coregister          # GUI coregistration
    python run_hpi.py check               # HPI quality check
    python run_hpi.py diagnose --hpi ...  # step-by-step diagnostic

This wrapper ensures the parent directory (NatMEG-utils/) is on sys.path
before importing any opm_utility_scripts modules, so that absolute imports
like ``from opm_utility_scripts.channels import ...`` resolve correctly
regardless of working directory.
"""
import sys
import runpy
from pathlib import Path

# Insert NatMEG-utils/ (parent of opm_utility_scripts/) at the front of
# sys.path so that ``import opm_utility_scripts`` always works.
_pkg_root = str(Path(__file__).resolve().parent.parent)
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

_SCRIPTS = {
    'coregister': 'opm_utility_scripts.hpi.coregister',
    'check':      'opm_utility_scripts.hpi.check',
    'diagnose':   'opm_utility_scripts.hpi.diagnose',
}

if len(sys.argv) < 2 or sys.argv[1] not in _SCRIPTS:
    print(f"Usage: python run_hpi.py [{' | '.join(_SCRIPTS)}] [args...]")
    sys.exit(1)

script = sys.argv[1]
# Shift argv so the target script sees its own args in sys.argv.
sys.argv = [script] + sys.argv[2:]

runpy.run_module(_SCRIPTS[script], run_name='__main__', alter_sys=True)
