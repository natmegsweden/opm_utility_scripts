"""HPI coregistration and quality-check utilities."""

__all__ = ['fit_hpi', 'fit_hpi_amplitudes', 'apply_transform', 'save_raw',
           'compute_fit_diagnostics']


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module 'opm_utility_scripts.hpi' has no attribute {name!r}")
    from . import _core
    return getattr(_core, name)
