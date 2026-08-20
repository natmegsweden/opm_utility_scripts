"""HPI coregistration and quality-check utilities."""

__all__ = ['fit_hpi', 'apply_transform', 'save_raw']


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module 'opm_utility_scripts.hpi' has no attribute {name!r}")
    from . import _core
    return getattr(_core, name)
