"""
caput

Submodules
==========

.. autosummary::
    :toctree: _autosummary

    config
    interferometry
    memh5
    misc
    mpiarray
    mpiutil
    pfb
    pipeline
    time
    tod
    weighted_median
"""
from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
