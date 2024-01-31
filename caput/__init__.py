"""caput.

Submodules
----------
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

from . import _version

__version__ = _version.get_versions()["version"]
