"""caput.

Submodules
----------
.. autosummary::
    :toctree: _autosummary

    config
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

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("caput")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
