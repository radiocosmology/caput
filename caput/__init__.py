"""caput.

Cluster Astronomical Python Utilities.

Submodules
----------
.. autosummary::
    :toctree: _autosummary

    config
    mpiarray
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("caput")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
