"""caput.

Submodules
----------
.. autosummary::
    :toctree: _autosummary

    config
    mpiarray
    pipeline
    random
    truncate
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("caput")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
