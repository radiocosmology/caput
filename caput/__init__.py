"""caput.

Submodules
----------
.. autosummary::
    :toctree: _autosummary

    cache
    config
    fileformats
    memh5
    misc
    mpiarray
    mpiutil
    pfb
    pipeline
    profile
    random
    time
    tod
    tools
    truncate
    units
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("caput")
except PackageNotFoundError:
    # package is not installed
    pass

del version, PackageNotFoundError
