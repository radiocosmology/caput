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

# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
