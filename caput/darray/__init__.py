"""Caput distributed arrays.

An MPI-aware numpy array and array utilities.
"""

from . import _invert_no_zero, _mpiarray
from ._invert_no_zero import *
from ._mpiarray import *


__all__ = [
    *_invert_no_zero.__all__,
    *_mpiarray.__all__,
]
