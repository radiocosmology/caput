"""caput.memdata.

A backend format for in-memory or on-disk hdf5-like datasets.

Submodules
----------
.. autosummary::
   :toctree: _autosummary

   memh5
   io
   fileformats
"""

from .memh5 import (
    MemGroup as MemGroup,
    BasicCont as BasicCont,
    MemDataset as MemDataset,
    MemDatasetDistributed as MemDatasetDistributed,
    copyattrs as copyattrs,
    is_group as is_group,
)
from .io import lock_file as lock_file
from . import fileformats as fileformats, io as io
