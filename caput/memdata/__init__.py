"""A backend format for in-memory or on-disk HDF5-like datasets.

It is useful to have a consistent API for data that is independent of whether that
data lives on disk or in memory. :py:mod:`h5py` provides this to a certain extent,
having :py:class:`h5py.Dataset` objects that act very much like :py:mod:`numpy` arrays.
:py:mod:`~caput.memdata` extends this, providing in-memory containers, analogous to
:py:class:`h5py.Group`, :py:class:`h5py.AttributeManager` and :py:class:`h5py.Dataset` objects.

In addition to these basic classes that copy the :py:mod:`h5py` API, a higher-level data
container is provided that utilizes these classes along with the :py:mod:`h5py` to provide
data that is transparently stored either in memory or on disk.

This also allows the creation and use of :py:mod:`~caput.memdata` objects which can hold
data distributed over a number of MPI processes. These :py:class:`MemDatasetDistributed`
datasets hold :py:class:`~caput.mpiarray.MPIArray` objects and can be written to, and
loaded from disk like normal :py:class:`~caput.memdata` objects.  Support for this must be
explicitly enabled in the root group at creation with the `distributed=True` flag.

.. warning::
    It has been observed that the parallel write of distributed datasets can
    lock up. This was when using macOS using `ompio` of OpenMPI 3.0.
    Switching to `romio` as the MPI-IO backend helped here, but please report
    any further issues.
"""

from ._memh5 import (
    MemAttrs as MemAttrs,
    MemGroup as MemGroup,
    MemDiskGroup as MemDiskGroup,
    MemDataset as MemDataset,
    MemDatasetCommon as MemDatasetCommon,
    MemDatasetDistributed as MemDatasetDistributed,
    BasicCont as BasicCont,
    ro_dict as ro_dict,
    copyattrs as copyattrs,
    is_group as is_group,
)

# For now, also re-export some private classes so
# that they get included in the API reference
from ._memh5 import (
    _MemObjMixin as _MemObjMixin,
    _BaseGroup as _BaseGroup,
    _StorageRoot as _StorageRoot,
    _Storage as _Storage,
)
from ._io import lock_file as lock_file
from . import fileformats as fileformats

__all__ = [
    "BasicCont",
    "MemAttrs",
    "MemDataset",
    "MemDatasetCommon",
    "MemDatasetDistributed",
    "MemDiskGroup",
    "MemGroup",
    "_BaseGroup",
    "_MemObjMixin",
    "_Storage",
    "_StorageRoot",
    "copyattrs",
    "fileformats",
    "is_group",
    "local_file",
    "ro_dict",
]
