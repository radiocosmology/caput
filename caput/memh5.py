"""Module for making in-memory mock-ups of :mod:`h5py` objects.

It is sometimes useful to have a consistent API for data that is independent
of whether that data lives on disk or in memory. :mod:`h5py` provides this to a
certain extent, having :class:`h5py.Dataset` objects that act very much like
:mod:`numpy` arrays. :mod:`memh5` extends this, providing an in-memory
containers, analogous to :class:`h5py.Group`, :class:`h5py.AttributeManager` and
:class:`h5py.Dataset` objects.

In addition to these basic classes that copy the :mod:`h5py` API, A higher
level data container is provided that utilizes these classes along with the
:mod:`h5py` to provide data that is transparently stored either in memory or on
disk.

This also allows the creation and use of :mod:`memh5` objects which can hold
data distributed over a number of MPI processes. These
:class:`MemDatasetDistributed` datasets hold :class:`caput.mpiarray.MPIArray`
objects and can be written to, and loaded from disk like normal :class:`memh5`
objects.  Support for this must be explicitly enabled in the root group at
creation with the `distributed=True` flag.

.. warning::
    It has been observed that the parallel write of distributed datasets can
    lock up. This was when using macOS using `ompio` of OpenMPI 3.0.
    Switching to `romio` as the MPI-IO backend helped here, but please report
    any further issues.

Basic Classes
=============
- :py:class:`ro_dict`
- :py:class:`MemGroup`
- :py:class:`MemAttrs`
- :py:class:`MemDataset`
- :py:class:`MemDatasetCommon`
- :py:class:`MemDatasetDistributed`

High Level Container
====================
- :py:class:`MemDiskGroup`
- :py:class:`BasicCont`

Utility Functions
=================
- :py:meth:`attrs2dict`
- :py:meth:`is_group`
- :py:meth:`get_h5py_File`
- :py:meth:`copyattrs`
- :py:meth:`deep_group_copy`
"""

from __future__ import annotations

import datetime
import json
import logging
import posixpath
import warnings
from ast import literal_eval
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from . import fileformats, misc, mpiarray, mpiutil, tools

if TYPE_CHECKING:
    from mpi4py import MPI


logger = logging.getLogger(__name__)

try:
    import zarr
except ImportError as err:
    logger.info(f"zarr support disabled. Install zarr to change this: {err}")
    zarr_available = False
else:
    zarr_available = True

# Basic Classes
# -------------


class ro_dict(Mapping):
    """A dict that is read-only to the user.

    This class isn't strictly read-only but it cannot be modified through the
    traditional dict interface. This prevents the user from mistaking this for
    a normal dictionary.

    Provides the same interface for reading as the builtin python
    :class:`dict` but no methods for writing.

    Parameters
    ----------
    d : dict
        Initial data for the new dictionary.
    """

    def __init__(self, d=None):
        if not d:
            d = {}
        else:
            d = dict(d)
        self._dict = d

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return self._dict.__len__()

    def __iter__(self):
        return self._dict.__iter__()

    def __eq__(self, other):
        if not isinstance(other, ro_dict):
            return False
        return Mapping.__eq__(self, other) and tools.allequal(self._dict, other._dict)


class _Storage(dict):
    """Underlying container that provides storage backing for in-memory groups."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._attrs = MemAttrs()

    @property
    def attrs(self):
        """Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs
        """
        return self._attrs

    def __eq__(self, other):
        if not isinstance(other, _Storage):
            return False
        return dict.__eq__(self, other) and tools.allequal(self._attrs, other._attrs)


class _StorageRoot(_Storage):
    """Root level of the storage tree."""

    def __init__(self, distributed=False, comm=None):
        super().__init__()

        if distributed and comm is None:
            logger.debug(
                "No communicator set for distributed object, using `MPI.COMM_WORLD`"
            )
            comm = mpiutil.world

        self._comm = comm
        self._distributed = distributed

    @property
    def comm(self):
        """Reference to the MPI communicator."""
        return self._comm

    @property
    def distributed(self):
        return self._distributed

    def __getitem__(self, key):
        """Implements Hierarchical path lookup."""
        if "/" not in key:
            return super().__getitem__(key)

        # Format and split the path.
        key = format_abs_path(key)
        if key == "/":
            return self

        path_parts = key.split("/")[1:]

        # Crawl the path.
        out = self
        for part in path_parts:
            out = out[part]
        return out

    def __eq__(self, other):
        if not isinstance(other, _StorageRoot):
            return False
        return (
            _Storage.__eq__(self, other)
            and self._comm == other._comm
            and self._distributed == other._distributed
        )


class MemAttrs(dict):
    """In memory implementation of the :class:`h5py.AttributeManager`.

    Currently just a normal dictionary.
    """

    pass


class _MemObjMixin:
    """Mixin represents the identity of an in-memory h5py-like object.

    Implement a few attributes that all memh5 objects have, such as `parent`,
    and `file`.
    """

    @property
    def _group_class(self):
        return None

    # Here I have to implement __new__ not __init__ since MemDiskGroup
    # implements new and messes with parameters.
    def __init__(self, storage_root=None, name=""):
        self._storage_root = storage_root
        if storage_root is not None and not posixpath.isabs(name):
            # Should never happen, so this is mostly for debugging.
            raise ValueError("Must be given an absolute path.")
        self._name = name

    @property
    def name(self):
        """String giving the full path to this entry."""
        return self._name

    @property
    def parent(self):
        """Parent :class:`MemGroup` that contains this group."""
        parent_name, _ = posixpath.split(self.name)
        return self._group_class._from_storage_root(self._storage_root, parent_name)

    @property
    def file(self):
        """Not a file at all but the top most :class:`MemGroup` of the tree."""
        return self._group_class._from_storage_root(self._storage_root, "/")

    def __eq__(self, other):
        if hasattr(other, "_storage_root") and hasattr(other, "name"):
            return (self._storage_root is other._storage_root) and (
                self.name == other.name
            )
        return False

    def __neq__(self, other):
        return not self.__eq__(other)


class _BaseGroup(_MemObjMixin, Mapping):
    """Implement the majority of the Group interface.

    Subclasses must setup the underlying storage in thier constructors, as well
    as implement `create_group` and `create_dataset`.
    """

    @property
    def _group_class(self):
        return self.__class__

    @property
    def comm(self):
        """Reference to the MPI communicator."""
        return getattr(self._storage_root, "comm", None)

    @property
    def distributed(self):
        return getattr(self._storage_root, "distributed", False)

    @property
    def attrs(self):
        """Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs
        """
        return self._get_storage().attrs

    @classmethod
    def _from_storage_root(cls, storage_root, name):
        self = super().__new__(cls)
        super(_BaseGroup, self).__init__(storage_root, name)
        return self

    def _get_storage(self):
        return self._storage_root[self.name]

    def __getitem__(self, name):
        """Retrieve an object.

        The *name* may be a relative or absolute path
        """
        path = format_abs_path(posixpath.join(self.name, name))
        out = self._storage_root[path]

        # Cast the output.
        if is_group(out) or isinstance(out, _Storage):
            # Group like.
            return self._group_class._from_storage_root(self._storage_root, path)

        if isinstance(out, MemDataset):
            # Create back references for user facing mem datasets.
            out = out.view()
            out._storage_root = self._storage_root
            return out

        # H5py dataset.
        return out

    def __delitem__(self, name):
        """Delete item from group."""
        if name not in self:
            raise KeyError(f"Key {name} not present.")
        path = posixpath.join(self.name, name)
        parent_path, name = posixpath.split(path)
        parent = self._storage_root[parent_path]
        del parent[name]

    def __len__(self):
        return len(self._get_storage())

    def __iter__(self):
        keys = list(self._get_storage().keys())
        yield from keys

    def require_dataset(self, name, shape, dtype, **kwargs):
        """Require a dataset to exist, create if it doesn't.

        All arguments are passed through to create_dataset.

        """
        try:
            d = self[name]
        except KeyError:
            return self.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
        if is_group(d):
            msg = f"Entry '{name}' exists and is not a Dataset."
            raise TypeError(msg)

        return d

    def require_group(self, name):
        """Require a group to exist, create if it doesn't."""
        try:
            g = self[name]
        except KeyError:
            return self.create_group(name)
        if not is_group(g):
            msg = f"Entry '{name}' exists and is not a Group."
            raise TypeError(msg)

        return g


class MemGroup(_BaseGroup):
    """In memory implementation of the :class:`h5py.Group`.

    This class doubles as the memory implementation of :class:`h5py.File`,
    object, since the distinction between a file and a group for in-memory data
    is moot.

    Parameters
    ----------
    distributed : boolean, optional
        Allow memh5 object to hold distributed datasets.
    comm : MPI.Comm, optional
        MPI Communicator to distributed over. If not set, use :obj:`MPI.COMM_WORLD`.
    """

    def __init__(self, distributed=False, comm=None):
        # Default constructor is only used to create the root group.
        storage_root = _StorageRoot(distributed=distributed, comm=comm)
        name = "/"
        super().__init__(storage_root, name)

    @property
    def mode(self):
        """String indicating if group is readonly ("r") or read-write ("r+").

        :class:`MemGroup` is always read-write.
        """
        return "r+"

    @classmethod
    def from_group(cls, group):
        """Create a new instance by deep copying an existing group.

        Agnostic as to whether the group to be copied is a `MemGroup` or an
        `h5py.Group` (which includes `h5py.File` and `zarr.File` objects).
        """
        if isinstance(group, MemGroup):
            self = cls()
            deep_group_copy(group, self)
            return self

        if isinstance(group, (str, bytes)):
            file_format = fileformats.guess_file_format(group)
            return cls.from_file(group, file_format=file_format)

        raise RuntimeError(
            f"Can't create an instance from type {type(group).__name__} "
            f"(expected MemGroup, str or bytes)."
        )

    @classmethod
    def from_hdf5(
        cls,
        filename,
        distributed=False,
        hints=True,
        comm=None,
        selections=None,
        convert_dataset_strings=False,
        convert_attribute_strings=True,
        **kwargs,
    ):
        """Create a new instance by copying from an hdf5 group.

        Any keyword arguments are passed on to the constructor for `h5py.File`.

        Parameters
        ----------
        filename : string
            Name of file to load.
        distributed : boolean, optional
            Whether to load file in distributed mode.
        hints : boolean, optional
            If in distributed mode use hints to determine whether datasets are
            distributed or not.
        comm : MPI.Comm, optional
            MPI communicator to distributed over. If :obj:`None` use
            :obj:`MPI.COMM_WORLD`.
        selections : dict
            If this is not None, it should map dataset names to axis selections as valid
            numpy indexes.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to unicode. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to unicode. Default is `False`.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        group : memh5.Group
            Root group of loaded file.
        """
        return cls.from_file(
            filename,
            distributed,
            hints,
            comm,
            selections,
            convert_dataset_strings,
            convert_attribute_strings,
            file_format=fileformats.HDF5,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls,
        filename,
        distributed=False,
        hints=True,
        comm=None,
        selections=None,
        convert_dataset_strings=False,
        convert_attribute_strings=True,
        file_format=None,
        **kwargs,
    ):
        """Create a new instance by copying from a file group.

        Any keyword arguments are passed on to the constructor for `h5py.File` or
        `zarr.File`.

        Parameters
        ----------
        filename : string
            Name of file to load.
        distributed : boolean, optional
            Whether to load file in distributed mode.
        hints : boolean, optional
            If in distributed mode use hints to determine whether datasets are
            distributed or not.
        comm : MPI.Comm, optional
            MPI communicator to distributed over. If :obj:`None` use
            :obj:`MPI.COMM_WORLD`.
        selections : dict
            If this is not None, it should map dataset names to axis selections as valid
            numpy indexes.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to unicode. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to unicode. Default is `False`.
        file_format : `fileformats.FileFormat`, optional
            File format to use. Default is `None`, i.e. guess from the name.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        group : memh5.Group
            Root group of loaded file.
        """
        if comm is None:
            comm = mpiutil.world

        if file_format is None:
            file_format = fileformats.guess_file_format(filename)

        if comm is None:
            if distributed:
                warnings.warn(
                    "Cannot load file in distributed mode when there is no MPI"
                    "communicator!!"
                )
            distributed = False

        if not distributed or not hints:
            kwargs["mode"] = "r"
            with file_format.open(filename, **kwargs) as f:
                self = cls(distributed=distributed, comm=comm)
                deep_group_copy(
                    f,
                    self,
                    selections=selections,
                    convert_attribute_strings=convert_attribute_strings,
                    convert_dataset_strings=convert_dataset_strings,
                    file_format=file_format,
                )
        else:
            self = _distributed_group_from_file(
                filename,
                comm=comm,
                hints=hints,
                selections=selections,
                convert_attribute_strings=convert_attribute_strings,
                convert_dataset_strings=convert_dataset_strings,
                file_format=file_format,
            )

        return self

    def to_hdf5(
        self,
        filename,
        mode="w",
        hints=True,
        convert_attribute_strings=True,
        convert_dataset_strings=False,
        **kwargs,
    ):
        """Replicate object on disk in an hdf5 file.

        Any keyword arguments are passed on to the constructor for `h5py.File`.

        Parameters
        ----------
        filename : str
            File to save into.
        mode : str, optional
            Mode in which to open file
        hints : boolean, optional
            Whether to write hints into the file that described whether datasets
            are distributed, or not.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to a unicode type that HDF5
            understands. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to bytestrings. Default is `False`.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        self.to_file(
            filename,
            mode,
            hints,
            convert_attribute_strings,
            convert_dataset_strings,
            fileformats.HDF5,
            **kwargs,
        )

    def to_file(
        self,
        filename,
        mode="w",
        hints=True,
        convert_attribute_strings=True,
        convert_dataset_strings=False,
        file_format=None,
        **kwargs,
    ):
        """Replicate object on disk in an hdf5 or zarr file.

        Any keyword arguments are passed on to the constructor for `h5py.File` or `zarr.File`.

        Parameters
        ----------
        filename : str
            File to save into.
        mode : str, optional
            Mode in which t open file
        hints : boolean, optional
            Whether to write hints into the file that described whether datasets
            are distributed, or not.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to a unicode type that HDF5
            understands. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to bytestrings. Default is `False`.
        file_format : `fileformats.FileFormat`, optional
            File format to use. Default is `None`, i.e. guess from the name.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        if file_format is None:
            file_format = fileformats.guess_file_format(filename)

        if not self.distributed:
            with file_format.open(filename, mode, **kwargs) as f:
                deep_group_copy(
                    self,
                    f,
                    convert_attribute_strings=convert_attribute_strings,
                    convert_dataset_strings=convert_dataset_strings,
                    file_format=file_format,
                )
        else:
            _distributed_group_to_file(
                self,
                filename,
                mode,
                convert_attribute_strings=convert_attribute_strings,
                convert_dataset_strings=convert_dataset_strings,
                file_format=file_format,
            )

    def create_group(self, name):
        """Create a group within the storage tree."""
        path = format_abs_path(posixpath.join(self.name, name))
        try:
            self[name]
        except KeyError:
            pass
        else:
            raise ValueError(f"Entry {name} exists.")

        # If distributed, synchronise to ensure that we create group collectively
        if self.distributed:
            self.comm.Barrier()

        parent_name = "/"
        path_parts = path.split("/")
        # In this loop, exception guaranteed not to be raised on first
        # iteration, since we know that `parent_name + ''` exists.
        for part in path_parts:
            try:
                parent_name = posixpath.join(parent_name, part)
                parent_storage = self._storage_root[parent_name]
            except KeyError:
                parent_storage[part] = _Storage()
                parent_name = posixpath.join(parent_name, part)
                parent_storage = parent_storage[part]
            if not isinstance(parent_storage, _Storage):
                raise ValueError(f"Entry {parent_name} exists and is not a Group.")

        # Underlying storage has been created. Return the group object.
        return self[name]

    def create_dataset(
        self,
        name,
        shape=None,
        dtype=None,
        data=None,
        distributed=False,
        distributed_axis=None,
        chunks=None,
        compression=None,
        compression_opts=None,
        **kwargs,
    ):
        """Create a new dataset.

        Parameters
        ----------
        name : string
            Dataset name.
        shape : tuple, optional
            Shape tuple. This gives the global shape for a distributed dataset.
        dtype : np.dtype, optional
            Numpy datatype of the dataset.
        data : np.ndarray or MPIArray, optional
            Data array to initialise from. Uses a view of the original where possible.
        distributed : boolean, optional
            Create a distributed dataset or not.
        distributed_axis : int, optional
            Axis to distribute the data over. If specified with initialisation
            data this will cause create a copy with the correct distribution.
        chunks
            Chunking arguments for dataset
        compression : str or int
            Name or identifier of HDF5 or Zarr compression filter.
        compression_opts
            See HDF5 and Zarr documentation for compression filters.
            Compression options for the dataset.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        dset : memh5.MemDataset
        """
        parent_name, name = posixpath.split(posixpath.join(self.name, name))
        parent_name = format_abs_path(parent_name)
        parent_storage = self.require_group(parent_name)._get_storage()

        # If distributed, synchronise to ensure that we create group collectively
        if self.distributed:
            self.comm.Barrier()

        if self.comm is None:
            if distributed:
                warnings.warn(
                    "Cannot create distributed dataset when there is no MPI communicator!!"
                )
            distributed = False

        if kwargs:
            msg = (
                "No extra keyword arguments accepted, this is not an hdf5"
                " object but a memory object mocked up to look like one."
            )
            raise TypeError(msg)
            # XXX In future could accept extra arguments and use them if
            # writing to disk.

        # If data is set, copy out params from it.
        if data is not None:
            if shape is None:
                shape = data.shape
            if dtype is None:
                dtype = data.dtype

        # Otherwise shape is required.
        if shape is None:
            raise ValueError("shape must be provided.")
        # Default dtype is float.
        if dtype is None:
            dtype = np.float64

        # Convert to numpy dtype.
        dtype = np.dtype(dtype)

        # Create distributed dataset if data is an MPIArray
        if isinstance(data, mpiarray.MPIArray) and data.comm is not None:
            distributed = True

        # Enforce that distributed datasets can only exist in distributed memh5 groups.
        if not self.distributed and distributed:
            raise RuntimeError(
                "Cannot create a distributed dataset in a non-distributed group."
            )

        # Set the properties of the new dataset
        full_path_name = posixpath.join(parent_name, name)
        storage_root = None  # Do no store the storage root. Creates cyclic references.

        # If data is set (and consistent with shape/type), initialise the numpy array from it.
        if (
            data is not None
            and shape == data.shape
            and dtype is data.dtype
            and hasattr(data, "view")
        ):
            # Create parallel array if requested
            if distributed:
                # Ensure we are creating from an MPIArray
                if not isinstance(data, mpiarray.MPIArray):
                    raise TypeError(
                        "Can only create distributed dataset from MPIArray."
                    )

                # Ensure that we are distributing over the same communicator
                if data.comm != self.comm:
                    raise RuntimeError(
                        "MPI communicator of array must match that of memh5 group."
                    )

                # If the distributed_axis is specified ensure the data is distributed along it.
                if distributed_axis is not None:
                    data = data.redistribute(axis=distributed_axis)

                # Create distributed dataset
                new_dataset = MemDatasetDistributed.from_mpi_array(
                    data,
                    name=full_path_name,
                    storage_root=storage_root,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                # Create common dataset
                new_dataset = MemDatasetCommon.from_numpy_array(
                    data,
                    name=full_path_name,
                    storage_root=storage_root,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )

        # Otherwise create an empty array and copy into it (if needed)
        else:
            # Just copy the data.
            if distributed:
                # Ensure that distributed_axis is set.
                if distributed_axis is None:
                    raise RuntimeError(
                        "Distributed axis must be specified when creating dataset."
                    )

                new_dataset = MemDatasetDistributed(
                    shape=shape,
                    dtype=dtype,
                    axis=distributed_axis,
                    comm=self.comm,
                    name=full_path_name,
                    storage_root=storage_root,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )
            else:
                new_dataset = MemDatasetCommon(
                    shape=shape,
                    dtype=dtype,
                    name=full_path_name,
                    storage_root=storage_root,
                    chunks=chunks,
                    compression=compression,
                    compression_opts=compression_opts,
                )

            if data is not None:
                new_dataset[:] = data[:]

        # Add new dataset to group
        parent_storage[name] = new_dataset

        # Ensure __getitem__ is called.
        return self[full_path_name]

    def dataset_common_to_distributed(self, name, distributed_axis=0):
        """Convert a common dataset to a distributed one.

        Parameters
        ----------
        name : string
            Dataset name.
        distributed_axis : int, optional
            Axis to distribute the data over.

        Returns
        -------
        dset : memh5.MemDatasetDistributed
        """
        dset = self[name]

        if dset.distributed:
            warnings.warn(
                "%s is already a distributed dataset, redistribute it along the required axis %d"
                % (name, distributed_axis)
            )
            dset.redistribute(distributed_axis)
            return dset

        dset_shape = dset.shape
        dset_type = dset.dtype
        dset_chunks = dset.chunks
        dset_compression = dset.compression
        dset_compression_opts = dset.compression_opts
        dist_len = dset_shape[distributed_axis]
        _, sd, ed = mpiutil.split_local(dist_len, comm=self.comm)
        md = mpiarray.MPIArray(
            dset_shape, axis=distributed_axis, comm=self.comm, dtype=dset_type
        )
        md.local_array[:] = dset[sd:ed].copy()
        attr_dict = {}  # temporarily save attrs of this dataset
        copyattrs(dset.attrs, attr_dict)
        del dset
        new_dset = self.create_dataset(
            name,
            shape=dset_shape,
            dtype=dset_type,
            chunks=dset_chunks,
            compression=dset_compression,
            compression_opts=dset_compression_opts,
            data=md,
            distributed=True,
            distributed_axis=distributed_axis,
        )
        copyattrs(attr_dict, new_dset.attrs)

        return new_dset

    def dataset_distributed_to_common(self, name):
        """Convert a distributed dataset to a common one.

        Parameters
        ----------
        name : string
            Dataset name.

        Returns
        -------
        dset : memh5.MemDatasetCommon
        """
        dset = self[name]

        if dset.common:
            warnings.warn(f"{name} is already a common dataset, no need to convert")
            return dset

        dset_shape = dset.shape
        dset_type = dset.dtype
        dset_chunks = dset.chunks
        dset_compression = dset.compression
        dset_compression_opts = dset.compression_opts
        global_array = np.zeros(dset_shape, dtype=dset_type)
        local_start = dset.local_offset
        nproc = 1 if self.comm is None else self.comm.size
        # gather local distributed dataset to a global array for all procs
        for rank in range(nproc):
            mpiutil.gather_local(
                global_array, dset.local_data, local_start, root=rank, comm=self.comm
            )
        attr_dict = {}  # temporarily save attrs of this dataset
        copyattrs(dset.attrs, attr_dict)
        del dset
        new_dset = self.create_dataset(
            name,
            data=global_array,
            shape=dset_shape,
            dtype=dset_type,
            chunks=dset_chunks,
            compression=dset_compression,
            compression_opts=dset_compression_opts,
        )
        copyattrs(attr_dict, new_dset.attrs)

        return new_dset


class MemDataset(_MemObjMixin):
    """Base class for an in memory implementation of :class:`h5py.Dataset`.

    This is only an abstract base class. Use :class:`MemDatasetCommon` or
    :class:`MemDatasetDistributed`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._attrs = MemAttrs()

        # Must be implemented by child classes
        self._data = NotImplemented

    @property
    def _group_class(self):
        return MemGroup

    def copy(self, order: str = "A", shallow: bool = False) -> MemDataset:
        """Create a new MemDataset from an existing one.

        This creates a deep copy by default.

        Parameters
        ----------
        order
            Memory layout of copied data. See
            https://numpy.org/doc/stable/reference/generated/numpy.copy.html
        shallow
            True if this should be a shallow copy

        Returns
        -------
        new_data
            deep copy of this dataset
        """
        new = self.__class__.__new__(self.__class__)
        super(MemDataset, new).__init__(name=self.name, storage_root=self._storage_root)

        _copy = deepcopy if not shallow else lambda x: x
        # Call the properties rather than the underlying values so that an error
        # is properly raised if they are not implemented. Blindly use deepcopy as
        # we don't make assumptions about immutability
        new._chunks = _copy(self.chunks)
        new._compression = _copy(self.compression)
        new._compression_opts = _copy(self.compression_opts)
        new._attrs = _copy(self._attrs)

        if shallow:
            new._data = self._data
        else:
            new._data = deep_copy_dataset(self._data, order=order)

        return new

    def __deepcopy__(self, memo, /) -> MemDataset:
        """Called when copy.deepcopy is called on this class."""
        return self.copy()

    def view(self) -> MemDataset:
        """Return a pseudo-view of this dataset.

        This technically makes a new `MemDataset` object, but the
        underlying attributes are views.

        Returns
        -------
        view
            shallow copy of this dataset
        """
        return self.copy(shallow=True)

    @property
    def attrs(self):
        """Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs

        """
        return self._attrs

    def resize(self):
        """h5py datasets reshape() is different from numpy reshape."""
        msg = "Dataset reshaping not allowed. Perhapse make an new array view."
        raise NotImplementedError(msg)

    @property
    def shape(self):
        """Shape of the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def dtype(self):
        """Numpy data type of the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def chunks(self):
        """Chunk shape of the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def compression(self):
        """Name or identifier of HDF5 compression filter for the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def compression_opts(self):
        """Compression options for the dataset.

        See HDF5 documentation for compression filters.
        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    def __getitem__(self, obj):
        raise NotImplementedError("Not implemented in base class.")

    def __setitem__(self, obj, val):
        raise NotImplementedError("Not implemented in base class.")

    def __len__(self):
        raise NotImplementedError("Not implemented in base class.")

    def __eq__(self, other):
        if not isinstance(other, MemDataset):
            return False
        return _MemObjMixin.__eq__(self, other) and tools.allequal(
            self._attrs, other._attrs
        )


class MemDatasetCommon(MemDataset):
    """In memory implementation of :class:`h5py.Dataset`.

    Inherits from :class:`MemDataset`. Encapsulates a numpy array mocked up to
    look like an hdf5 dataset. Similar to h5py datasets, this implements
    slicing like a numpy array but as it is not actually a many operations
    won't work (e.g. ufuncs).

    Parameters
    ----------
    shape : tuple
        Shape of array to initialise.
    dtype : numpy dtype
        Type of array to create.
    """

    def __init__(
        self,
        shape,
        dtype,
        chunks=None,
        compression=None,
        compression_opts=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._data = np.zeros(shape, dtype)
        self._chunks = chunks
        self._compression = compression
        self._compression_opts = compression_opts

    @classmethod
    def from_numpy_array(
        cls, data, chunks=None, compression=None, compression_opts=None, **kwargs
    ):
        """Initialise from a numpy array.

        Parameters
        ----------
        data : np.ndarray
            Array to initialise from.
        chunks
            Chunking arguments
        compression : str or int
            Name or identifier of HDF5 or Zarr compression filter.
        compression_opts
            See HDF5 and Zarr documentation for compression filters.
            Compression options for the dataset.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        dset : MemDatasetCommon
            Dataset encapsulating the numpy array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Object must be a numpy array (or subclass).")

        self = cls.__new__(cls)
        super(MemDatasetCommon, self).__init__(**kwargs)

        self._data = ensure_native_byteorder(data)
        self._chunks = chunks
        self._compression = compression
        self._compression_opts = compression_opts

        return self

    @property
    def comm(self):
        """Reference to the MPI communicator."""
        return

    @property
    def common(self):
        """Assert that this is a common dataset."""
        return True

    @property
    def distributed(self):
        """Assert that this is not a distributed dataset."""
        return False

    @property
    def data(self):
        """Access the underlying data array."""
        return self._data

    @property
    def local_data(self):
        """Access the underlying data array."""
        return self._data

    @property
    def shape(self):
        """Access the shape of the underlying array."""
        return self._data.shape

    @property
    def dtype(self):
        """Access the data type."""
        return self._data.dtype

    @property
    def chunks(self):
        """Access the data chunking information."""
        return self._chunks

    @chunks.setter
    def chunks(self, val):
        """Set the data chunking information."""
        if val is None:
            chunks = val
        elif len(val) != len(self.shape):
            raise ValueError(
                f"Chunk size {val} is not compatible with dataset shape {self.shape}."
            )
        else:
            chunks = ()
            for i, l in enumerate(self.shape):
                chunks += (min(val[i], l),)
        self._chunks = chunks

    @property
    def compression(self):
        """Access the compression information."""
        return self._compression

    @compression.setter
    def compression(self, val):
        """Set the compression information."""
        self._compression = val

    @property
    def compression_opts(self):
        """Access the compression options."""
        return self._compression_opts

    @compression_opts.setter
    def compression_opts(self, val):
        """Set the compression options."""
        self._compression_opts = val

    def __getitem__(self, obj):
        """Index directly into the data array."""
        return self._data[obj]

    def __setitem__(self, obj, val):
        """Index directly into the data array and set values at that location."""
        self._data[obj] = val

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        # This needs to be implemented to stop craziness happening when doing
        # np.array(dset)
        return self._data.__iter__()

    def __repr__(self):
        return f'<memh5 common dataset {self._name!r}: shape {self.shape!r}, type "{self.dtype!r}">'

    def __eq__(self, other):
        if not isinstance(other, MemDatasetCommon):
            return False
        return (
            MemDataset.__eq__(self, other)
            and tools.allequal(self._data, other._data)
            and tools.allequal(self._chunks, other._chunks)
            and tools.allequal(self._compression, other._compression)
            and tools.allequal(self._compression_opts, other._compression_opts)
        )


class MemDatasetDistributed(MemDataset):
    """Parallel, in-memory implementation of :class:`h5py.Dataset`.

    Inherits from :class:`MemDataset`. Encapsulates an :class:`MPIArray` mocked
    up to look like an `h5py` dataset.  Similar to h5py datasets, this
    implements slicing like a numpy array but as it is not actually a many
    operations won't work (e.g. ufuncs).

    Parameters
    ----------
    shape : tuple
        Shape of array to initialise. This is the *global* shape.
    dtype : numpy dtype
        Type of array to create.
    axis : int, optional
        Index of axis to distribute the array over.
    comm : MPI.Comm, optional
        MPI communicator to distribute over. If :obj:`None` use
        :obj:`MPI.COMM_WORLD`.
    """

    def __init__(
        self,
        shape,
        dtype,
        axis=0,
        comm=None,
        chunks=None,
        compression=None,
        compression_opts=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._data = mpiarray.MPIArray(shape, axis=axis, comm=comm, dtype=dtype)
        self._chunks = chunks
        self._compression = compression
        self._compression_opts = compression_opts

    @classmethod
    def from_mpi_array(
        cls, data, chunks=None, compression=None, compression_opts=None, **kwargs
    ):
        """Initialise from a MPIArray.

        Parameters
        ----------
        data : mpiarray.MPIArray
            Array to initialise from.
        chunks
            Chunking arguments
        compression : str or int
            Name or identifier of HDF5 or Zarr compression filter.
        compression_opts
            See HDF5 and Zarr documentation for compression filters.
            Compression options for the dataset.
        **kwargs : dict
            Arbitrary keyword arguments.

        Returns
        -------
        dset : MemDatasetDistributed
            Dataset encapsulating the MPIArray.
        """
        if not isinstance(data, mpiarray.MPIArray):
            raise TypeError("Object must be a numpy array (or subclass).")

        self = cls.__new__(cls)
        super(MemDatasetDistributed, self).__init__(**kwargs)

        self._data = data
        self._chunks = chunks
        self._compression = compression
        self._compression_opts = compression_opts

        return self

    @property
    def common(self):
        """Assert that this is not a common dataset."""
        return False

    @property
    def distributed(self):
        """Assert that this is a distributed dataset."""
        return True

    @property
    def data(self):
        """Access the underlying data array."""
        return self._data

    @property
    def local_data(self):
        """Access tthe underlying local data as a numpy array."""
        return self._data.local_array

    @property
    def shape(self):
        """Access the global shape of the array."""
        return self.global_shape

    @property
    def global_shape(self):
        """Global shape of the distributed dataset.

        The shape of the whole array that is distributed between multiple nodes.
        """
        return self._data.global_shape

    @property
    def local_shape(self):
        """Local shape of the distributed dataset.

        The shape of the part of the distributed array that is allocated to *this* node.
        """
        return self._data.local_shape

    @property
    def local_offset(self):
        """Access the local offset of the array on this rank."""
        return self._data.local_offset

    @property
    def dtype(self):
        """The numpy data type of the dataset."""
        return self._data.dtype

    @property
    def chunks(self):
        """Acess the chunk shape of the dataset."""
        return self._chunks

    @chunks.setter
    def chunks(self, val):
        """Set the chunk shape of the dataset."""
        if val is None:
            chunks = val
        elif len(val) != len(self.shape):
            raise ValueError(
                f"Chunk size {val} is not compatible with dataset shape {self.shape}."
            )
        else:
            chunks = ()
            for i, l in enumerate(self.shape):
                chunks += (min(val[i], l),)
        self._chunks = chunks

    @property
    def compression(self):
        """Access compression information."""
        return self._compression

    @compression.setter
    def compression(self, val):
        """Set compression information."""
        self._compression = val

    @property
    def compression_opts(self):
        """Access compression options."""
        return self._compression_opts

    @compression_opts.setter
    def compression_opts(self, val):
        """Set compression options."""
        self._compression_opts = val

    @property
    def distributed_axis(self):
        """The index of the axis over which this dataset is distributed."""
        return self._data.axis

    @property
    def comm(self):
        """Reference to the MPI communicator."""
        return self._data._comm

    def redistribute(self, axis):
        """Change the axis that the dataset is distributed over.

        Parameters
        ----------
        axis : integer
            Axis to distribute over.
        """
        if self._storage_root is not None:
            # This is a view, and we should modify the base dataset
            base_dset = self._storage_root[self.name]
            base_dset.redistribute(axis=axis)
            self._data = base_dset._data
        else:
            # This is the the base dataset, call the MPIArray redistribution
            self._data = self._data.redistribute(axis=axis)

    def __getitem__(self, obj):
        return self._data.global_slice[obj]

    def __setitem__(self, obj, val):
        self._data.global_slice[obj] = val

    def __iter__(self):
        # This needs to be implemented to stop craziness happening when doing
        # np.array(dset)
        return self._data.__iter__()

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return (
            f"<memh5 distributed dataset {self._name!r}: "
            f"global_shape {self.global_shape!r}, "
            f"dist_axis {self.distributed_axis!r}, "
            f"type '{self.dtype!r}'>"
        )

    def __eq__(self, other):
        if not isinstance(other, MemDatasetDistributed):
            return False
        return (
            MemDataset.__eq__(self, other)
            and tools.allequal(self._data, other._data)
            and tools.allequal(self._chunks, other._chunks)
            and tools.allequal(self._compression, other._compression)
            and tools.allequal(self._compression_opts, other._compression_opts)
        )


# Higher Level Data Containers
# ----------------------------


class MemDiskGroup(_BaseGroup):
    """Container whose data may either be stored on disk or in memory.

    This container is intended to have the same basic API :class:`h5py.Group`
    and :class:`MemGroup` but whose underlying data could live either on disk
    or in memory.

    Aside from providing a few convenience methods, this class isn't that
    useful by itself. It is almost as easy to use :class:`h5py.Group`
    or :class:`MemGroup` directly. Where it becomes more useful is for creating
    more specialized data containers which can subclass this class.  A basic
    but useful example is provided in :class:`BasicCont`.

    This class also supports the same distributed features as :class:`MemGroup`,
    but only when wrapping that class. Attempting to create a distributed object
    wrapping a :class:`h5py.File` object will raise an exception. For similar
    reasons, :meth:`MemDiskGroup.to_disk` will not work, however,
    :meth:`MemDiskGroup.save` will work fine.

    Parameters
    ----------
    data_group : :class:`h5py.Group`, :class:`MemGroup` or string, optional
        Underlying :mod:`h5py` like data container where data will be stored.
        If a string, open a h5py file with that name. If not
        provided a new :class:`MemGroup` instance will be created.
    distributed : boolean, optional
        Allow the container to hold distributed datasets.
    comm : MPI.Comm, optional
        MPI Communicator to distributed over. If not set, use :obj:`MPI.COMM_WORLD`.
    detect_subclass: boolean, optional
        If *data_group* is specified, whether to inspect for a
        '__memh5_subclass' attribute which specifies a subclass to return.
    file_format : `fileformats.FileFormat`
        File format to use. File format will be guessed if not supplied. Default `None`.
    """

    def __init__(self, data_group=None, distributed=False, comm=None, file_format=None):
        toclose = False

        if comm is None:
            comm = mpiutil.world

        if distributed and comm is None:
            warnings.warn(
                "Cannot create distributed MemDiskGroup when there is no MPI communicator!!"
            )
            distributed = False

        # If data group is not set, initialise a new MemGroup
        if data_group is None:
            data_group = MemGroup(distributed=distributed, comm=comm)
        # If it is a MemDiskGroup then initialise a shallow copy
        elif isinstance(data_group, MemDiskGroup):
            data_group = data_group._storage_root
        # Otherwise, presume it is an HDF5 Group-like object (which includes
        # MemGroup and h5py.Group).
        else:
            data_group, toclose = get_file(
                data_group, mode="a", file_format=file_format
            )
            # Zarr arrays are automatically flushed and closed
            toclose = False if file_format == fileformats.HDF5 else toclose

        # Check the distribution settings
        if distributed:
            if isinstance(data_group, h5py.Group) or (
                zarr_available and isinstance(data_group, zarr.Group)
            ):
                raise ValueError(
                    "Distributed MemDiskGroup cannot be created around h5py or zarr objects."
                )
            # Check parallel distribution is the same
            if not data_group.distributed:
                raise ValueError(
                    "Cannot create MemDiskGroup with different distributed setting to MemGroup to wrap."
                )
            # Check parallel communicator is the same
            if comm and comm != data_group.comm:
                raise ValueError(
                    "Cannot create MemDiskGroup with different MPI communicator to MemGroup to wrap."
                )

        self._toclose = toclose
        super().__init__(storage_root=data_group, name=data_group.name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @classmethod
    def _detect_subclass_path(cls, group):
        """Determine the true class of `group` from its attributes (otherwise `cls`)."""
        return group.attrs.get("__memh5_subclass", None)

    @classmethod
    def _resolve_subclass(cls, clspath):
        """Validate and return the subclass corresponding to classpath."""
        if clspath is None:
            return cls

        # Try to get a reference to the requested class (warn if we cannot find it)
        try:
            new_cls = misc.import_class(clspath)
        except (ImportError, KeyError):
            warnings.warn(f"Could not import memh5 subclass {clspath}")

        # Check that it is a subclass of MemDiskGroup
        if not issubclass(new_cls, MemDiskGroup):
            raise RuntimeError(
                f"Requested type ({clspath}) is not an subclass of memh5.MemDiskGroup."
            )
        return new_cls

    @classmethod
    def from_group(cls, data_group=None, detect_subclass=True):
        """Create data object from a given group.

        This wraps the given group object, optionally returning the correct
        subclass. This does *not* call `__init__` on the subclass when
        this happens.

        Parameters
        ----------
        data_group : :class:`h5py.Group`, :class:`MemGroup` or string, optional
            :mod:`h5py` like data containerto wrap.
        detect_subclass: boolean, optional
            If *data_group* is specified, whether to inspect for a
            '__memh5_subclass' attribute which specifies a subclass to return.

        Returns
        -------
        grp : MemDiskGroup
        """
        if detect_subclass:
            new_cls = cls._resolve_subclass(cls._detect_subclass_path(data_group))
        else:
            new_cls = cls

        self = new_cls.__new__(new_cls)
        MemDiskGroup.__init__(self, data_group=data_group)

        return self

    @property
    def _data(self):
        """_data was renamed to _storage_root. This added for compatibility."""
        return self._storage_root

    def _finish_setup(self):
        """Finish the class setup *after* importing from a file."""
        pass

    def close(self):
        """Closes file if on disk if file was opened on initialization."""
        if self.ondisk and hasattr(self, "_toclose") and self._toclose:
            self._storage_root.close()

        if hasattr(self, "_lockfile") and (self.comm is None or self.comm.rank is None):
            fileformats.remove_file_or_dir(self._lockfile)

    def __getitem__(self, name):
        """Retrieve an object.

        The *name* may be a relative or absolute path
        """
        value = super().__getitem__(name)
        path = value.name
        if is_group(value):
            if not self.group_name_allowed(path):
                msg = f"Access to group {path} not allowed."
                raise KeyError(msg)
        else:
            if not self.dataset_name_allowed(path):
                msg = f"Access to dataset {path} not allowed."
                raise KeyError(msg)
        return value

    def __len__(self):
        n = 0
        for _ in self:
            n += 1
        return n

    def __iter__(self):
        for key in super().__iter__():
            try:
                _ = self[key]
            except KeyError:
                # This key name is not allowed (see __getitem__)
                continue
            yield key

    @property
    def ondisk(self):
        """Whether the data is stored on disk as opposed to in memory."""
        return hasattr(self, "_storage_root") and (
            isinstance(self._storage_root, h5py.File)
            or (zarr_available and isinstance(self._storage_root, zarr.Group))
        )

    @classmethod
    def _make_selections(cls, sel_args):
        """Overwrite this method in your subclass if you want to implement axis downselection.

        This may be useful when loading a container from an HDF5 file

        Parameters
        ----------
        sel_args : dict
            Should contain valid numpy indexes as values and axis names (str) as keys.

        Returns
        -------
        dict
            Mapping of dataset names to numpy indexes for downselection of the data.
            May include multiple layers of dicts to map the dataset tree
        """
        return

    # For creating new instances. #

    @classmethod
    def from_file(
        cls,
        file_,
        ondisk=False,
        distributed=False,
        comm=None,
        detect_subclass=True,
        convert_attribute_strings=None,
        convert_dataset_strings=None,
        file_format=None,
        **kwargs,
    ):
        """Create data object from analysis hdf5 file, store in memory or on disk.

        If *ondisk* is True, do not load into memory but store data in h5py objects
        that remain associated with the file on disk. This is almost identical
        to the default constructor, when providing a file as the *data_group*
        object, however provides more flexibility when opening the file through
        the additional keyword arguments.

        This does *not* call `__init__` on the subclass when restoring.

        Parameters
        ----------
        file_ : string or :class:`h5py.Group` object
            File with the hdf5 data. File must be compatible with memh5 objects.
        ondisk : bool
            Whether the data should be stored in-place in *file_* or should be copied
            into memory.
        distributed : boolean, optional
            Allow the container to hold distributed datasets.
        comm : MPI.Comm, optional
            MPI Communicator to distributed over. If not set, use :obj:`MPI.COMM_WORLD`.
        detect_subclass: boolean, optional
            If *data_group* is specified, whether to inspect for a
            '__memh5_subclass' attribute which specifies a subclass to return.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to unicode. If not specified, look
            up the name as a class attribute to find a default, and otherwise use
            `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to unicode. If not specified, look
            up the name as a class attribute to find a default, and otherwise use
            `False`.
        <axis_name>_sel : list or slice
            Axis selections can be given to only read a subset of the containers. A
            slice can be given, or a list of specific array indices for that axis.
        file_format : `fileformats.FileFormat`
            File format to use. Default is `None`, i.e. guess from file name.
        **kwargs : any other arguments
            Any additional keyword arguments are passed to :class:`h5py.File`'s
            constructor if *file_* is a filename and silently ignored otherwise.
        """
        if file_format is None and not is_group(file_):
            file_format = fileformats.guess_file_format(file_)

        if file_format == fileformats.Zarr and not zarr_available:
            raise RuntimeError("Unable to read zarr file, please install zarr.")

        # Get a value for the conversion parameters, looking up on the class type if
        # not supplied
        if convert_attribute_strings is None:
            convert_attribute_strings = getattr(cls, "convert_attribute_strings", True)
        if convert_dataset_strings is None:
            convert_dataset_strings = getattr(cls, "convert_dataset_strings", False)

        lockfile = None

        if not ondisk:
            if (zarr_available and isinstance(file_, zarr.Group)) or isinstance(
                file_, h5py.Group
            ):
                file_ = file_.filename

            if "mode" in kwargs:
                del kwargs["mode"]

            # Look for *_sel parameters in kwargs, collect and remove them from kwargs
            sel_args = {}
            for a in list(kwargs):
                if a.endswith("_sel"):
                    sel_args[a[:-4]] = kwargs.pop(a)

            # Axis selections won't warn if called from a baseclass without detecting
            # the subclass
            if sel_args and not detect_subclass and cls in [MemDiskGroup, BasicCont]:
                warnings.warn(
                    "Cannot process axis selections as subclass is not known."
                )

            # Map selections to datasets
            sel = cls._make_selections(sel_args)

            data = MemGroup.from_file(
                file_,
                distributed=distributed,
                comm=comm,
                selections=sel,
                convert_attribute_strings=convert_attribute_strings,
                convert_dataset_strings=convert_dataset_strings,
                file_format=file_format,
                **kwargs,
            )
            toclose = False
        else:
            # Again, a compatibility hack
            if is_group(file_):
                data = file_
                toclose = False
            else:
                kwargs.setdefault("mode", "r")
                if distributed and file_format == fileformats.Zarr:
                    lockfile = f"{file_}.sync"
                    kwargs["synchronizer"] = zarr.ProcessSynchronizer(lockfile)

                # NOTE: hints is not supported for ondisk files, remove the argument in
                # case it's been passed indirectly
                kwargs.pop("hints", None)
                data = file_format.open(file_, **kwargs)
                toclose = file_format == fileformats.HDF5

        # Here we explicitly avoid calling __init__ on any derived class. Like
        # with a pickle we want to restore the saved state only.
        self = cls.from_group(data_group=data, detect_subclass=detect_subclass)

        # ... skip the class initialisation, and use a special method
        self._finish_setup()

        self._toclose = toclose

        if lockfile is not None:
            self._comm = comm
            self._lockfile = lockfile
        return self

    # Methods for manipulating and building the class. #

    @staticmethod
    def group_name_allowed(name):
        """Used by subclasses to restrict creation of and access to groups.

        This method is called by :meth:`create_group`, :meth:`require_group`,
        and :meth:`__getitem__` to check that the supplied group name is
        allowed.

        The idea is that subclasses that want to specialize and restrict the
        layout of the data container can implement this method instead of
        re-implementing the above mentioned methods.

        Parameters
        ----------
        name: string
            Absolute path to proposed group.

        Returns
        -------
        allowed : bool
            ``True``

        """
        return True

    @staticmethod
    def dataset_name_allowed(name):
        """Used by subclasses to restrict creation of and access to datasets.

        This method is called by :meth:`create_dataset`,
        :meth:`require_dataset`, and :meth:`__getitem__` to check that the
        supplied group name is allowed.

        The idea is that subclasses that want to specialize and restrict the
        layout of the data container can implement this method instead of
        re-implementing the above mentioned methods.

        Parameters
        ----------
        name: string
            Absolute path to proposed dataset.

        Returns
        -------
        allowed : bool
            ``True``

        """
        return True

    def create_dataset(self, name, *args, **kwargs):
        """Create and return a new dataset.

        All parameters are passed through to the :meth:`create_dataset` method of
        the underlying storage, whether it be an :class:`h5py.Group` or a
        :class:`MemGroup`.
        """
        path = posixpath.join(self.name, name)
        if not self.dataset_name_allowed(path):
            msg = f"Dataset name {path} not allowed."
            raise ValueError(msg)

        return self._data.create_dataset(path, *args, **kwargs)

    def dataset_common_to_distributed(self, name, distributed_axis=0):
        """Convert a common dataset to a distributed one.

        Parameters
        ----------
        name : string
            Dataset name.
        distributed_axis : int, optional
            Axis to distribute the data over.

        Returns
        -------
        dset : memh5.MemDatasetDistributed
        """
        if isinstance(self._data, MemGroup):
            return self._data.dataset_common_to_distributed(name, distributed_axis)

        raise RuntimeError(
            f"Can not convert a h5py or zarr dataset {name} to distributed"
        )

    def dataset_distributed_to_common(self, name):
        """Convert a distributed dataset to a common one.

        Parameters
        ----------
        name : string
            Dataset name.

        Returns
        -------
        dset : memh5.MemDatasetCommon

        """
        if isinstance(self._data, MemGroup):
            return self._data.dataset_distributed_to_common(name)

        raise RuntimeError(
            f"Can not convert a h5py or zarr dataset {name} to distributed"
        )

    def create_group(self, name):
        """Create and return a new group."""
        path = posixpath.join(self.name, name)
        if not self.group_name_allowed(path):
            msg = f"Group name {path} not allowed."
            raise ValueError(msg)
        self._data.create_group(path)
        return self._group_class._from_storage_root(self._data, path)

    def to_memory(self):
        """Return a version of this data that lives in memory."""
        if isinstance(self._data, MemGroup):
            return self

        return self.__class__.from_file(self._data)

    def to_disk(self, filename, file_format=fileformats.HDF5, **kwargs):
        """Return a version of this data that lives on disk.

        Parameters
        ----------
        filename : str
            File name.
        file_format : `fileformats.FileFormat`
            File format to use. Default `fileformats.HDF5`.
        **kwargs
            Keyword arguments passed through to the file creating, e.g. `mode`.

        Returns
        -------
        Instance of this data object that is written to disk.
        """
        if not isinstance(self._data, MemGroup):
            msg = "This data already lives on disk.  Copying to new file anyway."
            warnings.warn(msg)
        elif self._data.distributed:
            raise NotImplementedError(
                "Cannot run to_disk on a distributed object. Try running save instead."
            )

        self.save(filename, file_format=file_format)
        return self.__class__.from_file(
            filename, ondisk=True, file_format=file_format, **kwargs
        )

    def flush(self):
        """Flush the buffers of the underlying hdf5 file if on disk."""
        if self.ondisk:
            self._data.flush()

    def copy(self, shared: list = [], shallow: bool = False) -> MemDiskGroup:
        """Return a deep copy of this class or subclass.

        Parameters
        ----------
        shared
            dataset names to share (i.e. don't deep copy)
        shallow
            True if this should be a shallow copy

        Returns
        -------
        copy
            Copy of this group
        """
        cls = self.__class__.__new__(self.__class__)
        MemDiskGroup.__init__(cls, distributed=self.distributed, comm=self.comm)
        deep_group_copy(
            self._data, cls._data, deep_copy_dsets=not shallow, shared=shared
        )

        return cls

    def __deepcopy__(self, memo, /) -> MemDiskGroup:
        """Called when copy.deepcopy is called on this class."""
        return self.copy()

    def save(
        self,
        filename,
        convert_attribute_strings=None,
        convert_dataset_strings=None,
        file_format=fileformats.HDF5,
        **kwargs,
    ):
        """Save data to hdf5/zarr file.

        Parameters
        ----------
        filename : str
            Name of the file to save into.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to a format HDF5 understands. If
            not specified, look up the name as a class attribute to find a default,
            and otherwise use `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to bytestrings before saving to
            HDF5. If not specified, look up the name as a class attribute to find a
            default, and otherwise use `False`.
        file_format : `fileformats.FileFormat`
            File format to use. Default `fileformats.HDF5`.
        **kwargs
            Keyword arguments passed through to the file creating, e.g. `mode`.
        """
        if file_format == fileformats.Zarr and not zarr_available:
            raise RuntimeError("Unable to write to zarr file, please install zarr.")

        # Get a value for the conversion parameters, looking up on the instance if
        # not supplied
        if convert_attribute_strings is None:
            convert_attribute_strings = getattr(self, "convert_attribute_strings", True)
        if convert_dataset_strings is None:
            convert_dataset_strings = getattr(self, "convert_dataset_strings", False)

        # Write out a hint as to what the class of this object is, do this by
        # inserting it into the attributes before saving out.
        if "__memh5_subclass" not in self.attrs:
            clspath = self.__class__.__module__ + "." + self.__class__.__name__

            self.attrs["__memh5_subclass"] = clspath

        if (zarr_available and isinstance(self._data, zarr.Group)) or isinstance(
            self._data, h5py.File
        ):
            with file_format.open(filename, **kwargs) as f:
                deep_group_copy(self._data, f)
        else:
            self._data.to_file(
                filename,
                convert_attribute_strings=convert_attribute_strings,
                convert_dataset_strings=convert_dataset_strings,
                file_format=file_format,
                **kwargs,
            )


class BasicCont(MemDiskGroup):
    """Basic high level data container.

    Inherits from :class:`MemDiskGroup`.

    Basic one-level data container that allows any number of datasets in the
    root group but no nesting. Data history tracking (in
    :attr:`BasicCont.history`) and array axis interpretation (in
    :attr:`BasicCont.index_map`) is also provided.

    This container is intended to be an example of how a high level container,
    with a strictly controlled data layout can be implemented by subclassing
    :class:`MemDiskGroup`.

    Parameters
    ----------
    Parameters are passed through to the base class constructor.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize new groups only if writable.
        if self._data.file.mode == "r+":
            self._data.require_group("history")
            self._data.require_group("index_map")
            self._data.require_group("reverse_map")

            if "order" not in self._data["history"].attrs:
                self._data["history"].attrs["order"] = "[]"

    @property
    def history(self):
        """Stores the analysis history for this data.

        Do not try to add a new entry by assigning to an element of this
        property. Use :meth:`~BasicCont.add_history` instead.

        Returns
        -------
        history : read only dictionary
            Each entry is a dictionary containing metadata about that stage in
            history.  There is also an 'order' entry which specifies how the
            other entries are ordered in time.
        """
        out = {}
        for name, value in self._data["history"].items():
            warnings.warn(
                f"memh5 dataset {self.name} is using a deprecated history format. Read support of "
                "files using this format will be continued for now, but you should "
                "update the instance of caput that wrote this file.",
                DeprecationWarning,
            )
            out[name] = value.attrs

        for name, value in self._data["history"].attrs.items():
            out[name] = value

        # TODO: this seems like a trememndous hack. I've changed it to a safer version of
        # eval, but this should probably be removed
        out["order"] = literal_eval(
            bytes_to_unicode(self._data["history"].attrs["order"])
        )

        return ro_dict(out)

    @property
    def index_map(self):
        """Stores representions of the axes of datasets.

        The index map contains arrays used to interpret the axes of the
        various datasets. For instance, the 'time', 'prod' and 'freq' axes of
        the visibilities are described in the index map.

        Do not try to add a new index_map by assigning to an item of this
        property. Use :meth:`~BasicCont.create_index_map` instead.

        Returns
        -------
        index_map : read only dictionary
            Entries are 1D arrays used to interpret the axes of datasets.
        """
        return ro_dict({k: v[:] for k, v in self._data["index_map"].items()})

    @property
    def index_attrs(self):
        """Exposes the attributes of each index_map entry.

        Allows the user to implement custom behaviour associated with
        the axis. Assignment to this dictionary does nothing but it does
        allow attribute values to be changed.

        Returns
        -------
        index_attrs : read-write dict
            Attribute dicts for each index_map entry
        """
        return ro_dict({k: v.attrs for k, v in self._data["index_map"].items()})

    @property
    def reverse_map(self):
        """Stores the reverse map from product index to stack index.

        Do not try to add a new reverse_map by assigning to an item of this
        property. Use :meth:`~BasicCont.create_reverse_map` instead.

        Returns
        -------
        reverse_map : read only dictionary
            Entry is a 1D arrays used to map from product index to stack index.
        """
        out = {}
        for name, value in self._data.get("reverse_map", {}).items():
            out[name] = value[:]
        return ro_dict(out)

    def group_name_allowed(self, name):
        """No groups are exposed to the user. Returns ``False``."""
        return False

    def dataset_name_allowed(self, name):
        """Datasets may only be created and accessed in the root level group.

        Returns ``True`` is *name* is a path in the root group i.e. '/dataset'.
        """
        parent_name, name = posixpath.split(name)
        return parent_name == "/"

    def create_index_map(self, axis_name, index_map):
        """Create a new index map."""
        self._data["index_map"].create_dataset(axis_name, data=index_map)

    def del_index_map(self, axis_name):
        """Delete an index map."""
        del self._data["index_map"][axis_name]

    def create_reverse_map(self, axis_name, index_map):
        """Create a new reverse map."""
        self._data["reverse_map"].create_dataset(axis_name, data=index_map)

    def del_reverse_map(self, axis_name):
        """Delete an index map."""
        del self._data["reverse_map"][axis_name]

    def add_history(self, name, history=None):
        """Create a new history entry.

        Parameters
        ----------
        name : str
            Name for history entry.
        history
            History entry (optional). Needs to be json serializable.

        Notes
        -----
        Previously only dictionaries with depth=1 were supported here. The key/value pairs of these
        where added as attributes to the history group when written to disk. Reading the old
        history format is still supported, however the history is now an attribute itself and
        dictionaries of any depth are allowed as history entries.
        """
        if name == "order":
            raise ValueError(
                '"order" is a reserved name and may not be the'
                " name of a history entry."
            )
        if history is None:
            history = {}
        order = self.history["order"]
        order = [*order, name]

        history_group = self._data["history"]
        history_group.attrs["order"] = str(order)
        history_group.attrs[name] = history

    def redistribute(self, dist_axis):
        """Redistribute parallel datasets along a specified axis.

        Parameters
        ----------
        dist_axis : int, string, or list of
            The axis can be specified by an integer index (positive or
            negative), or by a string label which must correspond to an entry in
            the `axis` attribute on the dataset. If a list is supplied, each
            entry is tried in turn, which allows different datasets to be
            redistributed along differently labelled axes.
        """
        if not isinstance(dist_axis, (list, tuple)):
            dist_axis = [dist_axis]

        stack = list(self._data._storage_root.items())

        # Crawl over the dataset tree and redistribute any matching datasets.
        # NOTE: this is done using a non-recursive stack-based tree walk, the previous
        # implementation used a recursive closure which generated a reference
        # cycle and caused the entire container to be kept alive until an
        # explicit gc run. So let this be a warning to be careful in this code.
        while stack:
            name, item = stack.pop()

            # Recurse into subgroups
            if isinstance(item, _Storage):
                stack += list(item.items())

            # Okay, we've found a distributed dataset, let's try and redistribute it
            if isinstance(item, MemDatasetDistributed):
                naxis = len(item.shape)

                for axis in dist_axis:
                    # Try processing if this is a string
                    if isinstance(axis, str):
                        if "axis" in item.attrs and axis in item.attrs["axis"]:
                            axis = list(item.attrs["axis"]).index(axis)
                        else:
                            continue

                    # Process if axis is an integer
                    elif isinstance(axis, int):
                        # Deal with negative axis index
                        if axis < 0:
                            axis = naxis + axis

                    # Check axis is within bounds
                    if axis >= naxis:
                        continue

                    # Excellent, found a matching axis, time to redistribute
                    item.redistribute(axis)
                    break

                # Note that this clause is on the FOR.
                else:
                    # If we are here we didn't find a matching axis, emit a warning
                    logger.info(
                        "Could not find axis (from %s) to distribute dataset %s over.",
                        str(dist_axis),
                        name,
                    )


# Utilities
# ---------


def attrs2dict(attrs):
    """Safely copy an h5py attributes object to a dictionary."""
    return {k: deepcopy(v) for k, v in attrs.items()}


def is_group(obj):
    """Check if the object is a Group, which includes File objects.

    In most cases, if it isn't a Group it's a Dataset, so this can be used to
    check for Datasets as well.
    """
    return hasattr(obj, "create_group")


def get_h5py_File(f, **kwargs):
    """Convenience function in order to not break old functionality."""
    return get_file(f, file_format=fileformats.HDF5, **kwargs)


def get_file(f, file_format=None, **kwargs):
    """Checks if input is a `zarr`/`h5py.File` or filename and returns the former.

    Parameters
    ----------
    f : h5py/zarr Group or filename string
        File to check.
    file_format : `fileformats.FileFormat`
            File format to use. File format will be guessed if not supplied. Default `None`.
    **kwargs : all keyword arguments
        Passed to :class:`h5py.File` constructor or `zarr.open_group`. If `f` is already an open file,
        silently ignores all keywords.

    Returns
    -------
    f : hdf5 or zarr group
    opened : bool
        Whether the a file was opened or not (i.e. was already open).
    """
    # Figure out if F is a file or a filename, and whether the file should be
    # closed.
    if is_group(f):
        return f, False

    if file_format is None:
        file_format = fileformats.guess_file_format(f)

    if file_format == fileformats.Zarr and not zarr_available:
        raise RuntimeError("Unable to open zarr file. Please install zarr.")
    try:
        f = file_format.open(f, **kwargs)
    except OSError as e:
        msg = f"Opening file {f!s} caused an error: "
        raise OSError(msg + str(e)) from e

    return f, True


def copyattrs(a1, a2, convert_strings=False):
    """Copy attributes from one h5py/zarr/memh5 attribute object to another.

    Parameters
    ----------
    a1 : h5py/zarr/memh5 object
        Attributes to copy from.
    a2 : h5py/zarr/memh5 object
        Attributes to copy into.
    convert_strings : bool, optional
        Convert string attributes (or lists/arrays of them) to ensure that they are
        unicode.
    """
    # Make sure everything is a copy.
    a1 = attrs2dict(a1)

    # When serializing dictionaries, add this in front of the string
    json_prefix = "!!_memh5_json:"

    def _map_unicode(value):
        if not convert_strings:
            return value

        # Any arrays of numpy type unicode strings must be transformed before being
        # copied into HDF5
        if isinstance(a2, h5py.AttributeManager):
            # As h5py will coerce the value to an array anyway, do it now such
            # that the following test works
            if isinstance(value, (tuple, list)):
                value = np.array(value)

            if isinstance(value, np.ndarray) and value.dtype.kind == "U":
                value = value.astype(h5py.special_dtype(vlen=str))

            return value

        # If we are copying into memh5 ensure that any string are unicode
        return bytes_to_unicode(value)

    def _map_json(value):
        # Serialize/deserialize "special" json values

        class Memh5JSONEncoder(json.JSONEncoder):
            """Properly handle some odd formats.

            - Datetimes often appear in the configs (as they are parsed by PyYAML),
              so we need to serialise them back to strings.
            - Some old data format may have numpy arrays in `history["acq"]`. We have to convert
              those to lists and decode byte objects.
            """

            def default(self, o):
                if isinstance(o, datetime.datetime):
                    return o.isoformat()
                if isinstance(o, np.number):
                    return o.data
                if isinstance(o, np.ndarray):
                    if len(o) == 1:
                        return o.tolist()[0]
                    return o.tolist()
                if isinstance(o, bytes):  # pragma: py3
                    return o.decode()

                # Let the default method raise the TypeError
                return json.JSONEncoder.default(self, o)

        if (
            isinstance(value, (dict, np.ndarray, datetime.datetime))
            and zarr_available
            and isinstance(a2, zarr.attrs.Attributes)
        ) or (
            isinstance(value, (dict, datetime.datetime))
            and isinstance(a2, h5py.AttributeManager)
        ):
            # Save to JSON converting datetimes.
            encoder = Memh5JSONEncoder()
            value = json_prefix + encoder.encode(value)
        elif isinstance(value, str) and value.startswith(json_prefix):
            # Read from JSON, keep serialised datetimes as strings
            value = json.loads(value[len(json_prefix) :])
        return value

    for key in sorted(a1):
        val = _map_unicode(a1[key])
        val = _map_json(val)
        if isinstance(val, np.generic):  # zarr can't handle numpy types
            val = val.item()
        a2[key] = val


def deep_group_copy(
    g1,
    g2,
    selections=None,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    file_format=fileformats.HDF5,
    skip_distributed=False,
    postprocess=None,
    deep_copy_dsets=False,
    shared=[],
):
    """Copy full data tree from one group to another.

    Copies from g1 to g2. An axis downselection can be specified by supplying the
    parameter 'selections'. For example to select the first two indexes in
    g1["foo"]["bar"], do

    >>> g1 = MemGroup()
    >>> foo = g1.create_group("foo")
    >>> ds = foo.create_dataset(name="bar", data=np.arange(3))
    >>> g2 = MemGroup()
    >>> deep_group_copy(g1, g2, selections={"foo/bar": slice(2)})
    >>> list(g2["foo"]["bar"])
    [0, 1]

    Parameters
    ----------
    g1 : h5py.Group or zarr.Group
        Deep copy from this group.
    g2 : h5py.Group or zarr.Group
        Deep copy to this group.
    selections : dict
        If this is not None, it should have a subset of the same hierarchical structure
        as g1, but ultimately describe axis selections for group entries as valid
        numpy indexes.
    convert_attribute_strings : bool, optional
        Convert string attributes (or lists/arrays of them) to ensure that they are
        unicode.
    convert_dataset_strings : bool, optional
        Convert strings within datasets to ensure that they are unicode.
    file_format : `fileformats.FileFormat`
        File format to use. Default `fileformats.HDF5`.
    skip_distributed : bool, optional
        If `True` skip the write for any distributed dataset, and return a list of the
        names of all datasets that were skipped. If `False` (default) throw a
        `ValueError` if any distributed datasets are encountered.
    postprocess : function, optional
        A function that takes is called on each node, with the source and destination
        entries, and can modify either.
    deep_copy_dsets : bool, optional
        Explicitly deep copy all datasets. This will only alter behaviour when copying
        from memory to memory. XXX: enabling this in places where it is not currently
        enabled could break legacy code, so be very careful
    shared : list, optional
        List of datasets to share, if `deep_copy_dsets` is True. Otherwise, no effect.

    Returns
    -------
    distributed_dataset_names : list
        Names of the distributed datasets if `skip_distributed` is True. Otherwise
        `None` is returned.
    """
    distributed_dset_names = []

    # only the case if zarr is not installed
    if file_format.module is None:
        raise RuntimeError("Can't deep_group_copy zarr file. Please install zarr.")
    to_file = isinstance(g2, file_format.module.Group)

    # Prepare a dataset for writing out, applying selections and transforming any
    # datatypes
    # Returns: dict(dtype, shape, data_to_write)
    def _prepare_dataset(dset):
        # Look for a selection for this dataset (also try without the leading "/")
        try:
            selection = selections.get(
                dset.name, selections.get(dset.name[1:], slice(None))
            )
        except AttributeError:
            selection = slice(None)

        # Check if this is a distributed dataset and figure out if we can make this work
        # out
        if to_file and isinstance(dset, MemDatasetDistributed):
            if not skip_distributed:
                raise ValueError(
                    f"Cannot write out a distributed dataset ({dset.name}) "
                    "via this method."
                )
            if selection != slice(None):
                raise ValueError(
                    "Cannot apply a slice when writing out a distributed dataset "
                    f"({dset.name}) via this method."
                )

            # If we get here, we should create the dataset, but not write out any data into it (i.e. return None)
            distributed_dset_names.append(dset.name)
            return {"dtype": dset.dtype, "shape": dset.shape, "data": None}

        # Extract the data for the selection
        data = dset[selection]

        if convert_dataset_strings:
            # Convert unicode strings back into ascii byte strings. This will break
            # if there are characters outside of the ascii range
            if to_file:
                data = ensure_bytestring(data)

            # Convert strings in an HDF5 dataset into unicode
            else:
                data = ensure_unicode(data)

        elif to_file:
            # If we shouldn't convert we at least need to ensure there aren't any
            # Unicode characters before writing
            data = check_unicode(entry)

        if not to_file:
            # reading from h5py can result in arrays with explicit endian set
            # which mpi4py cannot handle when Bcasting memh5.Group
            # needed until fixed: https://github.com/mpi4py/mpi4py/issues/177
            data = ensure_native_byteorder(data)

        dset_args = {"dtype": data.dtype, "shape": data.shape, "data": data}
        # If we're copying memory to memory we can allow distributed datasets
        if not to_file and isinstance(dset, MemDatasetDistributed):
            dset_args.update(
                {"distributed": True, "distributed_axis": dset.distributed_axis}
            )

        return dset_args

    # get compression options/chunking for this dataset
    # Returns dict of compression and chunking arguments for create_dataset
    def _prepare_compression_args(dset):
        compression = getattr(dset, "compression", None)
        compression_opts = getattr(dset, "compression_opts", None)

        if to_file:
            # massage args according to file format
            compression_kwargs = file_format.compression_kwargs(
                compression=compression,
                compression_opts=compression_opts,
                compressor=getattr(dset, "compressor", None),
            )
        else:
            # in-memory case; use HDF5 compression args format for this case
            compression_kwargs = fileformats.HDF5.compression_kwargs(
                compression=compression, compression_opts=compression_opts
            )
        compression_kwargs["chunks"] = getattr(dset, "chunks", None)

        # disable compression if not enabled for HDF5 files
        # https://github.com/chime-experiment/Pipeline/issues/33
        if (
            to_file
            and file_format == fileformats.HDF5
            and not fileformats.HDF5.compression_enabled()
            and isinstance(dset, MemDatasetDistributed)
        ):
            compression_kwargs = {}

        return compression_kwargs

    # Do a non-recursive traversal of the tree, recreating the structure and attributes,
    # and copying over any non-distributed datasets
    stack = [g1]
    while stack:
        entry = stack.pop()
        key = entry.name

        if is_group(entry):
            if key != g1.name:
                # Only create group if we are above the starting level
                g2.create_group(key)
            stack += [entry[k] for k in sorted(entry, reverse=True)]
        else:  # Is a dataset
            dset_args = _prepare_dataset(entry)
            compression_kwargs = _prepare_compression_args(entry)

            if deep_copy_dsets and key not in shared:
                # Make a deep copy of the dataset
                dset_args["data"] = deep_copy_dataset(dset_args["data"])

            g2.create_dataset(
                key,
                **dset_args,
                **compression_kwargs,
            )

        target = g2[key]
        copyattrs(entry.attrs, target.attrs, convert_strings=convert_attribute_strings)

        if postprocess:
            postprocess(entry, target)

    return distributed_dset_names if skip_distributed else None


def deep_copy_dataset(dset: Any, order: str = "A") -> Any:
    """Return a deep copy of a dataset.

    If the dataset is a ndarray or subclass, the memory
    layout can be set.

    Parameters
    ----------
    dset
        Dataset to deep copy
    order
        Controls the memory layout of the copy, for any dataset which
        takes this parameter (np.ndarray and subclasses)

    Returns
    -------
    dset_copy
        A deep copy of the dataset
    """
    if isinstance(dset, np.ndarray):
        # Set the order
        dset_copy = dset.copy(order=order)

        _o = np.dtype(object)
        _d = np.dtype(dset_copy.dtype)
        # `ndarray.copy` won't create a deep copy of the
        # array, so this has to be done if the array contains
        # some mutable python objects. This assumed that an
        # array with dtype `object` could be holding mutable
        # objects, so it will get deepcopied. This annoyingly
        # means that we're copying the array twice.
        if _d is _o:
            dset_copy = deepcopy(dset_copy)

        elif _d.names is not None:
            # This is a structured dtype, so check each field
            for name in _d.names:
                if _d.fields[name][0] is _o:
                    dset_copy[name] = deepcopy(dset_copy[name])

    else:
        # Deep copy whatever object was provided
        dset_copy = deepcopy(dset)

    return dset_copy


def format_abs_path(path):
    """Return absolute path string, formated without any extra '/'s."""
    if not posixpath.isabs(path):
        raise ValueError("Absolute path must be provided.")

    path_parts = path.split("/")
    # Strip out any empty key parts.  Takes care of '//', trailing '/', and
    # removes leading '/'.
    path_parts = [p for p in path_parts if p]

    out = "/"
    for p in path_parts:
        out = posixpath.join(out, p)
    return out


def _distributed_group_to_file(
    group,
    fname,
    mode,
    hints=True,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    file_format=None,
    serial=False,
    **kwargs,
):
    """Copy full data tree from distributed memh5 object into the destination file.

    This routine works in two stages:

    - First rank=0 copies all of the groups, attributes and non-distributed datasets
    into the target file. The distributed datasets are identified and created in this
    step, but their contents are not written. This is done by `deep_group_copy` to try
    and centralize as much of the copying code.
    - In the second step, the distributed datasets are written to disk. This is mostly
    offloaded to `MPIArray.to_file`, but some code around this needs to change depending
    on the file type, and if the data can be written in parallel.
    """
    comm = group.comm

    def apply_hints(source, dest):
        if dest.name == "/":
            dest.attrs["__memh5_distributed_file"] = True
        elif isinstance(source, MemDatasetCommon):
            dest.attrs["__memh5_distributed_dset"] = False
        elif isinstance(source, MemDatasetDistributed):
            dest.attrs["__memh5_distributed_dset"] = True
            dest.attrs["__memh5_distributed_axis"] = source.distributed_axis

    # Walk the full structure and separate out what we need to write
    if comm.rank == 0:
        with file_format.open(fname, mode) as fh:
            distributed_dataset_names = deep_group_copy(
                group,
                fh,
                convert_dataset_strings=convert_dataset_strings,
                convert_attribute_strings=convert_attribute_strings,
                skip_distributed=True,
                file_format=file_format,
                postprocess=(apply_hints if hints else None),
            )
    else:
        distributed_dataset_names = None

    distributed_dataset_names = comm.bcast(distributed_dataset_names)

    def _write_distributed_datasets(dest):
        for name in distributed_dataset_names:
            dset = group[name]
            data = check_unicode(dset)

            data.to_file(
                dest,
                name,
                chunks=dset.chunks,
                compression=dset.compression,
                compression_opts=dset.compression_opts,
                file_format=file_format,
            )
        comm.Barrier()

    # Write out the distributed parts of the file, this needs to be done slightly
    # differently depending on the actual format we want to use (and if HDF5+MPI is
    # available)
    # NOTE: need to use mode r+ as the file should already exist
    if file_format == fileformats.Zarr:
        with fileformats.ZarrProcessSynchronizer(
            f".{fname}.sync", group.comm
        ) as synchronizer, zarr.open_group(
            store=fname, mode="r+", synchronizer=synchronizer
        ) as f:
            _write_distributed_datasets(f)

    elif file_format == fileformats.HDF5:
        # Use MPI IO if possible, else revert to serialising
        if h5py.get_config().mpi:
            # Open file on all ranks
            with misc.open_h5py_mpi(fname, "r+", comm=group.comm) as f:
                if not f.is_mpi:
                    raise RuntimeError(f"Could not create file {fname!s} in MPI mode")
                _write_distributed_datasets(f)
        else:
            _write_distributed_datasets(fname)

    else:
        raise ValueError(f"Unknown format={file_format}")


def _distributed_group_from_file(
    fname: str | Path,
    comm: MPI.Comm | None = None,
    hints: bool | dict = True,
    convert_dataset_strings: bool = False,
    convert_attribute_strings: bool = True,
    file_format: type[fileformats.FileFormat] = fileformats.HDF5,
    **kwargs,
):
    """Restore full tree from an HDF5 or Zarr into a distributed memh5 object.

    A `selections=` parameter may be supplied as parts of 'kwargs'. See
    `_deep_group_copy' for a description.

    Hints may be a dictionary that can override the settings in the file itself. The
    keys should be the path to the dataset and the value a dictionary with keys
    `distributed` (boolean, required) and `axis` (integer, optional).
    """
    # Create root group
    group = MemGroup(distributed=True, comm=comm)
    comm = group.comm

    selections = kwargs.pop("selections", None)

    # Fill the hints dict if set
    hints_dict = {}
    if isinstance(hints, dict):
        hints_dict = hints
        hints = True

    # == Create some internal functions for doing the read ==
    # Copy over attributes with a broadcast from rank = 0
    def _copy_attrs_bcast(h5item, memitem, **kwargs):
        attr_dict = None
        if comm.rank == 0:
            attr_dict = dict(h5item.attrs)
        attr_dict = comm.bcast(attr_dict, root=0)
        copyattrs(attr_dict, memitem.attrs, **kwargs)

    # Function to perform a recursive clone of the tree structure
    def _copy_from_file(h5group, memgroup, selections=None):
        # Copy over attributes
        _copy_attrs_bcast(h5group, memgroup, convert_strings=convert_attribute_strings)

        # Sort items to ensure consistent order
        for key in sorted(h5group):
            item = h5group[key]
            try:
                selection = selections.get(
                    item.name, selections.get(item.name[1:], None)
                )
            except AttributeError:
                selection = None
            # If group, create the entry and the recurse into it
            if is_group(item):
                new_group = memgroup.create_group(key)
                _copy_from_file(item, new_group, selections)

            # If dataset, create dataset
            else:
                dset_hints = hints_dict.get(item.name, {})

                distributed = hints and (
                    dset_hints.get("distributed", False)
                    or item.attrs.get("__memh5_distributed_dset", False)
                )
                # Check if we are in a distributed dataset
                if distributed:
                    distributed_axis = (
                        dset_hints["axis"]
                        if "axis" in dset_hints
                        else item.attrs.get("__memh5_distributed_axis", 0)
                    )

                    # Read from file into MPIArray
                    pdata = mpiarray.MPIArray.from_file(
                        fname,
                        item.name,
                        axis=distributed_axis,
                        comm=comm,
                        sel=selection,
                        file_format=file_format,
                    )

                    # Create dataset from MPIArray
                    dset = memgroup.create_dataset(
                        key, data=pdata, distributed=True, distributed_axis=pdata.axis
                    )
                else:
                    # Read common data onto rank zero and broadcast
                    cdata = None
                    if comm.rank == 0:
                        if selection is None:
                            cdata = item[:]
                        else:
                            cdata = item[selection]

                        # Convert ascii string back to unicode if requested
                        if convert_dataset_strings:
                            cdata = ensure_unicode(cdata)

                        # only needed until fixed: https://github.com/mpi4py/mpi4py/issues/177
                        cdata = ensure_native_byteorder(cdata)

                    cdata = comm.bcast(cdata, root=0)

                    # Create dataset from array
                    dset = memgroup.create_dataset(key, data=cdata, distributed=False)

                # Copy attributes over into dataset
                _copy_attrs_bcast(item, dset, convert_strings=convert_attribute_strings)

    if file_format == fileformats.HDF5:
        # Open file on all ranks
        with misc.open_h5py_mpi(fname, "r", comm=comm) as f:
            # Start recursive file read
            _copy_from_file(f, group, selections)
    else:
        f = file_format.open(fname, "r")
        _copy_from_file(f, group, selections)

    # Final synchronisation
    comm.Barrier()

    return group


# Some extra functions for types. Should maybe move elsewhere


def bytes_to_unicode(s):
    """Ensure that a string (or collection of) are unicode.

    Any byte strings found will be transformed into unicode. Standard
    collections are processed recursively. Numpy arrays of byte strings
    are converted. Any other types are returned as is.

    Note that as HDF5 files will often contain ASCII strings which h5py
    converts to byte strings this will be needed even when fully
    transitioned to Python 3.

    Parameters
    ----------
    s : object
        Object to convert.

    Returns
    -------
    u : object
        Converted object.
    """
    if isinstance(s, bytes):
        return s.decode("utf8")

    if isinstance(s, np.ndarray) and s.dtype.kind == "S":
        return s.astype(str)

    if isinstance(s, (list, tuple, set)):
        return s.__class__(bytes_to_unicode(t) for t in s)

    if isinstance(s, dict):
        return {bytes_to_unicode(k): bytes_to_unicode(v) for k, v in s.items()}

    return s


def dtype_to_unicode(dt):
    """Convert byte strings in a dtype to unicode.

    This will attempt to parse a numpy dtype and convert strings to unicode.

    .. warning:: Custom alignment will not be preserved in these type conversions as
                 the byte and unicode string types are of different sizes.

    Parameters
    ----------
    dt : np.dtype
        Data type to convert.

    Returns
    -------
    new_dt : np.dtype
        A new datatype with the converted string type.
    """
    return _convert_dtype(dt, "|S", "<U")


def dtype_to_bytestring(dt):
    """Convert unicode strings in a dtype to byte strings.

    This will attempt to parse a numpy dtype and convert strings to bytes.

    .. warning:: Custom alignment will not be preserved in these type conversions as
                 the byte and unicode string types are of different sizes.

    Parameters
    ----------
    dt : np.dtype
        Data type to convert.

    Returns
    -------
    new_dt : np.dtype
        A new datatype with the converted string type.
    """
    return _convert_dtype(dt, "<U", "|S")


def _convert_dtype(dt, type_from, type_to):
    """Convert types in a numpy dtype to another type.

    .. warning:: Custom alignment will not be preserved in these type conversions as
                 the byte and unicode string types are of different sizes.

    Parameters
    ----------
    dt : np.dtype
        Data type to convert.
    type_from : str
        Type code (with alignment) to find.
    type_to : str
        Type code (with alignment) to convert to.

    Returns
    -------
    new_dt : np.dtype
        A new datatype with the converted string types.
    """

    def _conv(t):
        # If we get a tuple that should mean it's a type with some extended metadata, extract the
        # type and throw away the metadata
        if isinstance(t, tuple):
            t = t[0]
        return t.replace(type_from, type_to)

    # For compound types we must recurse over the full compound type structure
    def _iter_conv(x):
        items = []

        for item in x:
            name = item[0]
            type_ = item[1]

            # Recursively convert the type
            newtype = _iter_conv(type_) if isinstance(type_, list) else _conv(type_)

            items.append((name, newtype))

        return items

    # For scalar types the conversion is easy
    if not dt.names:
        return np.dtype(_conv(dt.str))
    # For compound types we need to iterate through
    return np.dtype(_iter_conv(dt.descr))


def check_byteorder(arr_byteorder):
    """Test if a native byteorder; if not, check if byteorder matches the architecture.

    Parameters
    ----------
    arr_byteorder : np.dtype.byteorder
        Array byteorder to check.

    Returns
    -------
    check_byteorder : bool
        True if the byteorder should be set to native. False, otherwise.
    """
    if arr_byteorder == "=":
        return False

    if has_matching_byteorder(arr_byteorder):
        return True

    return False


def has_matching_byteorder(arr_byteorder):
    """Test if byteorder marches the architecture.

    Parameters
    ----------
    arr_byteorder : np.dtype.byteorder
        Array byteorder to check.

    Returns
    -------
    has_matching_byteorder : bool
        True if the byteorder matches the architecture.
    """
    from sys import byteorder

    return (arr_byteorder == "<" and byteorder == "little") or (
        arr_byteorder == ">" and byteorder == "big"
    )


def has_kind(dt, kind):
    """Test if a numpy datatype has any fields of a specified type.

    Parameters
    ----------
    dt : np.dtype
        Data type to convert.
    kind : str
        Numpy type code character. e.g. "S" for bytestring and "U" for unicode.

    Returns
    -------
    has_kind : bool
        True if it contains the requested kind.
    """
    # For scalar types the conversion is easy
    if not dt.names:
        return dt.kind == kind

    # For compound types we must recurse over the full compound type structure
    def _iter_conv(x):
        for item in x:
            type_ = item[1]

            # Recursively convert the type
            if isinstance(type_, list) and _iter_conv(type_):
                return True
            if isinstance(type_, tuple) and type_[0][1] == kind:
                return True
            if type_[1] == kind:
                return True

        return False

    return _iter_conv(dt.descr)


def has_unicode(dt):
    """Test if data type contains any unicode fields.

    See `has_kind`.
    """
    return has_kind(dt, "U")


def has_bytestring(dt):
    """Test if data type contains any unicode fields.

    See `has_kind`.
    """
    return has_kind(dt, "S")


def ensure_native_byteorder(arr):
    """If architecture and arr byteorder are the same, ensure byteorder is native.

    Because of https://github.com/mpi4py/mpi4py/issues/177 mpi4py does not handle
    explicit byte order of little endian. A byteorder of native ("=" in numpy) however,
    works fine.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    arr_conv : np.ndarray
        The converted array. If no conversion was required, just returns `arr`.
    """
    if check_byteorder(arr.dtype.byteorder):
        return arr.newbyteorder("=")

    return arr


def ensure_bytestring(arr):
    """If needed convert the array to contain bytestrings not unicode.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    arr_conv : np.ndarray
        The converted array. If no conversion was required, just returns `arr`.
    """
    if has_unicode(arr.dtype):
        return arr.astype(dtype_to_bytestring(arr.dtype))

    return arr


def ensure_unicode(arr):
    """If needed convert the array to contain unicode strings not bytestrings.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    arr_conv : np.ndarray
        The converted array. If no conversion was required, just returns `arr`.
    """
    if has_bytestring(arr.dtype):
        return arr.astype(dtype_to_unicode(arr.dtype))

    return arr


def check_unicode(dset):
    """Test if dataset contains unicode so we can raise an appropriate error.

    If there is no unicode, return the data from the array.

    Parameters
    ----------
    dset : MemDataset
        Dataset to check.

    Returns
    -------
    dset :
        The converted array. If no conversion was required, just returns `arr`.
    """
    if has_unicode(dset.dtype):
        raise TypeError(
            f'Can not write dataset "{dset.name!s}" of unicode type into HDF5.'
        )

    return dset.data
