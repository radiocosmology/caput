"""
Module for making in-memory mock-ups of :mod:`h5py` objects.

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

from collections.abc import Mapping
import datetime
import warnings
import posixpath
from ast import literal_eval
import json
import logging

import numpy as np
import h5py

from . import fileformats
from . import mpiutil
from . import mpiarray
from . import misc


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
        return Mapping.__eq__(self, other) and self._dict == other._dict


class _Storage(dict):
    """Underlying container that provides storage backing for in-memory groups."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._attrs = MemAttrs()

    @property
    def attrs(self):
        """
        Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs
        """
        return self._attrs

    def __eq__(self, other):
        if not isinstance(other, _Storage):
            return False
        return dict.__eq__(self, other) and self._attrs == other._attrs


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
    """
    In memory implementation of the :class:`h5py.AttributeManager`.

    Currently just a normal dictionary.
    """

    pass


class _MemObjMixin:
    """
    Mixin represents the identity of an in-memory h5py-like object.

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
    """
    Implement the majority of the Group interface.

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
        """
        Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs
        """
        return self._get_storage().attrs

    @classmethod
    def _from_storage_root(cls, storage_root, name):
        self = super(_BaseGroup, cls).__new__(cls)
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
        elif isinstance(out, MemDataset):
            # Create back references for user facing mem datasets.
            out = out.view()
            out._storage_root = self._storage_root
            return out
        else:
            # H5py dataset.
            return out

    def __delitem__(self, name):
        """Delete item from group."""
        if name not in self:
            raise KeyError("Key %s not present." % name)
        path = posixpath.join(self.name, name)
        parent_path, name = posixpath.split(path)
        parent = self._storage_root[parent_path]
        del parent[name]

    def __len__(self):
        return len(self._get_storage())

    def __iter__(self):
        keys = list(self._get_storage().keys())
        for key in keys:
            yield key

    def require_dataset(self, name, shape, dtype, **kwargs):
        """Require a dataset to exist, create if it doesn't.

        All arguments are passed through to create_dataset.

        """
        try:
            d = self[name]
        except KeyError:
            return self.create_dataset(name, shape=shape, dtype=dtype, **kwargs)
        if is_group(d):
            msg = "Entry '%s' exists and is not a Dataset." % name
            raise TypeError(msg)
        else:
            return d

    def require_group(self, name):
        """Require a group to exist, create if it doesn't."""
        try:
            g = self[name]
        except KeyError:
            return self.create_group(name)
        if not is_group(g):
            msg = "Entry '%s' exists and is not a Group." % name
            raise TypeError(msg)
        else:
            return g


class MemGroup(_BaseGroup):
    """
    In memory implementation of the :class:`h5py.Group`.

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
        `h5py.Group` (which includes `hdf5.File` and `zarr.File` objects).

        """

        if isinstance(group, MemGroup):
            self = cls()
            deep_group_copy(group, self)
            return self
        elif isinstance(group, (str, bytes)):
            file_format = fileformats.guess_file_format(group)
            return cls.from_file(group, file_format=file_format)
        else:
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
        **_,
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
        file_format=fileformats.HDF5,
        **kwargs,
    ):
        """Create a new instance by copying from a file group.

        Any keyword arguments are passed on to the constructor for `h5py.File` or `zarr.File`.

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
        file_format : `fileformats.FileFormat`
            File format to use. Default `fileformats.HDF5`.

        Returns
        -------
        group : memh5.Group
            Root group of loaded file.
        """

        if comm is None:
            comm = mpiutil.world

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
        compression=None,
        compression_opts=None,
        **kwargs,
    ):
        """Replicate object on disk in an hdf5 file.

        Any keyword arguments are passed on to the constructor for `h5py.File`.

        Parameters
        ----------
        filename : str
            File to save into.
        hints : boolean, optional
            Whether to write hints into the file that described whether datasets
            are distributed, or not.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to a unicode type that HDF5
            understands. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to bytestrings. Default is `False`.
        """
        self.to_file(
            filename,
            mode,
            hints,
            convert_attribute_strings,
            convert_dataset_strings,
            fileformats.HDF5,
            compression,
            compression_opts,
            **kwargs,
        )

    def to_file(
        self,
        filename,
        mode="w",
        hints=True,
        convert_attribute_strings=True,
        convert_dataset_strings=False,
        file_format=fileformats.HDF5,
        compression=None,
        compression_opts=None,
        **kwargs,
    ):
        """Replicate object on disk in an hdf5 or zarr file.

        Any keyword arguments are passed on to the constructor for `h5py.File` or `zarr.File`.

        Parameters
        ----------
        filename : str
            File to save into.
        hints : boolean, optional
            Whether to write hints into the file that described whether datasets
            are distributed, or not.
        convert_attribute_strings : bool, optional
            Try and convert attribute string types to a unicode type that HDF5
            understands. Default is `True`.
        convert_dataset_strings : bool, optional
            Try and convert dataset string types to bytestrings. Default is `False`.
        file_format : `fileformats.FileFormat`
            File format to use. Default `fileformats.HDF5`.
        """
        if not self.distributed:
            with file_format.open(filename, mode, **kwargs) as f:
                deep_group_copy(
                    self,
                    f,
                    convert_attribute_strings=convert_attribute_strings,
                    convert_dataset_strings=convert_dataset_strings,
                    file_format=file_format,
                    compression=compression,
                    compression_opts=compression_opts,
                )
        elif file_format == fileformats.HDF5:
            if h5py.get_config().mpi:
                _distributed_group_to_hdf5_parallel(
                    self,
                    filename,
                    mode,
                    convert_attribute_strings=convert_attribute_strings,
                    convert_dataset_strings=convert_dataset_strings,
                )
            else:
                _distributed_group_to_hdf5_serial(
                    self,
                    filename,
                    mode,
                    convert_attribute_strings=convert_attribute_strings,
                    convert_dataset_strings=convert_dataset_strings,
                )
        else:
            _distributed_group_to_zarr(
                self,
                filename,
                mode,
                convert_attribute_strings=convert_attribute_strings,
                convert_dataset_strings=convert_dataset_strings,
                compression=compression,
                compression_opts=compression_opts,
            )

    def create_group(self, name):
        """Create a group within the storage tree."""

        path = format_abs_path(posixpath.join(self.name, name))
        try:
            self[name]
        except KeyError:
            pass
        else:
            raise ValueError("Entry %s exists." % name)

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
                raise ValueError("Entry %s exists and is not a Group." % parent_name)

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
            warnings.warn("%s is already a common dataset, no need to convert" % name)
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

    def view(self):
        cls = self.__class__
        out = cls.__new__(cls)
        super(MemDataset, out).__init__(name=self.name, storage_root=self._storage_root)
        out._attrs = self._attrs
        out._data = self._data
        out.chunks = self.chunks
        out.compression = self.compression
        out.compression_opts = self.compression_opts
        return out

    @property
    def attrs(self):
        """Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs

        """
        return self._attrs

    def resize(self):
        # h5py datasets reshape() is different from numpy reshape.
        msg = "Dataset reshaping not allowed. Perhapse make an new array view."
        raise NotImplementedError(msg)

    @property
    def shape(self):
        """
        Shape of the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def dtype(self):
        """
        numpy data type of the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def chunks(self):
        """
        Chunk shape of the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def compression(self):
        """
        Name or identifier of HDF5 compression filter for the dataset.

        Not implemented in base class.
        """
        raise NotImplementedError("Not implemented in base class.")

    @property
    def compression_opts(self):
        """
        Compression options for the dataset.

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
        return _MemObjMixin.__eq__(self, other) and self._attrs == other._attrs


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

        Returns
        -------
        dset : MemDatasetCommon
            Dataset encapsulating the numpy array.
        """

        if not isinstance(data, np.ndarray):
            raise TypeError("Object must be a numpy array (or subclass).")

        self = cls.__new__(cls)
        super(MemDatasetCommon, self).__init__(**kwargs)

        self._data = data
        self._chunks = chunks
        self._compression = compression
        self._compression_opts = compression_opts
        return self

    @property
    def comm(self):
        """Reference to the MPI communicator."""
        return None

    @property
    def common(self):
        return True

    @property
    def distributed(self):
        return False

    @property
    def data(self):
        return self._data

    @property
    def local_data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def chunks(self):
        return self._chunks

    @chunks.setter
    def chunks(self, val):
        self._chunks = val

    @property
    def compression(self):
        return self._compression

    @compression.setter
    def compression(self, val):
        self._compression = val

    @property
    def compression_opts(self):
        return self._compression_opts

    @compression_opts.setter
    def compression_opts(self, val):
        self._compression_opts = val

    def __getitem__(self, obj):
        return self._data[obj]

    def __setitem__(self, obj, val):
        self._data[obj] = val

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        # This needs to be implemented to stop craziness happening when doing
        # np.array(dset)
        return self._data.__iter__()

    def __repr__(self):
        return '<memh5 common dataset %s: shape %s, type "%s">' % (
            repr(self._name),
            repr(self.shape),
            repr(self.dtype),
        )

    def __eq__(self, other):
        if not isinstance(other, MemDatasetCommon):
            return False
        return (
            MemDataset.__eq__(self, other)
            and self._data == other._data
            and self._chunks == other._chunks
            and self._compression == other._compression
            and self._compression_opts == other._compression_opts
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
        return False

    @property
    def distributed(self):
        return True

    @property
    def data(self):
        return self._data

    @property
    def local_data(self):
        return self._data.local_array

    @property
    def shape(self):
        return self.global_shape

    @property
    def global_shape(self):
        """
        Global shape of the distributed dataset.

        The shape of the whole array that is distributed between multiple nodes.
        """
        return self._data.global_shape

    @property
    def local_shape(self):
        """
        Local shape of the distributed dataset.

        The shape of the part of the distributed array that is allocated to *this* node.
        """
        return self._data.local_shape

    @property
    def local_offset(self):
        return self._data.local_offset

    @property
    def dtype(self):
        """The numpy data type of the dataset"""
        return self._data.dtype

    @property
    def chunks(self):
        """The chunk shape of the dataset."""
        return self._chunks

    @chunks.setter
    def chunks(self, val):
        self._chunks = val

    @property
    def compression(self):
        return self._compression

    @compression.setter
    def compression(self, val):
        self._compression = val

    @property
    def compression_opts(self):
        return self._compression_opts

    @compression_opts.setter
    def compression_opts(self, val):
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
            '<memh5 distributed dataset %s: global_shape %s, dist_axis %s, type "%s">'
            % (
                repr(self._name),
                repr(self.global_shape),
                repr(self.distributed_axis),
                repr(self.dtype),
            )
        )

    def __eq__(self, other):
        if not isinstance(other, MemDatasetDistributed):
            return False
        return (
            MemDataset.__eq__(self, other)
            and self._data == other._data
            and self._chunks == other._chunks
            and self._compression == other._compression
            and self._compression_opts == other._compression_opts
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

        # Try and get a reference to the requested class (warn if we cannot find it)
        try:
            new_cls = misc.import_class(clspath)
        except (ImportError, KeyError):
            warnings.warn("Could not import memh5 subclass %s" % clspath)

        # Check that it is a subclass of MemDiskGroup
        if not issubclass(new_cls, MemDiskGroup):
            raise RuntimeError(
                "Requested type (%s) is not an subclass of memh5.MemDiskGroup."
                % clspath
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
                msg = "Access to group %s not allowed." % path
                raise KeyError(msg)
        else:
            if not self.dataset_name_allowed(path):
                msg = "Access to dataset %s not allowed." % path
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
        """
        Overwrite this method in your subclass if you want to implement downselection
        of axes (e.g. when loading a container from an HDF5 file).

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
        return None

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
        file_format=fileformats.HDF5,
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
            File format to use. Default `fileformats.HDF5`.
        **kwargs : any other arguments
            Any additional keyword arguments are passed to :class:`h5py.File`'s
            constructor if *file_* is a filename and silently ignored otherwise.
        """
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
            zarr_available = False
            if (zarr_available and isinstance(file_, zarr.Group)) or isinstance(file_, h5py.Group):
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
                kwargs.setdefault("mode", "a")
                if distributed and file_format == fileformats.Zarr:
                    lockfile = f"{file_}.sync"
                    kwargs["synchronizer"] = zarr.ProcessSynchronizer(lockfile)
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
            msg = "Dataset name %s not allowed." % path
            raise ValueError(msg)
        new_dataset = self._data.create_dataset(path, *args, **kwargs)

        return new_dataset

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
        else:
            raise RuntimeError(
                "Can not convert a h5py or zarr dataset %s to distributed" % name
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
        else:
            raise RuntimeError(
                "Can not convert a h5py or zarr dataset %s to distributed" % name
            )

    def create_group(self, name):
        """Create and return a new group."""

        path = posixpath.join(self.name, name)
        if not self.group_name_allowed(path):
            msg = "Group name %s not allowed." % path
            raise ValueError(msg)
        self._data.create_group(path)
        return self._group_class._from_storage_root(self._data, path)

    def to_memory(self):
        """Return a version of this data that lives in memory."""

        if isinstance(self._data, MemGroup):
            return self
        else:
            return self.__class__.from_file(self._data)

    def to_disk(self, filename, file_format=fileformats.HDF5, **kwargs):
        """
        Return a version of this data that lives on disk.

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

        if (zarr_available and isinstance(self._data, zarr.Group)) or isinstance(self._data, h5py.File):
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
                "memh5 dataset {} is using a deprecated history format. Read support of "
                "files using this format will be continued for now, but you should "
                "update the instance of caput that wrote this file.".format(self.name),
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

        out = {}
        for name, value in self._data["index_map"].items():
            out[name] = value[:]
        return ro_dict(out)

    @property
    def reverse_map(self):
        """Stores the reverse map from product index to stack index.

        Do not try to add a new index_map by assigning to an item of this
        property. Use :meth:`~BasicCont.create_index_map` instead.

        Returns
        -------
        reverse_map : read only dictionary
            Entry is a 1D arrays used to map from product index to stack index.

        """

        out = {}
        for name, value in self._data["reverse_map"].items():
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
        """
        Create a new history entry.

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
        order = order + [name]

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
                            axis = np.argwhere(item.attrs["axis"] == axis)[0, 0]
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

    out = {}
    for key, value in attrs.items():
        if isinstance(value, np.ndarray):
            value = value.copy()
        out[key] = value
    return out


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
    file_format : `fileformats.FileFormat`
            File format to use. File format will be guessed if not supplied. Default `None`.
    **kwargs : all keyword arguments
        Passed to :class:`h5py.File` constructor or `zarr.open_group`. If `f` is already an open file,
        silently ignores all keywords.

    Returns
    -------
    f : hdf5 group
    opened : bool
        Whether the a file was opened or not (i.e. was already open).

    """

    # Figure out if F is a file or a filename, and whether the file should be
    # closed.
    if is_group(f):
        return f, False
    else:
        if file_format is None:
            file_format = fileformats.guess_file_format(f)
        if file_format == fileformats.Zarr and not zarr_available:
            raise RuntimeError("Unable to open zarr file. Please install zarr.")
        try:
            f = file_format.open(f, **kwargs)
        except IOError as e:
            msg = "Opening file %s caused an error: " % str(f)
            raise IOError(msg + str(e)) from e
        return f, file_format == fileformats.HDF5


def copyattrs(a1, a2, convert_strings=False):
    """Copy attributes from one h5py/zarr/memh5 attribute object to another.

    Parameters
    ----------
    a1 : h5py/zarr/memh5 object
        Attributes to copy from.
    a1 : h5py/zarr/memh5 object
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
            """
            - Datetimes often appear in the configs (as they are parsed by PyYAML),
              so we need to serialise them back to strings.
            - Some old data format may have numpy arrays in `history["acq"]`. We have to convert
              those to lists and decode byte objects.
            """

            def default(self, o):
                if isinstance(o, datetime.datetime):
                    return o.isoformat()
                elif isinstance(o, np.number):
                    return o.data
                elif isinstance(o, np.ndarray):
                    if len(o) == 1:
                        return o.tolist()[0]
                    return o.tolist()
                elif isinstance(o, bytes):  # pragma: py3
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
        a2[key] = val


def deep_group_copy(
    g1,
    g2,
    selections=None,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    file_format=fileformats.HDF5,
    compression=None,
    compression_opts=None,
):
    """
    Copy full data tree from one group to another.

    Copies from g1 to g2. An axis downselection can be specified by supplying the
    parameter 'selections'. For example to select the first two indexes in g1["foo"]["bar"], do

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
    """

    copyattrs(g1.attrs, g2.attrs, convert_strings=convert_attribute_strings)

    # Sort to ensure consistent insertion order
    for key in sorted(g1):
        entry = g1[key]
        if is_group(entry):
            g2.create_group(key)
            deep_group_copy(
                entry,
                g2[key],
                selections,
                convert_dataset_strings=convert_dataset_strings,
                convert_attribute_strings=convert_attribute_strings,
                file_format=file_format,
                compression=compression,
                compression_opts=compression_opts,
            )
        else:
            # look for selection for this dataset (also try withouth the leading "/")
            try:
                selection = selections.get(
                    entry.name, selections.get(entry.name[1:], slice(None))
                )
            except AttributeError:
                selection = slice(None)

            if convert_dataset_strings:
                # Convert unicode strings back into ascii byte strings. This will break
                # if there are characters outside of the ascii range
                if isinstance(g2, file_format.module.Group):
                    data = ensure_bytestring(entry[selection])

                # Convert strings in an HDF5 dataset into unicode
                else:
                    data = ensure_unicode(entry[selection])
            elif isinstance(g2, file_format.module.Group):
                data = check_unicode(entry)
                data = data[selection]
            else:
                data = entry[selection]

            if isinstance(g2, file_format.module.Group):
                compression_entry = getattr(entry, "compression", None)
                if compression_entry is not None:
                    compression = compression_entry
                    compression_opts = getattr(entry, "compression_opts", None)
                compression_kwargs = file_format.compression_kwargs(
                    compression=compression,
                    compression_opts=compression_opts,
                    compressor=getattr(entry, "compressor", None),
                )
            else:
                # use HDF5 compression args format
                compression_kwargs = fileformats.HDF5.compression_kwargs(
                    compression=compression, compression_opts=compression_opts
                )
            g2.create_dataset(
                key,
                shape=data.shape,
                dtype=data.dtype,
                data=data,
                chunks=entry.chunks,
                **compression_kwargs,
            )
            copyattrs(
                entry.attrs, g2[key].attrs, convert_strings=convert_attribute_strings
            )


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


def _distributed_group_to_hdf5_serial(
    group,
    fname,
    mode,
    hints=True,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    **kwargs,
):
    """Private routine to copy full data tree from distributed memh5 object
    into an HDF5 file.

    This version explicitly serialises all IO.
    """

    if not group.distributed:
        raise RuntimeError(
            "This should only run on distributed datasets [%s]." % group.name
        )

    comm = group.comm

    # Create group (or file)
    if comm.rank == 0:

        # If this is the root group, create the file and copy the file level attrs
        if group.name == "/":
            with h5py.File(fname, mode, **kwargs) as f:
                copyattrs(
                    group.attrs, f.attrs, convert_strings=convert_attribute_strings
                )

                if hints:
                    f.attrs["__memh5_distributed_file"] = True

        # Create this group and copy attrs
        else:
            with h5py.File(fname, "r+", **kwargs) as f:
                g = f.create_group(group.name)
                copyattrs(
                    group.attrs, g.attrs, convert_strings=convert_attribute_strings
                )

    comm.Barrier()

    # Write out groups and distributed datasets, these operations must be done
    # collectively
    # Sort to ensure insertion order is identical
    for key in sorted(group):

        entry = group[key]

        # Groups are written out by recursing
        if is_group(entry):
            _distributed_group_to_hdf5_serial(
                entry,
                fname,
                mode,
                convert_dataset_strings=convert_dataset_strings,
                convert_attribute_strings=convert_attribute_strings,
                **kwargs,
            )

        # Write out distributed datasets (only the data, the attributes are written below)
        elif isinstance(entry, MemDatasetDistributed):

            arr = check_unicode(entry)

            arr.to_hdf5(
                fname,
                entry.name,
                chunks=entry.chunks,
                compression=entry.compression,
                compression_opts=entry.compression_opts,
            )

        comm.Barrier()

    # Write out common datasets, and the attributes on distributed datasets
    if comm.rank == 0:

        with h5py.File(fname, "r+", **kwargs) as f:

            for key, entry in group.items():

                # Write out common datasets and copy their attrs
                if isinstance(entry, MemDatasetCommon):

                    # Deal with unicode numpy datasets that aren't supported by HDF5
                    if convert_dataset_strings:
                        # Attempt to coerce to a type that HDF5 supports
                        data = ensure_bytestring(entry.data)
                    else:
                        data = check_unicode(entry)

                    dset = f.create_dataset(
                        entry.name,
                        data=data,
                        chunks=entry.chunks,
                        compression=entry.compression,
                        compression_opts=entry.compression_opts,
                    )
                    copyattrs(
                        entry.attrs,
                        dset.attrs,
                        convert_strings=convert_attribute_strings,
                    )

                    if hints:
                        dset.attrs["__memh5_distributed_dset"] = False

                # Copy the attributes over for a distributed dataset
                elif isinstance(entry, MemDatasetDistributed):

                    if entry.name not in f:
                        raise RuntimeError(
                            "Distributed dataset should already have been created."
                        )

                    copyattrs(
                        entry.attrs,
                        f[entry.name].attrs,
                        convert_strings=convert_attribute_strings,
                    )

                    if hints:
                        f[entry.name].attrs["__memh5_distributed_dset"] = True
                        f[entry.name].attrs[
                            "__memh5_distributed_axis"
                        ] = entry.distributed_axis

    comm.Barrier()


def _distributed_group_to_hdf5_parallel(
    group,
    fname,
    mode,
    hints=True,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    **_,
):
    """Private routine to copy full data tree from distributed memh5 object
    into an HDF5 file.
    This version paralellizes all IO."""

    # == Create some internal functions for doing the read ==
    # Function to perform a recursive clone of the tree structure
    def _copy_to_file(memgroup, h5group):

        # Copy over attributes
        copyattrs(
            memgroup.attrs, h5group.attrs, convert_strings=convert_attribute_strings
        )

        # Sort the items to ensure we insert in a consistent order across ranks
        for key in sorted(memgroup):

            item = memgroup[key]

            # If group, create the entry and the recurse into it
            if is_group(item):
                new_group = h5group.create_group(key)
                _copy_to_file(item, new_group)

            # If dataset, create dataset
            else:

                # Check if we are in a distributed dataset
                if isinstance(item, MemDatasetDistributed):

                    data = check_unicode(item)

                    # Write to file from MPIArray
                    data.to_hdf5(
                        h5group,
                        key,
                        chunks=item.chunks,
                        compression=item.compression,
                        compression_opts=item.compression_opts,
                    )
                    dset = h5group[key]

                    if hints:
                        dset.attrs["__memh5_distributed_dset"] = True
                        dset.attrs["__memh5_distributed_axis"] = item.distributed_axis

                # Create common dataset (collective)
                else:

                    # Convert from unicode to bytestring
                    if convert_dataset_strings:
                        data = ensure_bytestring(item.data)
                    else:
                        data = check_unicode(item)

                    dset = h5group.create_dataset(
                        key,
                        shape=data.shape,
                        dtype=data.dtype,
                        chunks=item.chunks if fileformats.HDF5.compression_enabled() else None,
                        **fileformats.HDF5.compression_kwargs(
                            item.compression, item.compression_opts
                        ),
                    )

                    # Write common data from rank 0
                    if memgroup.comm.rank == 0:
                        dset[:] = data

                    if hints:
                        dset.attrs["__memh5_distributed_dset"] = False

                # Copy attributes over into dataset
                copyattrs(
                    item.attrs, dset.attrs, convert_strings=convert_attribute_strings
                )

    # Open file on all ranks
    with misc.open_h5py_mpi(fname, mode, comm=group.comm) as f:
        if not f.is_mpi:
            raise RuntimeError("Could not create file %s in MPI mode" % fname)

        # Start recursive file write
        _copy_to_file(group, f)

        if hints:
            f.attrs["__memh5_distributed_file"] = True

    # Final synchronisation
    group.comm.Barrier()


def _distributed_group_to_zarr(
    group,
    fname,
    mode,
    hints=True,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    compression=None,
    compression_opts=None,
    **_,
):
    """Private routine to copy full data tree from distributed memh5 object into a Zarr file.

    This paralellizes all IO."""

    if not zarr_available:
        raise RuntimeError("Can't write to zarr file. Please install zarr.")

    # == Create some internal functions for doing the read ==
    # Function to perform a recursive clone of the tree structure
    def _copy_to_file(memgroup, group, compression, compression_opts):

        # Copy over attributes
        if memgroup.comm.rank == 0:
            copyattrs(
                memgroup.attrs, group.attrs, convert_strings=convert_attribute_strings
            )

        # Sort the items to ensure we insert in a consistent order across ranks
        for key in sorted(memgroup):

            item = memgroup[key]

            # If group, create the entry and the recurse into it
            if is_group(item):
                if memgroup.comm.rank == 0:
                    group.create_group(key)
                memgroup.comm.Barrier()
                _copy_to_file(item, group[key], compression, compression_opts)

            # If dataset, create dataset
            else:
                if item.compression is not None:
                    compression = item.compression
                    compression_opts = item.compression_opts

                # Check if we are in a distributed dataset
                if isinstance(item, MemDatasetDistributed):

                    data = check_unicode(item)

                    # Write to file from MPIArray
                    data.to_file(
                        group,
                        key,
                        chunks=item.chunks,
                        compression=compression,
                        compression_opts=compression_opts,
                        file_format=fileformats.Zarr,
                    )
                    dset = group[key]

                    if memgroup.comm.rank == 0 and hints:
                        dset.attrs["__memh5_distributed_dset"] = True

                # Create common dataset (collective)
                else:

                    # Convert from unicode to bytestring
                    if convert_dataset_strings:
                        data = ensure_bytestring(item.data)
                    else:
                        data = check_unicode(item)

                    # Write common data from rank 0
                    if memgroup.comm.rank == 0:
                        dset = group.create_dataset(
                            key,
                            shape=data.shape,
                            dtype=data.dtype,
                            chunks=item.chunks,
                            **fileformats.Zarr.compression_kwargs(
                                item.compression, item.compression_opts
                            ),
                        )

                        dset[:] = data

                        if hints:
                            dset.attrs["__memh5_distributed_dset"] = False

                # Copy attributes over into dataset
                if memgroup.comm.rank == 0:
                    copyattrs(
                        item.attrs,
                        dset.attrs,
                        convert_strings=convert_attribute_strings,
                    )

    # Make sure file exists
    if group.comm.rank == 0:
        zarr.open_group(store=fname, mode=mode)
    group.comm.Barrier()

    # Open file on all ranks

    with fileformats.ZarrProcessSynchronizer(
        f".{fname}.sync", group.comm
    ) as synchronizer, zarr.open_group(
        store=fname, mode="r+", synchronizer=synchronizer
    ) as f:
        # Start recursive file write
        _copy_to_file(group, f, compression, compression_opts)

        if hints and group.comm.rank == 0:
            f.attrs["__memh5_distributed_file"] = True

    # Final synchronisation
    group.comm.Barrier()


def _distributed_group_from_file(
    fname,
    comm=None,
    _=True,
    convert_dataset_strings=False,
    convert_attribute_strings=True,
    file_format=fileformats.HDF5,
    **kwargs,
):
    """
    Restore full tree from an HDF5 file into a distributed memh5 object.

    A `selections=` parameter may be supplied as parts of 'kwargs'. See
    `_deep_group_copy' for a description.
    """

    # Create root group
    group = MemGroup(distributed=True, comm=comm)
    comm = group.comm

    selections = kwargs.pop("selections", None)

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

                # Check if we are in a distributed dataset
                if ("__memh5_distributed_dset" in item.attrs) and item.attrs[
                    "__memh5_distributed_dset"
                ]:

                    distributed_axis = item.attrs.get("__memh5_distributed_axis", 0)

                    # Read from file into MPIArray
                    pdata = mpiarray.MPIArray.from_file(
                        h5group,
                        key,
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
    else:
        return np.dtype(_iter_conv(dt.descr))


def has_kind(dt, kind):
    """Test if a numpy datatype has any fields of a specified type.

    Parameters
    ----------
    dt : np.dtype
        Data type to convert.
    kind : str
        Numpy type code character. e.g. "S" for bytestring and "U" for unicode.

    Returns
    ------
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
            elif isinstance(type_, tuple) and type_[0][1] == kind:
                return True
            elif type_[1] == kind:
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
    else:
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
    else:
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
            'Can not write dataset "%s" of unicode type into HDF5.' % dset.name
        )

    return dset.data
