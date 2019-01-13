"""
Module for making in-memory mock-ups of :mod:`h5py` objects.

.. currentmodule:: caput.memh5

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


Basic Classes
=============

.. autosummary::
   :toctree: generated/

    ro_dict
    MemGroup
    MemAttrs
    MemDataset
    MemDatasetCommon
    MemDatasetDistributed


High Level Container
====================

.. autosummary::
   :toctree: generated/

    MemDiskGroup
    BasicCont


Utility Functions
=================

.. autosummary::
   :toctree: generated/

    attrs2dict
    is_group
    get_h5py_File
    copyattrs
    deep_group_copy

"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

from past.builtins import basestring
from future.utils import raise_from, text_type

import sys
import collections
import warnings
import posixpath
from ast import literal_eval

import numpy as np
import h5py

from . import mpiutil
from . import mpiarray


# Basic Classes
# -------------



class ro_dict(collections.Mapping):
    """A dict that is read-only to the user.

    This class isn't strictly read-only but it cannot be modified through the
    traditional dict interface. This prevents the user from mistaking this for
    a normal dictionary.

    Provides the same interface for reading as the builtin python
    :class:`dict`s but no methods for writing.

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


class _Storage(dict):
    """Underlying container that provides storage backing for in-memory groups.

    """

    def __init__(self, **kwargs):
        super(_Storage, self).__init__(**kwargs)
        self._attrs = MemAttrs()

    @property
    def attrs(self):
        return self._attrs


class _StorageRoot(_Storage):
    """Root level of the storage tree.

    """

    def __init__(self, distributed=False, comm=None):
        super(_StorageRoot, self).__init__()

        if comm is None:
            comm = mpiutil.world

        self._comm = comm

        if self._comm is None:
            if distributed:
                warnings.warn('Cannot not be in distributed mode when there is no MPI communicator!!')
            self._distributed = False
        else:
            self._distributed = distributed

    @property
    def comm(self):
        return self._comm

    @property
    def distributed(self):
        return self._distributed

    def __getitem__(self, key):
        """Implements Hierarchical path lookup."""

        if '/' not in key:
            return super(_StorageRoot, self).__getitem__(key)

        # Format and split the path.
        key = format_abs_path(key)
        if key == '/':
            return self

        path_parts = key.split('/')[1:]

        # Crawl the path.
        out = self
        for part in path_parts:
            out = out[part]
        return out


class MemAttrs(dict):
    """In memory implementation of the :class:`h5py.AttributeManager`.

    Currently just a normal dictionary.

    """

    pass


class _MemObjMixin(object):
    """Mixin represents the identity of an in-memory h5py-like object.

    Implement a few attributes that all memh5 objects have, such as `parent`,
    and `file`.

    """

    @property
    def _group_class(self):
        return None

    # Here I have to implement __new__ not __init__ since MemDiskGroup
    # implements new and messes with parameters.
    def __init__(self, storage_root=None, name=''):
        super(_MemObjMixin, self).__init__()
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
        parent_name, myname = posixpath.split(self.name)
        return self._group_class._from_storage_root(self._storage_root, parent_name)

    @property
    def file(self):
        """Not a file at all but the top most :class:`MemGroup` of the tree."""
        return self._group_class._from_storage_root(self._storage_root, '/')

    def __eq__(self, other):
        if hasattr(other, '_storage_root') and hasattr(other, 'name'):
            return ((self._storage_root is other._storage_root)
                    and (self.name == other.name))
        return False

    def __neq__(self, other):
        return not self.__eq__(other)


class _BaseGroup(_MemObjMixin, collections.Mapping):
    """Implement the majority of the Group interface.

    Subclasses must setup the underlying storage in thier constructors, as well
    as implement `create_group` and `create_dataset`.

    """

    @property
    def _group_class(self):
        return self.__class__

    @property
    def comm(self):
        """Reference to the MPI communicator.
        """
        return getattr(self._storage_root, 'comm', None)

    @property
    def distributed(self):
        return getattr(self._storage_root, 'distributed', False)

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
        else:
            # A dataset
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

    Attributes
    ----------
    attrs
    name
    parent
    file

    Methods
    -------
    __getitem__
    from_group
    from_hdf5
    to_hdf5
    create_group
    require_group
    create_dataset
    require_dataset

    """

    def __init__(self, distributed=False, comm=None):
        # Default constructor is only used to create the root group.
        storage_root = _StorageRoot(distributed=distributed, comm=comm)
        name = '/'
        super(MemGroup, self).__init__(storage_root, name)

    @property
    def mode(self):
        """String indicating if group is readonly ("r") or read-write ("r+").

        :class:`MemGroup`s are always read-write.

        """
        return 'r+'


    @classmethod
    def from_group(cls, group):
        """Create a new instance by deep copying an existing group.

        Agnostic as to whether the group to be copyed is a `MemGroup` or an
        `h5py.Group` (which includes `hdf5.File` objects).

        """

        if isinstance(group, MemGroup):
            self = cls()
            deep_group_copy(group, self)
            return self
        else:
            return cls.from_hdf5(group)

    @classmethod
    def from_hdf5(cls, filename, distributed=False, hints=True, comm=None, **kwargs):
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

        Returns
        -------
        group : memh5.Group
            Root group of loaded file.
        """

        if comm is None:
            comm = mpiutil.world

        if comm is None:
            if distributed:
                warnings.warn('Cannot load file in distributed mode when there is no MPI communicator!!')
            distributed = False

        if not distributed or not hints:
            with h5py.File(filename, **kwargs) as f:
                self = cls(distributed=distributed, comm=comm)
                deep_group_copy(f, self)
        else:
            self = _distributed_group_from_hdf5(filename, comm=comm, hints=hints)

        return self

    def to_hdf5(self, filename, hints=True, **kwargs):
        """Replicate object on disk in an hdf5 file.

        Any keyword arguments are passed on to the constructor for `h5py.File`.

        Parameters
        ----------
        filename : str
            File to save into.
        hints : boolean, optional
            Whether to write hints into the file that described whether datasets
            are distributed, or not.
        """

        if not self.distributed:
            with h5py.File(filename, **kwargs) as f:
                deep_group_copy(self, f)
        else:
            _distributed_group_to_hdf5(self, filename, **kwargs)

    def create_group(self, name):
        """Create a group within the storage tree."""

        path = format_abs_path(posixpath.join(self.name, name))
        try:
            self[name]
        except KeyError:
            pass
        else:
            raise ValueError('Entry %s exists.' % name)

        # If distributed, synchronise to ensure that we create group collectively
        if self.distributed:
            self.comm.Barrier()

        parent_name = '/'
        path_parts = path.split('/')
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
                raise ValueError('Entry %s exists and is not a Group.'
                                 % parent_name)

        # Underlying storage has been created. Return the group object.
        return self[name]


    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       distributed=False, distributed_axis=None, **kwargs):
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
                warnings.warn('Cannot create distributed dataset when there is no MPI communicator!!')
            distributed = False

        if kwargs:
            msg = ("No extra keyword arguments accepted, this is not an hdf5"
                   " object but a memory object mocked up to look like one.")
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
            raise ValueError('shape must be provided.')
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
            raise RuntimeError('Cannot create a distributed dataset in a non-distributed group.')

        # If data is set (and consistent with shape/type), initialise the numpy array from it.
        if (data is not None and shape == data.shape
           and dtype is data.dtype and hasattr(data, 'view')):

            # Create parallel array if requested
            if distributed:

                # Ensure we are creating from an MPIArray
                if not isinstance(data, mpiarray.MPIArray):
                    raise TypeError('Can only create distributed dataset from MPIArray.')

                # Ensure that we are distributing over the same communicator
                if data.comm != self.comm:
                    raise RuntimeError('MPI communicator of array must match that of memh5 group.')

                # If the distributed_axis is specified ensure the data is distributed along it.
                if distributed_axis is not None:
                    data = data.redistribute(axis=distributed_axis)

                # Create distributed dataset
                new_dataset = MemDatasetDistributed.from_mpi_array(data)
            else:
                # Create common dataset
                new_dataset = MemDatasetCommon.from_numpy_array(data)

        # Otherwise create an empty array and copy into it (if needed)
        else:
            # Just copy the data.
            if distributed:

                # Ensure that distributed_axis is set.
                if distributed_axis is None:
                    raise RuntimeError('Distributed axis must be specified when creating dataset.')

                new_dataset = MemDatasetDistributed(shape=shape, dtype=dtype,
                                                    axis=distributed_axis,
                                                    comm=self.comm)
            else:
                new_dataset = MemDatasetCommon(shape=shape, dtype=dtype)

            if data is not None:
                new_dataset[:] = data[:]

        # Add new dataset to group
        parent_storage[name] = new_dataset

        # Set the properties of the new dataset
        new_dataset._name = posixpath.join(parent_name, name)
        new_dataset._storage_root = self._storage_root
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

        dset = self[name]

        if dset.distributed:
            warnings.warn('%s is already a distributed dataset, redistribute it along the required axis %d' % (name, distributed_axis))
            dset.redistribute(distributed_axis)
            return dset

        dset_shape = dset.shape
        dset_type = dset.dtype
        dist_len = dset_shape[distributed_axis]
        ld, sd, ed = mpiutil.split_local(dist_len, comm=self.comm)
        md = mpiarray.MPIArray(dset_shape, axis=distributed_axis, comm=self.comm, dtype=dset_type)
        md.local_array[:] = dset[sd:ed].copy()
        attr_dict = {} # temporarily save attrs of this dataset
        copyattrs(dset.attrs, attr_dict)
        del dset
        new_dset = self.create_dataset(name, shape=dset_shape, dtype=dset_type, data=md, distributed=True, distributed_axis=distributed_axis)
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
            warnings.warn('%s is already a common dataset, no need to convert' % name)
            return dset

        dset_shape = dset.shape
        dset_type = dset.dtype
        global_array = np.zeros(dset_shape, dtype=dset_type)
        local_start = dset.local_offset
        nproc = 1 if self.comm is None else self.comm.size
        # gather local distributed dataset to a global array for all procs
        for rank in range(nproc):
            mpiutil.gather_local(global_array, dset.local_data, local_start, root=rank, comm=self.comm)
        attr_dict = {} # temporarily save attrs of this dataset
        copyattrs(dset.attrs, attr_dict)
        del dset
        new_dset = self.create_dataset(name, data=global_array, shape=dset_shape, dtype=dset_type)
        copyattrs(attr_dict, new_dset.attrs)

        return new_dset



class MemDataset(_MemObjMixin):
    """Base class for an in memory implementation of :class:`h5py.Dataset`.

    This is only an abstract base class. Use :class:`MemDatasetCommon` or
    :class:`MemDatasetDistributed`.

    Attributes
    ----------
    attrs
    name
    parent
    file

    """

    def __init__(self, **kwargs):
        super(MemDataset, self).__init__(**kwargs)
        self._attrs = MemAttrs()

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
        raise NotImplementedError("Not implmemented in base class.")

    @property
    def dtype(self):
        raise NotImplementedError("Not implmemented in base class.")

    def __getitem__(self, obj):
        raise NotImplementedError("Not implmemented in base class.")

    def __setitem__(self, obj, val):
        raise NotImplementedError("Not implmemented in base class.")

    def __len__(self):
        raise NotImplementedError("Not implmemented in base class.")


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

    Attributes
    ----------
    common
    distributed
    data
    local_data
    shape
    dtype

    Methods
    -------
    from_numpy_array

    """

    def __init__(self, shape, dtype):
        super(MemDatasetCommon, self).__init__()

        self._data = np.zeros(shape, dtype)

    @classmethod
    def from_numpy_array(cls, data):
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

        dset = cls.__new__(cls)
        super(MemDatasetCommon, dset).__init__()

        dset._data = data
        return dset

    @property
    def comm(self):
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
        return "<memh5 common dataset %s: shape %s, type \"%s\">" % (repr(self._name), repr(self.shape), repr(self.dtype))


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

    Attributes
    ----------
    common
    distributed
    data
    local_data
    shape
    global_shape
    local_shape
    local_offset
    dtype
    comm
    distributed_axis

    """

    def __init__(self, shape, dtype, axis=0, comm=None):
        super(MemDatasetDistributed, self).__init__()

        self._data = mpiarray.MPIArray(shape, axis=axis, comm=comm, dtype=dtype)

    @classmethod
    def from_mpi_array(cls, data):
        dset = cls.__new__(cls)
        MemDataset.__init__(dset)

        if not isinstance(data, mpiarray.MPIArray):
            raise TypeError("Object must be a numpy array (or subclass).")

        dset._data = data
        return dset

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
        return self._data.global_shape

    @property
    def local_shape(self):
        return self._data.local_shape

    @property
    def local_offset(self):
        return self._data.local_offset

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def distributed_axis(self):
        return self._data.axis

    @property
    def comm(self):
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
        return ("<memh5 distributed dataset %s: global_shape %s, dist_axis %s, type \"%s\">"
                % (repr(self._name), repr(self.global_shape), repr(self.distributed_axis), repr(self.dtype)))


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


    Attributes
    ----------
    attrs
    name
    parent
    file
    ondisk

    Methods
    -------
    __getitem__
    __delitem__
    from_file
    dataset_name_allowed
    group_name_allowed
    create_dataset
    require_dataset
    create_group
    require_group
    to_memory
    to_disk
    flush
    close
    save

    """

    def __init__(self, data_group=None, distributed=False, comm=None):

        toclose = False

        if comm is None:
            comm = mpiutil.world

        if comm is None:
            if distributed:
                warnings.warn('Cannot create distributed MemDiskGroup when there is no MPI communicator!!')
            distributed = False
        else:
            distributed = distributed

        # If data group is not set, initialise a new MemGroup
        if data_group is None:
            data_group = MemGroup(distributed=distributed, comm=comm)
        # If it is a MemDiskGroup then initialise a shallow copy
        elif isinstance(data_group, MemDiskGroup):
            data_group = data_group._storage_root
        # Otherwise, presume it is an HDF5 Group-like object (which includes
        # MemGroup and h5py.Group).
        else:
            data_group, toclose = get_h5py_File(data_group)

        if distributed and isinstance(data_group, h5py.Group):
            raise ValueError('Distributed MemDiskGroup cannot be created around h5py objects.')
        # Check the distribution settings
        elif distributed:
            # Check parallel distribution is the same
            if not data_group.distributed:
                raise ValueError('Cannot create MemDiskGroup with different distributed setting to MemGroup to wrap.')
            # Check parallel communicator is the same
            if comm and comm != data_group.comm:
                raise ValueError('Cannot create MemDiskGroup with different MPI communicator to MemGroup to wrap.')

        self._toclose = toclose
        super(MemDiskGroup, self).__init__(storage_root=data_group, name=data_group.name)

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

        # Look for a hint as to the sub class we should return, this should be
        # in the attributes of the root.
        new_cls = cls
        if detect_subclass and '__memh5_subclass' in data_group.attrs:
            from .pipeline import _import_class

            clspath = data_group.attrs['__memh5_subclass']

            # Try and get a reference to the requested class (warn if we cannot find it)
            try:
                new_cls = _import_class(clspath)
            except (ImportError, KeyError):
                warnings.warn('Could not import memh5 subclass %s' % clspath)

            # Check that it is a subclass of MemDiskGroup
            if not issubclass(cls, MemDiskGroup):
                raise RuntimeError('Requested type (%s) is not an instance of memh5.MemDiskGroup.' % clspath)

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

    def __del__(self):
        """Closes file if on disk if file was opened on initialization."""
        if self.ondisk and hasattr(self, '_toclose') and self._toclose:
            self._storage_root.close()

    def __getitem__(self, name):
        """Retrieve an object.

        The *name* may be a relative or absolute path

        """

        value = super(MemDiskGroup, self).__getitem__(name)
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
        for key in self:
            n += 1
        return n

    def __iter__(self):
        for key in super(MemDiskGroup, self).__iter__():
            try:
                value = self[key]
            except KeyError:
                # This key name is not allowed (see __getitem__)
                continue
            yield key

    @property
    def ondisk(self):
        """Whether the data is stored on disk as opposed to in memory."""
        return hasattr(self, '_storage_root') and isinstance(self._storage_root, h5py.File)

    # For creating new instances. #

    @classmethod
    def from_file(cls, file_, ondisk=False, distributed=False, comm=None,
                  detect_subclass=True, **kwargs):
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
        **kwargs : any other arguments
            Any additional keyword arguments are passed to :class:`h5py.File`'s
            constructor if *file_* is a filename and silently ignored otherwise.
        """

        if not ondisk:
            if isinstance(file_, h5py.Group):
                file_ = file_.filename

            data = MemGroup.from_hdf5(file_, distributed=distributed, comm=comm, mode='r', **kwargs)
            toclose = False
        else:
            # Again, a compatibility hack
            if is_group(file_):
                data = file_
                toclose = False
            else:
                data = h5py.File(file_, **kwargs)
                toclose = True

        # Here we explicitly avoid calling __init__ on any derived class. Like
        # with a pickle we want to restore the saved state only.
        self = cls.from_group(data_group=data)

        # ... skip the class initialisation, and use a special method
        self._finish_setup()

        self._toclose = toclose
        return self

    # Methods for manipulating and building the class. #

    def group_name_allowed(self, name):
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

    def dataset_name_allowed(self, name):
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
            raise RuntimeError('Can not convert a h5py dataset %s to distributed' % name)

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
            raise RuntimeError('Can not convert a h5py dataset %s to distributed' % name)

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

    def to_disk(self, filename, **kwargs):
        """Return a version of this data that lives on disk."""

        if not isinstance(self._data, MemGroup):
            msg = ("This data already lives on disk.  Copying to new file"
                   " anyway.")
            warnings.warn(msg)
        elif self._data.distributed:
            raise NotImplementedError("Cannot run to_disk on a distributed object. Try running save instead.")

        self.save(filename)
        return self.__class__.from_file(filename, ondisk=True, **kwargs)

    def close(self):
        """Close underlying hdf5 file if on disk."""

        if self.ondisk:
            self._data.close()

    def flush(self):
        """Flush the buffers of the underlying hdf5 file if on disk."""

        if self.ondisk:
            self._data.flush()

    def save(self, filename, **kwargs):
        """Save data to hdf5 file."""

        # Write out a hint as to what the class of this object is, do this by
        # inserting it into the attributes before saving out.
        if '__memh5_subclass' not in self.attrs:
            clspath = self.__class__.__module__ + '.' + self.__class__.__name__

            self.attrs['__memh5_subclass'] = clspath

        if isinstance(self._data, h5py.File):
            with h5py.File(filename, **kwargs) as f:
                deep_group_copy(self._data, f)
        else:
            self._data.to_hdf5(filename, **kwargs)


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

    Attributes
    ----------
    index_map
    history

    Methods
    -------
    group_name_allowed
    dataset_name_allowed
    create_index_map
    del_index_map
    add_history
    redistribute

    """

    def __init__(self, *args, **kwargs):
        super(BasicCont,self).__init__(*args, **kwargs)
        # Initialize new groups only if writable.
        if self._data.file.mode == 'r+':
            self._data.require_group(u'history')
            self._data.require_group(u'index_map')
            if 'order' not in self._data['history'].attrs:
                self._data['history'].attrs[u'order'] = '[]'

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
        for name, value in self._data['history'].items():
            out[name] = value.attrs

        # TODO: this seems like a trememndous hack. I've changed it to a safer version of
        # eval, but this should probably be removed
        out[u'order'] = literal_eval(bytes_to_unicode(self._data['history'].attrs['order']))

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
        for name, value in self._data['index_map'].items():
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
        return True if parent_name == '/' else False

    def create_index_map(self, axis_name, index_map):
        """Create a new index map.

        """

        self._data['index_map'].create_dataset(axis_name, data=index_map)

    def del_index_map(self, axis_name):
        """Delete an index map."""
        del self._data['index_map'][axis_name]

    def add_history(self, name, history=None):
        """Create a new history entry."""

        if name == 'order':
            raise ValueError('"order" is a reserved name and may not be the'
                             ' name of a history entry.')
        if history is None:
            history = {}
        order = self.history['order']
        order = order + [name]
        history_group = self._data["history"]
        history_group.attrs[u'order'] = text_type(order)
        history_group.create_group(name)
        for key, value in history.items():
            history_group[name].attrs[key] = value

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

        # Worker routine to crawl the tree and redistribute any parallel datasets
        def _tree_crawl(group):

            for name, item in group.items():

                # Recurse into subgroups
                if isinstance(item, MemGroup):
                    _tree_crawl(item)

                # Okay, we've found a distributed dataset, let's try and redistribute it
                if isinstance(item, MemDatasetDistributed):

                    naxis = len(item.shape)

                    for axis in dist_axis:

                        # Try processing if this is a string
                        if isinstance(axis, basestring):
                            if 'axis' in item.attrs and axis in item.attrs['axis']:
                                axis = np.argwhere(item.attrs['axis'] == axis)[0, 0]
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
                        if group.comm.rank == 0:
                            warnings.warn(('Could not find an axis (out of %s)'
                                            + 'to distributed dataset %s over.') % (str(dist_axis), name))

        _tree_crawl(self._data)


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

    return hasattr(obj, 'create_group')


def get_h5py_File(f, **kwargs):
    """Checks if input is an `h5py.File` or filename and returns the former.

    Parameters
    ----------
    f : h5py Group or filename string
    **kwargs : all keyword arguments
        Passed to :class:`h5py.File` constructor. If `f` is already an open file,
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
        opened = False
        #if kwargs:
        #    msg = "Got some keyword arguments but File is alrady open."
        #    warnings.warn(msg)
    else:
        opened = True
        try:
            f = h5py.File(f, **kwargs)
        except IOError as e:
            msg = "Opening file %s caused an error: " % str(f)
            # TODO: Py3 exception chaining
            raise_from(IOError(msg + str(e)), e)
    return f, opened


def copyattrs(a1, a2):
    # Make sure everything is a copy.
    a1 = attrs2dict(a1)

    def _map_attr(value):

        # Any arrays of numpy type unicode strings must be transformed before being copied into HDF5
        if isinstance(a2, h5py.AttributeManager):

            # As h5py will coerce the value to an array anyway, do it now such
            # that the following test works
            if isinstance(value, (tuple, list)):
                value = np.array(value)

            if isinstance(value, np.ndarray) and value.dtype.kind == 'U':
                return value.astype(h5py.special_dtype(vlen=text_type))
            else:
                return value

        # If we are copying into memh5 ensure that any string are unicode
        return bytes_to_unicode(value)

    for key, value in a1.items():
        a2[key] = _map_attr(value)


def deep_group_copy(g1, g2):
    """Copy full data tree from one group to another."""

    copyattrs(g1.attrs, g2.attrs)
    for key, entry in g1.items():
        if is_group(entry):
            g2.create_group(key)
            deep_group_copy(entry, g2[key])
        else:
            g2.create_dataset(key, shape=entry.shape, dtype=entry.dtype,
                    data=entry)
            copyattrs(entry.attrs, g2[key].attrs)


def format_abs_path(path):
    """Return absolute path string, formated without any extra '/'s."""
    if not posixpath.isabs(path):
        raise ValueError("Absolute path must be provided.")

    path_parts = path.split('/')
    # Strip out any empty key parts.  Takes care of '//', trailing '/', and
    # removes leading '/'.
    path_parts = [p for p in path_parts if p]

    out = '/'
    for p in path_parts:
        out = posixpath.join(out, p)
    return out


def _distributed_group_to_hdf5(group, fname, hints=True, **kwargs):
    """Private routine to copy full data tree from distributed memh5 object into an
    HDF5 file."""

    if not group.distributed:
        raise RuntimeError('This should only run on distributed datasets [%s].' % group.name)

    comm = group.comm

    # Create a copy of the kwargs with no mode argument so that we can override it
    kwargs_nomode = kwargs.copy()
    if 'mode' in kwargs:
        del kwargs_nomode['mode']

    # Create group (or file)
    if comm.rank == 0:

        # If this is the root group, create the file and copy the file level
        # attrs
        if group.name == '/':
            with h5py.File(fname, 'w', **kwargs) as f:
                copyattrs(group.attrs, f.attrs)

                if hints:
                    f.attrs['__memh5_distributed_file'] = True

        # Create this group and copy attrs
        else:
            with h5py.File(fname, 'r+', **kwargs) as f:
                g = f.create_group(group.name)
                copyattrs(group.attrs, g.attrs)

    comm.Barrier()

    # Write out groups and distributed datasets, these operations must be done collectively
    for key, entry in group.items():

        # Groups are written out by recursing
        if is_group(entry):
            _distributed_group_to_hdf5(entry, fname, **kwargs)

        # Write out distributed datasets (only the data, the attributes are written below)
        elif isinstance(entry, MemDatasetDistributed):
            arr = entry._data

            arr.to_hdf5(fname, entry.name)

        comm.Barrier()

    # Write out common datasets, and the attributes on distributed datasets
    if comm.rank == 0:

        with h5py.File(fname, 'r+', **kwargs_nomode) as f:

            for key, entry in group.items():

                # Write out common datasets and copy their attrs
                if isinstance(entry, MemDatasetCommon):
                    dset = f.create_dataset(entry.name, data=entry._data)
                    copyattrs(entry.attrs, dset.attrs)

                    if hints:
                        dset.attrs['__memh5_distributed_dset'] = False

                # Copy the attributes over for a distributed dataset
                elif isinstance(entry, MemDatasetDistributed):

                    if entry.name not in f:
                        raise RuntimeError('Distributed dataset should already have been created.')

                    copyattrs(entry.attrs, f[entry.name].attrs)

                    if hints:
                        f[entry.name].attrs['__memh5_distributed_dset'] = True

    comm.Barrier()


def _distributed_group_from_hdf5(fname, comm=None, hints=True, **kwargs):
    """Private routine to restore full tree from an HDF5 file into a distributed memh5 object."""

    # Create root group
    group = MemGroup(distributed=True, comm=comm)
    comm = group.comm

    # == Create some internal functions for doing the read ==
    # Copy over attributes with a broadcast from rank = 0
    def _copy_attrs_bcast(h5item, memitem):
        attr_dict = None
        if comm.rank == 0:
            attr_dict = { k: v for k, v in h5item.attrs.items() }
        attr_dict = comm.bcast(attr_dict, root=0)
        copyattrs(attr_dict, memitem.attrs)

    # Function to perform a recursive clone of the tree structure
    def _copy_from_file(h5group, memgroup):

        # Copy over attributes
        _copy_attrs_bcast(h5group, memgroup)

        for key, item in h5group.items():

            # If group, create the entry and the recurse into it
            if is_group(item):
                new_group = memgroup.create_group(key)
                _copy_from_file(item, new_group)

            # If dataset, create dataset
            else:

                # Check if we are in a distributed dataset
                if ('__memh5_distributed_dset' in item.attrs) and item.attrs['__memh5_distributed_dset']:

                    # Read from file into MPIArray
                    pdata = mpiarray.MPIArray.from_hdf5(f, key, comm=comm)

                    # Create dataset from MPIArray
                    dset = memgroup.create_dataset(key, data=pdata, distributed=True)
                else:

                    # Read common data onto rank zero and broadcast
                    cdata = None
                    if comm.rank == 0:
                        cdata = item[:]
                    cdata = comm.bcast(cdata, root=0)

                    # Create dataset from array
                    dset = memgroup.create_dataset(key, data=cdata, distributed=False)

                # Copy attributes over into dataset
                _copy_attrs_bcast(item, dset)

    # Open file on all ranks
    with h5py.File(fname, 'r') as f:

        # Start recursive file read
        _copy_from_file(f, group)

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
        return s.decode('utf8')

    if isinstance(s, np.ndarray) and s.dtype.kind == 'S':
        return s.astype(text_type)

    if isinstance(s, (list, tuple, set)):
        return s.__class__(bytes_to_unicode(t) for t in s)

    if isinstance(s, dict):
        return {bytes_to_unicode(k): bytes_to_unicode(v) for k, v in s.items()}

    return s

