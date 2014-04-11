"""
Module for making in-memory mock-ups of :mod:`h5py` objects.

.. currentmodule:: caput.memh5

It is sometimes usefull to have a consistent API for data that is independent
of whether that data lives on disk or in memory. :mod:`h5py` provides this to a
certain extent, having :class:`Dataset`` objects that act very much like
:mod:`numpy` arrays. mod:`memh5` extends this, providing an in-memory
containers, analogous to :class:`h5py.Group` and :class:`h5py.Attribute` and
:class:`h5py.Dataset` objects.

In addition to these basic classes that copy the :mod:`h5py` API, A higher
level data container is provided that utilizes these classes along with the
:mod:`h5py` to provide data that is transparently stored either in memory or on
disk.


Basic Classes
=============

.. autosummary::
   :toctree: generated/

    ro_dict
    MemGroup
    MemAttrs
    MemDataset


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

import sys
import collections
import warnings

import numpy as np
import h5py


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


class MemGroup(ro_dict):
    """In memory implementation of the :class:`h5py.Group`.

    This class doubles as the memory implementation of :class:`h5py.File`,
    object, since the distinction between a file and a group for in-memory data
    is moot.

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

    def __init__(self):
        ro_dict.__init__(self)
        self._attrs = MemAttrs()
        # Set the following assuming we are the root group. If not, the method
        # that created us will reset.
        self._root = self
        self._parent = self
        self._name = ''

    def __getitem__(self, key):
        """Implement '/' for accessing nested groups."""
        if not key:
            return self
        # If this is an absolute path, index from the root group.
        if key[0] == '/':
            return self._root[key[1:]]
        key_parts = key.split('/')
        # Strip out any empty key parts.  Takes care of '//' and trailing '/'.
        key_parts = [p for p in key_parts if p]
        if len(key_parts) == 1:
            return ro_dict.__getitem__(self, key_parts[0])
        else:
            # Enter the first level and call __getitem__ recursively.
            return self[key_parts[0]]['/'.join(key_parts[1:])]

    @property
    def attrs(self):
        """Attributes attached to this object.

        Returns
        -------
        attrs : MemAttrs

        """

        return self._attrs

    @property
    def parent(self):
        """Parent :class:`MemGroup` that contains this group."""
        return self._parent

    @property
    def name(self):
        """String giving the full path to this group."""
        if self.parent is self._root:
            return '/' + self._name
        else:
            return self.parent.name + '/' + self._name

    @property
    def file(self):
        """Not a file at all but the top most :class:`MemGroup` of the tree."""
        return self._root

    @classmethod
    def from_group(cls, group):
        """Create a new instance by deep copying an existing group.

        Agnostic as to whether the group to be copyed is a `MemGroup` or an
        `h5py.Group` (which includes `hdf5.File` objects). 

        """

        self = cls()
        deep_group_copy(group, self)
        return self

    @classmethod
    def from_hdf5(cls, f, **kwargs):
        """Create a new instance by copying from an hdf5 group.

        This is the same as `from_group` except that an hdf5 filename is
        accepted.  Any keyword arguments are passed on to the constructor for
        `h5py.File`.

        """

        f, to_close = get_h5py_File(f, **kwargs)
        self = cls.from_group(f)
        if to_close:
            f.close()
        return self

    def to_hdf5(self, f, **kwargs):
        """Replicate object on disk in an hdf5 file.

        Any keyword arguments are passed on to the constructor for `h5py.File`.

        """

        f, opened = get_h5py_File(f, **kwargs)
        deep_group_copy(self, f)
        return f

    def create_group(self, key):
        # Corner case if empty key.
        if not key:
            msg = "Empty group names not allowed."
            raise ValueError(msg)
        if not '/' in key:
            # Create group directly.
            try:
                self[key]
            except KeyError:
                out = MemGroup()
                out._root = self._root
                out._parent = self
                out._name = key
                self._dict[key] = out
                return out
            else:
                msg = "Item '%s' already exists." % key
                raise ValueError(msg)
        else:
            # Recursively create groups.
            key_parts = key.split('/')
            # strip off trailing '/' if present.
            if not key_parts[-1]:
                key_parts = key_parts[:-1]
            # Corner case of '/group_name':
            if len(key_parts) == 2 and not key_parts[0]:
                g = self._root
            else:
                g = self.require_group('/'.join(key_parts[:-1]))
            return g.create_group(key_parts[-1])

    def require_group(self, key):
        try:
            g = self[key]
        except KeyError:
            return self.create_group(key)
        if not isinstance(g, MemGroup):
            msg = "Entry '%s' exists and is not a Group." % key
            raise TypeError(msg)
        else:
            return g

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       **kwargs):
        """Create a new dataset.

        """

        if kwargs:
            msg = ("No extra keyword arguments accepted, this is not an hdf5"
                   " object but a memory object mocked up to look like one.")
            raise TypeError(msg)
            # XXX In future could accept extra arguments and use them if
            # writing to disk.
        if not data is None:
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
        if (not data is None and shape == data.shape
            and dtype is data.dtype and hasattr(data, 'view')):
            new_dataset = data.view(MemDataset)
        else:
            # Just copy the data.
            new_dataset = np.empty(shape=shape,
                                   dtype=dtype).view(MemDataset)
            if not data is None:
                new_dataset[...] = data[...]
        self._dict[name] = new_dataset
        return new_dataset

    def require_dataset(self, shape, dtype):
        try:
            d = self[key]
        except KeyError:
            return self.create_dataset(key, shape=shape, dtype=dtype)
        if isinstance(g, MemGroup):
            msg = "Entry '%s' exists and is not a Dataset." % key
            raise TypeError(msg)
        else:
            return d


class MemAttrs(dict):
    """In memory implementation of the ``h5py.AttributeManager``.

    Currently just a normal dictionary.

    """

    pass


class MemDataset(np.ndarray):
    """In memory implementation of the ``h5py.Dataset`` class.

    Numpy array mocked up to look like an hdf5 dataset.  This just allows a
    numpy array to carry around ab `attrs` dictionary as a stand-in for hdf5
    attributes.

    Attributes
    ----------
    attrs : MemAttrs

    """

    def __array_finalize__(self, obj):
        self._attrs = MemAttrs(getattr(obj, 'attrs', {}))

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


# Higher Level Data Containers
# ----------------------------

class MemDiskGroup(collections.Mapping):
    """Container whose data may either be stored on disk or in memory.

    This container is intended to have the same basic API :class:`h5py.Group`
    and :class:`MemGroup` but whose underlying data could live either on disk
    in the former or in memory in the later.

    Aside from providing a few convenience methods, this class isn't that
    useful by itself. It is almost as easy to use :class:`h5py.Group`
    or :class:`MemGroup` directly. Where it becomes more useful is for creating
    more specialized data containers which can subclass this class.  A basic
    but useful example is provided in :class:`BasicCont`.

    Parameters
    ----------
    data_group : :class:`h5py.Group`, :class:`MemGroup or string, optional
        Underlying :mod:`h5py` like data container where data will be stored.
        If a string, open a h5py file with that name. If not
        provided a new :class:`MemGroup` instance will be created.

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
    from_file
    dataset_name_allowed
    group_name_allowed
    create_dataset
    create_group
    require_group
    to_memory
    to_disk
    flush
    close
    save

    """

    def __init__(self, data_group=None):
        if data_group is None:
            data_group = MemGroup()
        else:
            data_group, self._toclose = get_h5py_File(data_group)
        self._data = data_group

    def __del__(self):
        """Closes file if on disk and if file was opened on initialization."""
        if self.ondisk and self._toclose:
            self._data.close()

    def __getitem__(self, key):
        value = self._data[key]
        if is_group(value):
            if self.group_name_allowed(key):
                return value
            else:
                msg = "Access to group %s not allowed." % key
                raise KeyError(msg)
        else:
            if self.dataset_name_allowed(key):
                return value
            else:
                msg = "Access to dataset %s not allowed." % key
                raise KeyError(msg)

    def __len__(self):
        n = 0
        for key in self:
            n += 1
        return n

    def __iter__(self):
        for key, value in self._data.items():
            if ((is_group(value) and self.group_name_allowed(key))
                or (not is_group(value) and self.dataset_name_allowed(key))):
                yield key
            else:
                continue

    # TODO, something similar to __getitem__() for len() and __iter__().

    ## The main interface ##

    @property
    def attrs(self):
        return self._data.attrs

    @property
    def name(self):
        return self._data.name

    @property
    def parent(self):
        return self._data.parent

    @property
    def file(self):
        return self._data.file

    @property
    def ondisk(self):
        """Whether the data is stored on disk as opposed to in memory."""
        return not isinstance(self._data, MemGroup)

    ## For creating new instances. ##

    @classmethod
    def from_file(cls, f, ondisk=False, **kwargs):
        """Create data object from analysis hdf5 file, store in memory or on disk.

        If *ondisk* is True, do not load into memory but store data in h5py objects
        that remain associated with the file on disk.

        Parameters
        ----------
        f : filename or h5py.File object
            File with the hdf5 data. File must be compatible with analysis hdf5
            format. For loading acquisition format data see `from_acq_h5`.
        ondisk : bool
            Whether the data should be kept in the file on disk or should be copied
            into memory.

        Any additional keyword arguments are passed to :class:`h5py.File`
        constructor if *f* is a filename and silently ignored otherwise.

        """

        f, opened = get_h5py_File(f, **kwargs)
        if not ondisk:
            data = MemGroup.from_group(f)
            if opened:
                f.close()
            toclose = False
        else:
            data = f
            toclose = opened
        self = cls(data)
        self._toclose = toclose
        return self

    ## Methods for manipulating and building the class. ##

    def group_name_allowed(self, name):
        """Used by subclasses to restrict creation of and access to groups.

        This method is called by :meth:`create_group`, :meth:`require_group`,
        and :meth:`__getitem__` to check that the supplied group name is
        allowed.

        The idea is that subclasses that want to specialize and restrict the
        layout of the data container can implement this method instead of
        re-implementing the above mentioned methods.

        Returns
        -------
        allowed : bool
            ``True``.

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

        Returns
        -------
        allowed : bool
            ``True``.

        """
        return True

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       attrs=None, **kwargs):
        if not self.dataset_name_allowed(name):
            msg = "Dataset name %s not allowed." % name
            raise ValueError(msg)
        new_dataset = self._data.create_dataset(name, shape, dtype, data,
                                                **kwargs)
        # Copy over the attributes.  XXX Get rid of this?
        if attrs:
            for key, value in attrs.iteritems():
                new_dataset.attrs[key] = value
        return new_dataset

    def require_dataset(self, key, *args, **kwargs):
        if not self.dataset_name_allowed(key):
            msg = "Dataset name %s not allowed." % name
            raise ValueError(msg)
        return self._data.require_dataset(name, *args, **kwargs)

    def create_group(self, key):
        if not self.group_name_allowed(key):
            msg = "Group name %s not allowed." % key
            raise ValueError(msg)
        return self._data.create_group(key)

    def require_group(self, key):
        if not self.group_name_allowed(name):
            msg = "Group name %s not allowed." % name
            raise ValueError(msg)
        return self._data.require_group(key)

    def to_memory(self):
        """Return a version of this data that lives in memory."""

        if isinstance(self._data, MemGroup):
            return self
        else:
            return self.__class__.from_file(self._data)

    def to_disk(self, h5_file, **kwargs):
        """Return a version of this data that lives on disk."""

        if not isinstance(self._data, MemGroup):
            msg = ("This data already lives on disk.  Copying to new file"
                   " anyway.")
            warnings.warn(msg)
        self.save(h5_file)
        return self.__class__.from_file(h5_file, ondisk=True, **kwargs)

    def close(self):
        """Close underlying hdf5 file if on disk."""

        if self.ondisk:
            self._data.close()

    def flush(self):
        """Flush the buffers of the underlying hdf5 file if on disk."""

        if self.ondisk:
            self._data.flush()

    def save(self, f, **kwargs):
        """Save data to hdf5 file."""

        f, opened = get_h5py_File(f, **kwargs)
        deep_group_copy(self._data, f)
        if opened:
            f.close()


class BasicCont(MemDiskGroup):
    """Basic high level data container.

    Inherits from :class:`MemDiskGroup`.

    Basic one-level data container that allows any number of data sets in the
    root group but no nesting. Data history tracking (in
    :attr:`BasicCont.history`) and array axis interpretation (in
    :attr:`BasicCont.index_map`) is also provided.

    This container is intended to be an example of how a high level container,
    with a strictly controlled data layout can be implemented by subclassing
    :class:`MemDiskGroup`.

    Parameters
    ----------
    data_group : :class:`h5py.Group`, :class:`MemGroup or string, optional
        Underlying :mod:`h5py` like data container where data will be stored.
        If a string, open an h5py file with that name. If not
        provided a new :class:`MemGroup` instance will be created.

    Attributes
    ----------
    datasets
    index_map
    history

    Methods
    -------
    group_name_allowed
    dataset_name_allowed
    create_index_map
    add_history

    """

    def __init__(self, data_group=None):
        MemDiskGroup.__init__(self, data_group)
        self._data.require_group(u'history')
        self._data.require_group(u'index_map')
        if not 'order' in self._data['history'].attrs.keys():
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
        for name, value in self._data['history'].iteritems():
            out[name] = value.attrs
        out[u'order'] = eval(self._data['history'].attrs['order'])
        return ro_dict(out)

    @property
    def index_map(self):
        """Stores representions of the axes of datasets.

        The index map contains arrays used to interpret the axes of the
        variouse datasets. For instance, the 'time', 'prod' and 'freq' axes of
        the visibilities are described in the index map.

        Do not try to add a new index_map by assigning to an item of this
        property. Use :meth:`~BasicCont.create_index_map` instead.

        Returns
        -------
        index_map : read only dictionary
            Entries are 1D arrays used to interpret the axes of datasets.

        """

        out = {}
        for name, value in self._data['index_map'].iteritems():
            out[name] = value
        return ro_dict(out)

    def group_name_allowed(self, name):
        """No groups are exposed to the user. Returns ``False``."""
        return False

    def dataset_name_allowed(self, name):
        """Datasets may only be created and accessed in the root level group.

        Returns ``True`` is *name* contains no '/' characters.

        """
        return False if '/' in name else True

    def create_index_map(self, axis_name, index_map):
        """Create a new index map.

        """

        self._data['index_map'].create_dataset(axis_name, data=index_map)

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
        history_group.attrs[u'order'] = str(order)
        history_group.create_group(name)
        for key, value in history.items():
            history_group[name].attrs[key] = value


# Utilities
# ---------

def attrs2dict(attrs):
    """Safely copy an h5py attributes object to a dictionary."""

    out = {}
    for key, value in attrs.iteritems():
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
        Passed to `h5py.File` constructor. If `f` is already an open file,
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
        #    msg = "Got some keywork arguments but File is alrady open."
        #    warnings.warn(msg)
    else:
        opened = True
        try:
            f = h5py.File(f, **kwargs)
        except IOError as e:
            msg = "Opening file %s caused an error: " % str(f)
            new_e = IOError(msg + str(e))
            raise new_e.__class__, new_e, sys.exc_info()[2]
    return f, opened


def copyattrs(a1, a2):
    # Make sure everything is a copy.
    a1 = attrs2dict(a1)
    for key, value in a1.iteritems():
        a2[key] = value


def deep_group_copy(g1, g2):
    """Copy full data tree from one group to another."""
    
    copyattrs(g1.attrs, g2.attrs)
    for key, entry in g1.iteritems():
        if is_group(entry):
            g2.create_group(key)
            deep_group_copy(entry, g2[key])
        else:
            g2.create_dataset(key, shape=entry.shape, dtype=entry.dtype, 
                    data=entry)
            copyattrs(entry.attrs, g2[key].attrs)

