"""Module for making in-memory mockups of h5py objects."""


import collections

import numpy as np
import h5py


# Classes
# --------------

class ro_dict(collections.Mapping):
    """A dict that is read-only to the user.

    This class isn't strictly read-only but it cannot be modified through the
    traditional dict interface. This prevents the user from
    mistaking this for a normal dictionary.

    """
    
    def __init__(self, d=None):
        if not d:
            d = {}
        self._dict = d

    def __getitem__(self, key):
        return self._dict[key]

    def __len__(self):
        return self._dict.__len__()

    def __iter__(self):
        return self._dict.__iter__()


class MemGroup(ro_dict):
    """Dictionary mocked up to look like an hdf5 group.

    This exposes the bare minimum of the `h5py` `Group` interface.
    """

    def __init__(self):
        ro_dict.__init__(self)
        self._attrs = MemAttrs()

    def __getitem__(self, key):
        """Impliment '/' for accessing nested groups."""
        key_parts = key.split('/')
        if len(key_parts) == 1:
            return ro_dict.__getitem__(self, key)
        else:
            return self[key_parts[0]][key_parts[1:].join('/')]

    @property
    def attrs(self):
        return self._attrs

    @classmethod
    def from_group(cls, group):
        """Create a new instance by deep copying an existing group.

        Agnostic as to whether the group to be copyed is a `MemGroup` or an
        `h5py.Group` (which includes `hdf5.File` objects). 
        """

        self = cls()
        for key, entry in group.iteritems():
            if is_group(entry):
                self._dict[key] = MemGroup.from_group(entry)
                self[key].attrs = copyattrs(entry.attrs)
            else:
                self.create_dataset(key, entry)
        return self

    def create_group(self, key):
        if key in self.keys():
            msg = "Group '%s' already exists." % key
            raise ValueError(msg)
        else:
            self._dict[key] = MemGroup()

    def require_group(self, key):
        if key in self.keys():
            if not isinstance(self[key], MemGroup):
                msg = "Entry '%s' exists and is not a Group." % key
                raise TypeError(msg)
            else:
                pass
        else:
            self.create_group(key)

    def create_dataset(self, name, shape=None, dtype=None, data=None,
                       attrs=None, **kwargs):
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
            new_dataset[...] = data[...]
        self._datasets[name] = new_dataset
        # Copy over the attributes.
        # XXX What about data.attrs!
        # XXX What about making a copy using functions below.
        if attrs:
            for key, value in attrs.iteritems():
                new_dataset.attrs[key] = value


class MemAttrs(dict):
    """Dictionary mocked up to look like hdf5 attributes.

    Currently just a normal dictionary.

    """

    pass


class MemDataset(np.ndarray):
    """Numpy array mocked up to look like an hdf5 dataset.
    
    This just allows a numpy array to carry around ab `attrs` dictionary
    as a stand-in for hdf5 attributes.
    """
    
    def __array_finalize__(self, obj):
        self._attrs = MemAttrs(getattr(obj, 'attrs', {}))

    @property
    def attrs(self):
        return self._attrs


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

def copyattrs(attrs):
    out = attrs2dict(attrs)
    out = MemAttrs(out)
    return out

def is_group(obj):
    """Check if the object is an h5py Group, which includes File objects."""
    
    return hasattr(obj, 'create_group')

