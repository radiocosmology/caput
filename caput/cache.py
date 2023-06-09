"""Tools for caching expensive calculations."""
import weakref

import numpy as np
from cachetools import LRUCache


class NumpyCache(LRUCache):  # pylint: disable=too-many-ancestors
    """An LRU cache for numpy arrays that will expand to a maximum size in bytes.

    This should be used like a dictionary except that the least recently used entries
    are evicted to restrict memory usage to the specified maximum.

    Parameters
    ----------
    size_bytes
        The maximum size of the cache in bytes.
    """

    def __init__(self, size_bytes: int):
        def _array_size(arr: np.ndarray):
            if not isinstance(arr, np.ndarray):
                raise ValueError("Item must be a numpy array.")

            return arr.nbytes

        super().__init__(maxsize=size_bytes, getsizeof=_array_size)


class cached_property:
    """An immutable cached property.

    This exists in Python 3.8, and later but we're restricted to Python 3.6.

    Example
    -------
    >>> class A:
    ...     @cached_property
    ...     def v(self):
    ...         print("I'm in here")
    ...         return 2

    >>> a = A()
    >>> a.v
    I'm in here
    2
    >>> a.v
    2
    >>> a.v = 4
    Traceback (most recent call last):
       ...
    AttributeError: Can't set attribute
    """

    def __init__(self, func, doc=None):
        self.func = func
        if doc is None:
            doc = func.__doc__
        self.__doc__ = doc
        self._value_cache = weakref.WeakKeyDictionary()

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        try:
            return self._value_cache[obj]
        except KeyError:
            val = self.func(obj)
            self._value_cache[obj] = val
            return val

    def __set__(self, *args):
        raise AttributeError("Can't set attribute")
