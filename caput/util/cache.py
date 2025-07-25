"""Tools for caching expensive calculations."""

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
