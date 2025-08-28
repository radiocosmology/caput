"""Tools for working with numpy arrays and subclasses."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import cachetools
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Any

    import numpy.typing as npt

__all__ = [
    "LRUCache",
    "listize",
    "partition_list",
    "scalarize",
    "split_m",
    "unique_ordered",
    "vectorize",
]


class LRUCache(cachetools.LRUCache):
    """An LRU cache for numpy arrays that will expand to a maximum size in bytes.

    This should be used like a dictionary except that the least recently used entries
    are evicted to restrict memory usage to the specified maximum.

    Parameters
    ----------
    size_bytes : int
        The maximum size of the cache in bytes.
    """

    def __init__(self, size_bytes: int) -> None:
        def _array_size(arr: np.ndarray):
            if not isinstance(arr, np.ndarray):
                raise ValueError(f"Item must be a numpy array. Got {type(arr)}.")

            return arr.nbytes

        super().__init__(maxsize=size_bytes, getsizeof=_array_size)


def partition_list(full_list: list, i: int, n: int, method: str = "con") -> list:
    """Partition a list into `n` pieces. Return the `i` th partition.

    Parameters
    ----------
    full_list : list
        The full list to partition.
    i : int
        The index of the partition to return.
    n : int
        The number of partitions to create.
    method: {"con", "alt", "rand"}, optional
        The method to use for partitioning. Options are:

        - "con": contiguous partitionsI
        - "alt": alternating partitions
        - "rand": random partitions

        Default is "con".

    Returns
    -------
    partition : list
        The `i` th partition of the list.
    """

    def _partition(N: int, n: int, i: int) -> tuple[int, int]:
        # If partition `N` numbers into `n` pieces,
        # return the start and stop of the `i` th piece
        base = N // n
        rem = N % n
        num_lst = rem * [base + 1] + (n - rem) * [base]
        cum_num_lst = np.cumsum([0, *num_lst])

        return cum_num_lst[i], cum_num_lst[i + 1]

    N = len(full_list)
    start, stop = _partition(N, n, i)

    if method == "con":
        return full_list[start:stop]

    if method == "alt":
        return full_list[i::n]

    if method == "rand":
        choices = np.random.permutation(N)[start:stop]
        return [full_list[i] for i in choices]

    raise ValueError(f"Unknown partition method {method}")


def split_m(n: int, m: int) -> np.ndarray:
    """Split a range (0, n-1) into m sub-ranges of similar length.

    Parameters
    ----------
    n : int
        Length of range to split.
    m : int
        Number of subranges to split into.

    Returns
    -------
    split : np.ndarray
        :py:obj:`np.ndarray` of shape (3, m) where each column contains:

        - Number in each sub-range
        - Starting of each sub-range.
        - End of each sub-range.

    See Also
    --------
    :py:func:`split_all`, :py:func:`split_local`
    """
    base = n // m
    rem = n % m

    part = base * np.ones(m, dtype=int) + (np.arange(m) < rem).astype(int)

    bound = np.cumsum(np.insert(part, 0, 0))

    return np.array([part, bound[:m], bound[1 : (m + 1)]])


def unique_ordered(x: Iterable) -> list:
    """Take unique values from an iterable with order preserved.

    Parameters
    ----------
    x : Iterable
        An iterable which may or may not container duplicate values.

    Returns
    -------
    unique : list
        Unique items in x with order preserved.
    """
    seen = set()
    # So the method is only resolved once
    seen_add = seen.add

    return [i for i in x if not (i in seen or seen_add(i))]


class _decorator:
    """Signature-preserving decorator class."""

    # See http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
    # for a description of this pattern

    def __init__(self, func):
        # Save a reference to the function and set various properties so the
        # docstrings etc. get passed through
        self.func = func
        self.__doc__ = func.__doc__
        self.__name__ = func.__name__
        self.__module__ = func.__module__

        # Pulled from https://github.com/micheles/decorator/blob/master/src/decorator.py#L285
        self.__qualname__ = func.__qualname__
        self.__kwdefaults__ = getattr(func, "__kwdefaults__", None)
        self.__dict__.update(func.__dict__)

        sig = inspect.signature(func)
        dec_params = [
            p
            for p in sig.parameters.values()
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        self.__signature__ = sig.replace(parameters=dec_params)


def vectorize(**base_kwargs: Any) -> Callable:  # noqa: D417
    r"""Improved vectorization decorator.

    Unlike the :py:class:`np.vectorize` decorator this version works on methods in
    addition to functions. It also gives an actual scalar value back for any
    scalar input, instead of returning a 0-dimension array.

    Parameters
    ----------
    \**base_kwargs : Any
        Any keyword arguments accepted by :py:class:`np.vectorize`.

    Returns
    -------
    vectorized : callable
        Vectorized function.
    """

    class _vectorize_desc(_decorator):
        def __call__(self, *args, **kwargs):
            # This gets called whenever the wrapped function is invoked
            arr = np.vectorize(self.func, **base_kwargs)(*args, **kwargs)

            if not arr.shape:
                arr = arr.item()

            return arr

        def __get__(self, obj, objtype=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, objtype)
            return self.__class__(new_func)

    return _vectorize_desc


def scalarize(dtype: npt.DTypeLike = np.float64) -> Callable:
    """Handle scalars and other iterables being passed to numpy requiring code.

    Parameters
    ----------
    dtype : dtype
        The output datatype. Used only to set the return type of zero-length arrays.

    Returns
    -------
    scalarized : callable
        Scalarized function.
    """

    class _scalarize_desc(_decorator):
        def __call__(self, *args, **kwargs):
            # This gets called whenever the wrapped function is invoked

            args, scalar, empty = zip(*[self._make_array(a) for a in args])

            if all(empty):
                return np.array([], dtype=dtype)

            ret = self.func(*args, **kwargs)

            if all(scalar):
                ret = ret[0]

            return ret

        @staticmethod
        def _make_array(x):
            # Change iterables to arrays and scalars into length-1 arrays

            from skyfield import timelib

            # Special handling for the slightly awkward skyfield types
            if isinstance(x, timelib.Time):
                if isinstance(x.tt, np.ndarray):
                    scalar = False
                else:
                    scalar = True
                    x = x.ts.tt_jd(np.array([x.tt]))

            elif isinstance(x, np.ndarray):
                scalar = False

            elif isinstance(x, list | tuple):
                x = np.array(x)
                scalar = False

            else:
                x = np.array([x])
                scalar = True

            return (x, scalar, len(x) == 0)

        def __get__(self, obj, objtype=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, objtype)
            return self.__class__(new_func)

    return _scalarize_desc


def listize(**_) -> Callable:
    """Make functions that already work with :py:class:`np.ndarray` or scalars accept lists.

    Also works with tuples.

    Returns
    -------
    listized : callable
        Listized functioni.
    """

    class _listize_desc(_decorator):
        def __call__(self, *args, **kwargs):
            # This gets called whenever the wrapped function is invoked

            new_args = []
            for arg in args:
                if isinstance(arg, list | tuple):
                    arg = np.array(arg)
                new_args.append(arg)

            return self.func(*new_args, **kwargs)

        def __get__(self, obj, objtype=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, objtype)
            return self.__class__(new_func)

    return _listize_desc
