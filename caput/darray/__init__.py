"""Caput distributed arrays.

An MPI-aware numpy array and array utilities.
"""

import numpy as np

from . import _cache, _invert_no_zero, _mpiarray
from ._cache import *  # noqa: F403
from ._invert_no_zero import *  # noqa: F403
from ._mpiarray import *  # noqa: F403


__all__ = [
    "listize",
    "scalarize",
    "vectorize",
    *_cache.__all__,
    *_invert_no_zero.__all__,
    *_mpiarray.__all__,
]


def vectorize(**base_kwargs):
    """Improved vectorization decorator.

    Unlike the :class:`np.vectorize` decorator this version works on methods in
    addition to functions. It also gives an actual scalar value back for any
    scalar input, instead of returning a 0-dimension array.

    Parameters
    ----------
    **base_kwargs
        Any keyword arguments accepted by :class:`np.vectorize`

    Returns
    -------
    vectorized_function : func
    """

    class _vectorize_desc:
        # See
        # http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
        # for a description of this pattern

        def __init__(self, func):
            # Save a reference to the function and set various properties so the
            # docstrings etc. get passed through
            self.func = func
            self.__doc__ = func.__doc__
            self.__name__ = func.__name__
            self.__module__ = func.__module__

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


def scalarize(dtype=np.float64):
    """Handle scalars and other iterables being passed to numpy requiring code.

    Parameters
    ----------
    dtype : np.dtype, optional
        The output datatype. Used only to set the return type of zero-length arrays.

    Returns
    -------
    vectorized_function : func
    """

    class _scalarize_desc:
        # See
        # http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
        # for a description of this pattern

        def __init__(self, func):
            # Save a reference to the function and set various properties so the
            # docstrings etc. get passed through
            self.func = func
            self.__doc__ = func.__doc__
            self.__name__ = func.__name__
            self.__module__ = func.__module__

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


def listize(**_):
    """Make functions that already work with `np.ndarray` or scalars accept lists.

    Also works with tuples.

    Returns
    -------
    listized_function : func
    """

    class _listize_desc:
        def __init__(self, func):
            # Save a reference to the function and set various properties so the
            # docstrings etc. get passed through
            self.func = func
            self.__doc__ = func.__doc__
            self.__name__ = func.__name__
            self.__module__ = func.__module__

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
