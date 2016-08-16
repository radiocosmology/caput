"""
A set of miscellaneous routines that don't really fit anywhere more specific.

Routines
========

.. autosummary::
    :toctree: generated/

    vectorize
"""


import numpy as np


def vectorize(**base_kwargs):
    """An improved vectorization decorator.

    Unlike the :class:`np.vectorize` decorator this version works on methods in
    addition to functions. It also gives an actual scalar value back for any
    scalar input, instead of returning a 0-dimension array.

    Parameters
    ----------
    **kwargs
        Any keyword arguments accepted by :class:`np.vectorize`

    Returns
    -------
    vectorized_function : func
    """

    class _vectorize_desc(object):
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

            if not len(arr.shape):
                arr = np.asscalar(arr)

            return arr

        def __get__(self, obj, type=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, type)
            return self.__class__(new_func)

    return _vectorize_desc
