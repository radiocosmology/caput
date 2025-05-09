"""Collection of assorted tools."""

from collections import deque
from collections.abc import Iterable, Mapping
from itertools import chain
from numbers import Number
from sys import getsizeof
from types import ModuleType

import numpy as np

from caput._fast_tools import _invert_no_zero


def invert_no_zero(x, out=None):
    """Return the reciprocal, but ignoring zeros.

    Where `x != 0` return 1/x, or just return 0. Importantly this routine does
    not produce a warning about zero division.

    Parameters
    ----------
    x : np.ndarray
        Array to invert
    out : np.ndarray, optional
        Output array to insert results

    Returns
    -------
    r : np.ndarray
        Return the reciprocal of x. Where possible the output has the same memory layout
        as the input, if this cannot be preserved the output is C-contiguous.
    """
    if not isinstance(x, np.generic | np.ndarray) or np.issubdtype(x.dtype, np.integer):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.where(x == 0, 0.0, 1.0 / x)

    if out is not None:
        if x.shape != out.shape:
            raise ValueError(
                f"Input and output arrays don't have same shape: {x.shape} != {out.shape}."
            )
    else:
        # This works even for MPIArrays, producing a correctly shaped MPIArray
        out = np.empty_like(x, order="A")

    # In order to be able to flatten the arrays to do element by element operations, we
    # need to ensure the inputs are numpy arrays, and so we take a view which will work
    # even if `x` (and thus `out`) are MPIArray's
    _invert_no_zero(
        x.view(np.ndarray).ravel(order="A"), out.view(np.ndarray).ravel(order="A")
    )

    return out


def window_generalised(x, window="nuttall"):
    """A generalised high-order window at arbitrary locations.

    Parameters
    ----------
    x : np.ndarray[n]
        Location to evaluate at. Values outside the range 0 to 1 are zero.
    window : str
        Type of window function to return.  Must be one of the following strings:
        'uniform', 'hann', 'hanning', 'hamming', 'blackman', 'nuttall',
        'blackman_nuttall', 'blackman_harris', 'triangular', or 'tukey-0.X'.
        If "tukey-0.X", then 0.X is the fraction of the full window that
        will be tapered.

    Returns
    -------
    w : np.ndarray[n]
        Window function.
    """
    if window == "triangular":
        w = 1.0 - 2.0 * np.abs(x - 0.5)

    elif window.startswith("tukey"):

        r = float(window.split("-")[1])
        alpha = 0.5 * r

        w = np.ones_like(x)

        begin = np.flatnonzero(x < alpha)
        if begin.size > 0:
            w[begin] = 0.5 * (1.0 + np.cos(np.pi * (x[begin] - alpha) / alpha))

        end = np.flatnonzero(x >= (1.0 - alpha))
        if end.size > 0:
            w[end] = 0.5 * (1.0 + np.cos(np.pi * (x[end] - (1.0 - alpha)) / alpha))

    else:
        a_table = {
            "uniform": np.array([1, 0, 0, 0]),
            "hann": np.array([0.5, -0.5, 0, 0]),
            "hanning": np.array([0.5, -0.5, 0, 0]),
            "hamming": np.array([0.53836, -0.46164, 0, 0]),
            "blackman": np.array([0.42, -0.5, 0.08, 0]),
            "nuttall": np.array([0.355768, -0.487396, 0.144232, -0.012604]),
            "blackman_nuttall": np.array(
                [0.3635819, -0.4891775, 0.1365995, -0.0106411]
            ),
            "blackman_harris": np.array([0.35875, -0.48829, 0.14128, -0.01168]),
        }

        a = a_table[window]
        t = 2 * np.pi * np.arange(4)[:, np.newaxis] * x[np.newaxis, :]
        w = (a[:, np.newaxis] * np.cos(t)).sum(axis=0)

    return np.where((x >= 0) & (x <= 1), w, 0)


def unique_ordered(x: Iterable) -> list:
    """Take unique values from an iterable with order preserved.

    Parameters
    ----------
    x : Iterable
        An iterable to get unique values from

    Returns
    -------
    unique : list
        unique items in x with order preserved
    """
    seen = set()
    # So the method is only resolved once
    seen_add = seen.add

    return [i for i in x if not (i in seen or seen_add(i))]


def allequal(obj1, obj2):
    """Check if two objects are equal.

    This comparison can check standard python types and numpy types,
    even in the case where they are nested (ex: a dict with a numpy
    array as a value).

    Parameters
    ----------
    obj1 : scalar, list, tuple, dict, or np.ndarray
        Object to compare
    obj2 : scalar, list, tuple, dict, or np.ndarray
        Object to compare
    """
    try:
        _assert_equal(obj1, obj2)
    except AssertionError:
        return False
    return True


def _assert_equal(obj1, obj2):
    """Assert two objects are equal.

    This comparison can check standard python types and numpy types,
    even in the case where they are nested (ex: a dict with a numpy
    array as a value).

    For more information:
    https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_equal.html

    Parameters
    ----------
    obj1 : scalar, list, tuple, dict, or np.ndarray
        Object to compare
    obj2 : scalar, list, tuple, dict, or np.ndarray
        Object to compare
    """
    __tracebackhide__ = True
    if isinstance(obj1, dict):
        # Check that the dict-type objects are equivalent
        if not isinstance(obj2, dict):
            raise AssertionError(repr(type(obj2)))
        _assert_equal(len(obj1), len(obj2))
        # Check that each key:value pair are equal
        for k in obj1.keys():
            if k not in obj2:
                raise AssertionError(repr(k))
            _assert_equal(obj1[k], obj2[k])

        return

    if isinstance(obj1, list | tuple) and isinstance(obj2, list | tuple):
        # Check that the sequence-type objects are equivalent
        _assert_equal(len(obj1), len(obj2))
        # Check that each item is the same
        for k in range(len(obj1)):
            _assert_equal(obj1[k], obj2[k])

        return

    # If both objects are np.ndarray subclasses, check that they
    # have the same type
    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        assert type(obj1) is type(obj2)

        obj1 = obj1.view(np.ndarray)
        obj2 = obj2.view(np.ndarray)

    # Check all other built-in types and numpy types are equal
    np.testing.assert_equal(obj1, obj2)


def total_size(obj: object):
    """Return the approximate memory used by an object and anything it references.

    Parameters
    ----------
    obj
        Any object
    """
    # types not to iterate - these can handle their own internal
    # recursion/iteration when calling `sys.getsizeof`
    exclude_types = (str, bytes, Number, range, bytearray)

    seen = set()
    default = getsizeof(0)

    def sum_(x):
        try:
            return sum(x)
        except TypeError:
            return 0

    def sizeof(x):
        if isinstance(x, ModuleType):
            raise TypeError(
                f"function `total_size` is not implemented for type {ModuleType}"
            )
        # Don't check the same item twice
        if id(x) in seen:
            return 0

        seen.add(id(x))
        # Get the base size of this object
        size = getsizeof(x, default)
        # Exclude certain types
        if isinstance(x, exclude_types):
            pass
        # Check basic types and iterate accordingly
        elif isinstance(x, tuple | list | set | deque):
            size += sum_(map(sizeof, iter(x)))
        elif isinstance(x, Mapping):
            size += sum_(map(sizeof, chain.from_iterable(x.items())))

        # Check custom objects
        if hasattr(x, "__dict__"):
            size += sizeof(vars(x))

        if hasattr(x, "__slots__"):
            size += sum_(
                sizeof(getattr(x, attr)) for attr in x.__slots__ if hasattr(x, attr)
            )

        return size

    return sizeof(obj)
