"""Collection of assorted tools."""

from typing import Iterable

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
    if not isinstance(x, (np.generic, np.ndarray)) or np.issubdtype(
        x.dtype, np.integer
    ):
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

    if isinstance(obj1, (list, tuple)) and isinstance(obj2, (list, tuple)):
        # Check that the sequence-type objects are equivalent
        _assert_equal(len(obj1), len(obj2))
        # Check that each item is the same
        for k in range(len(obj1)):
            _assert_equal(obj1[k], obj2[k])

        return

    # If both objects are np.ndarray subclasses, check that they
    # have the same type
    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        assert type(obj1) == type(obj2)

        obj1 = obj1.view(np.ndarray)
        obj2 = obj2.view(np.ndarray)

    # Check all other built-in types and numpy types are equal
    np.testing.assert_equal(obj1, obj2)
