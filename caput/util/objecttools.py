"""Generic object utilities."""

from collections import deque
from collections.abc import Mapping
from itertools import chain
from numbers import Number
from sys import getsizeof
from types import ModuleType

import numpy as np

__all__ = ["allequal", "total_size"]


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
