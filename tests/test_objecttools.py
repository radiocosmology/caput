"""Unit tests for the tools module."""

import numpy as np
from caput import darray
from caput.util import objecttools


def test_allequal():
    # Test some basic types
    assert objecttools.allequal(1, 1)
    assert objecttools.allequal("x", "x")
    assert objecttools.allequal([1, 2, 3], [1, 2, 3])
    assert objecttools.allequal({"a": 1}, {"a": 1})
    assert objecttools.allequal({1, 2, 3}, {1, 2, 3})
    assert objecttools.allequal((1, 2, 3), (1, 2, 3))

    # Test numpy arrays and mpiarrays
    assert objecttools.allequal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert objecttools.allequal(
        darray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
        darray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )
    assert not objecttools.allequal(
        np.array([1, 2, 3]),
        darray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )

    # Test objects with numpy arrays in them
    assert objecttools.allequal(
        [np.array([1]), np.array([2])], [np.array([1]), np.array([2])]
    )
    assert objecttools.allequal({"a": np.array([1])}, {"a": np.array([1])})

    # Test objects with different types
    assert objecttools.allequal([1, 3.4, "a"], [1, 3.4, "a"])

    # Test for items that are not equal
    assert not objecttools.allequal(1, 3)
    assert not objecttools.allequal(1, 1.1)
    assert not objecttools.allequal([np.array([1])], [np.array([2])])
    assert not objecttools.allequal(
        np.array(["x"], dtype="U32"), np.array(["x"], dtype="S32")
    )

    # Test for lengths that are not equal
    assert not objecttools.allequal([1], [1, 2])
    assert not objecttools.allequal({"a": 1}, {"a": 1, "b": 2})
