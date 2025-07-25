"""Unit tests for the tools module."""

import numpy as np
from caput import mpiarray
from caput.util import objectutil


def test_allequal():
    # Test some basic types
    assert objectutil.allequal(1, 1)
    assert objectutil.allequal("x", "x")
    assert objectutil.allequal([1, 2, 3], [1, 2, 3])
    assert objectutil.allequal({"a": 1}, {"a": 1})
    assert objectutil.allequal({1, 2, 3}, {1, 2, 3})
    assert objectutil.allequal((1, 2, 3), (1, 2, 3))

    # Test numpy arrays and mpiarrays
    assert objectutil.allequal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert objectutil.allequal(
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )
    assert not objectutil.allequal(
        np.array([1, 2, 3]),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )

    # Test objects with numpy arrays in them
    assert objectutil.allequal(
        [np.array([1]), np.array([2])], [np.array([1]), np.array([2])]
    )
    assert objectutil.allequal({"a": np.array([1])}, {"a": np.array([1])})

    # Test objects with different types
    assert objectutil.allequal([1, 3.4, "a"], [1, 3.4, "a"])

    # Test for items that are not equal
    assert not objectutil.allequal(1, 3)
    assert not objectutil.allequal(1, 1.1)
    assert not objectutil.allequal([np.array([1])], [np.array([2])])
    assert not objectutil.allequal(
        np.array(["x"], dtype="U32"), np.array(["x"], dtype="S32")
    )

    # Test for lengths that are not equal
    assert not objectutil.allequal([1], [1, 2])
    assert not objectutil.allequal({"a": 1}, {"a": 1, "b": 2})
