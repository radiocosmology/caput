"""Unit tests for the tools module."""

import numpy as np
from caput import mpiarray
from caput.lib import misc


def test_allequal():
    # Test some basic types
    assert misc.allequal(1, 1)
    assert misc.allequal("x", "x")
    assert misc.allequal([1, 2, 3], [1, 2, 3])
    assert misc.allequal({"a": 1}, {"a": 1})
    assert misc.allequal({1, 2, 3}, {1, 2, 3})
    assert misc.allequal((1, 2, 3), (1, 2, 3))

    # Test numpy arrays and mpiarrays
    assert misc.allequal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert misc.allequal(
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )
    assert not misc.allequal(
        np.array([1, 2, 3]),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )

    # Test objects with numpy arrays in them
    assert misc.allequal([np.array([1]), np.array([2])], [np.array([1]), np.array([2])])
    assert misc.allequal({"a": np.array([1])}, {"a": np.array([1])})

    # Test objects with different types
    assert misc.allequal([1, 3.4, "a"], [1, 3.4, "a"])

    # Test for items that are not equal
    assert not misc.allequal(1, 3)
    assert not misc.allequal(1, 1.1)
    assert not misc.allequal([np.array([1])], [np.array([2])])
    assert not misc.allequal(np.array(["x"], dtype="U32"), np.array(["x"], dtype="S32"))

    # Test for lengths that are not equal
    assert not misc.allequal([1], [1, 2])
    assert not misc.allequal({"a": 1}, {"a": 1, "b": 2})
