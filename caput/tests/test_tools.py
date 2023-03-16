"""Unit tests for the tools module."""

import numpy as np

from caput import tools, mpiarray


def test_allequal():
    # Test some basic types
    assert tools.allequal(1, 1)
    assert tools.allequal("x", "x")
    assert tools.allequal([1, 2, 3], [1, 2, 3])
    assert tools.allequal({"a": 1}, {"a": 1})
    assert tools.allequal({1, 2, 3}, {1, 2, 3})
    assert tools.allequal((1, 2, 3), (1, 2, 3))

    # Test numpy arrays and mpiarrays
    assert tools.allequal(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert tools.allequal(
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )
    assert not tools.allequal(
        np.array([1, 2, 3]),
        mpiarray.MPIArray.wrap(np.array([1, 2, 3]), axis=0),
    )

    # Test objects with numpy arrays in them
    assert tools.allequal(
        [np.array([1]), np.array([2])], [np.array([1]), np.array([2])]
    )
    assert tools.allequal({"a": np.array([1])}, {"a": np.array([1])})

    # Test objects with different types
    assert tools.allequal([1, 3.4, "a"], [1, 3.4, "a"])

    # Test for items that are not equal
    assert not tools.allequal(1, 3)
    assert not tools.allequal(1, 1.1)
    assert not tools.allequal([np.array([1])], [np.array([2])])
    assert not tools.allequal(
        np.array(["x"], dtype="U32"), np.array(["x"], dtype="S32")
    )

    # Test for lengths that are not equal
    assert not tools.allequal([1], [1, 2])
    assert not tools.allequal({"a": 1}, {"a": 1, "b": 2})
