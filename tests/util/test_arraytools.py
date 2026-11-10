"""Tests for NumPy-oriented utility functions."""

import numpy as np
import pytest

from caput.util import arraytools


def test_partition_list_contiguous_and_alternating():
    """Partition a list using contiguous and alternating strategies."""
    values = list(range(8))

    # Uneven partitions should account for every value exactly once.
    contiguous = [arraytools.partition_list(values, i, 3) for i in range(3)]
    assert contiguous == [[0, 1, 2], [3, 4, 5], [6, 7]]

    alternating = [
        arraytools.partition_list(values, i, 3, method="alt") for i in range(3)
    ]
    assert alternating == [[0, 3, 6], [1, 4, 7], [2, 5]]


@pytest.mark.xfail(
    strict=True,
    reason="Each call currently generates a separate random permutation, causing overlap.",
)
def test_partition_list_random_and_invalid_method():
    """Check random partitions preserve membership and invalid methods fail."""
    values = list(range(12))
    parts = [arraytools.partition_list(values, i, 4, method="rand") for i in range(4)]
    assert sorted(value for part in parts for value in part) == values
    with pytest.raises(ValueError, match="Unknown partition method"):
        arraytools.partition_list(values, 0, 2, method="unknown")


def test_split_m_covers_uneven_and_empty_partitions():
    """Split a range into uneven partitions, including empty trailing parts."""
    split = arraytools.split_m(7, 3)
    np.testing.assert_array_equal(split, [[3, 2, 2], [0, 3, 5], [3, 5, 7]])

    empty_parts = arraytools.split_m(2, 4)
    np.testing.assert_array_equal(
        empty_parts, [[1, 1, 0, 0], [0, 1, 2, 2], [1, 2, 2, 2]]
    )


def test_unique_ordered_preserves_first_occurrences():
    """Return unique values in their original order, including for iterators."""
    assert arraytools.unique_ordered([3, 1, 3, 2, 1]) == [3, 1, 2]
    assert arraytools.unique_ordered(iter("abaac")) == ["a", "b", "c"]


def test_lru_cache_tracks_array_bytes_and_rejects_other_values():
    """Evict least-recently-used arrays when the byte limit is exceeded."""
    cache = arraytools.LRUCache(16)
    cache["first"] = np.zeros(1, dtype=np.float64)
    cache["second"] = np.ones(1, dtype=np.float64)
    assert "first" in cache

    cache["third"] = np.full(1, 2.0)
    assert "first" not in cache
    assert "second" in cache
    assert "third" in cache

    with pytest.raises(ValueError, match="Item must be a numpy array"):
        cache["not-an-array"] = [1, 2]


def test_vectorize_returns_scalars_for_scalar_inputs():
    """Vectorize a function while preserving scalar return values."""

    @arraytools.vectorize(otypes=[int])
    def double(value):
        return value * 2

    assert double(3) == 6
    np.testing.assert_array_equal(double([1, 2]), [2, 4])


def test_listize_converts_list_and_tuple_arguments():
    """Convert sequence arguments to arrays before calling a wrapped function."""

    @arraytools.listize()
    def add_one(values):
        return values + 1

    np.testing.assert_array_equal(add_one([1, 2]), [2, 3])
    np.testing.assert_array_equal(add_one((3, 4)), [4, 5])


def test_scalarize_handles_scalars_sequences_and_empty_inputs():
    """Normalize scalar inputs and preserve array results for sequence inputs."""

    @arraytools.scalarize(dtype=np.int64)
    def double(values):
        return values * 2

    # Scalar input returns a scalar, while empty input keeps the requested dtype.
    assert double(3) == 6
    np.testing.assert_array_equal(double([1, 2]), [2, 4])
    empty = double([])
    assert empty.dtype == np.dtype(np.int64)
    assert empty.shape == (0,)
