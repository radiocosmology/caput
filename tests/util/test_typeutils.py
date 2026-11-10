"""Tests for NumPy dtype and string conversion helpers."""

import sys

import numpy as np
import pytest

from caput.util import typeutils


def test_bytes_to_unicode_recurses_through_containers():
    """Decode bytes recursively while preserving unrelated values."""
    source = {"label": b"sky", "items": [b"field", 3], "array": np.array([b"a", b"b"])}
    converted = typeutils.bytes_to_unicode(source)

    assert converted["label"] == "sky"
    assert converted["items"] == ["field", 3]
    np.testing.assert_array_equal(converted["array"], ["a", "b"])
    assert typeutils.bytes_to_unicode(3) == 3


def test_dtype_string_conversion_handles_compound_and_nested_fields():
    """Convert string fields in compound dtypes without altering numeric fields."""
    dtype = np.dtype([("name", "S4"), ("nested", [("label", "S3"), ("count", "i4")])])

    as_unicode = typeutils.dtype_to_unicode(dtype)
    assert as_unicode["name"].kind == "U"
    assert as_unicode["nested"]["label"].kind == "U"
    assert as_unicode["nested"]["count"].kind == "i"

    as_bytes = typeutils.dtype_to_bytestring(as_unicode)
    assert as_bytes["name"].kind == "S"
    assert as_bytes["nested"]["label"].kind == "S"


@pytest.mark.xfail(
    strict=True,
    reason="has_kind indexes nested compound descriptors as if they were scalar fields.",
)
def test_string_kind_detection_recurses_into_compound_fields():
    """Detect byte and Unicode strings in both top-level and nested fields."""
    dtype = np.dtype([("plain", "i4"), ("nested", [("label", "U5")])])
    assert typeutils.has_unicode(dtype)
    assert not typeutils.has_bytestring(dtype)
    assert typeutils.has_bytestring(np.dtype([("label", "S5")]))
    assert not typeutils.has_unicode(np.dtype("i4"))


def test_ensure_string_array_conversions_preserve_numeric_fields():
    """Convert string fields in arrays while leaving numeric values unchanged."""
    values = np.array([("north", 2)], dtype=[("name", "U8"), ("count", "i4")])
    as_bytes = typeutils.ensure_bytestring(values)
    assert as_bytes.dtype["name"].kind == "S"
    np.testing.assert_array_equal(as_bytes["name"], [b"north"])
    np.testing.assert_array_equal(as_bytes["count"], [2])

    as_unicode = typeutils.ensure_unicode(as_bytes)
    assert as_unicode.dtype["name"].kind == "U"
    np.testing.assert_array_equal(as_unicode["name"], ["north"])


def test_byteorder_helpers_distinguish_native_and_non_native_orders():
    """Recognize native byte order across little- and big-endian platforms."""
    native = "<" if sys.byteorder == "little" else ">"
    opposite = ">" if sys.byteorder == "little" else "<"

    assert typeutils.check_byteorder("=") is False
    assert typeutils.has_matching_byteorder(native)
    assert typeutils.check_byteorder(native)
    assert not typeutils.has_matching_byteorder(opposite)
    assert not typeutils.check_byteorder(opposite)


def test_check_unicode_returns_data_or_names_unicode_dataset():
    """Return bytes data and reject Unicode datasets with the dataset name."""

    class Dataset:
        def __init__(self, dtype, name, data):
            self.dtype = np.dtype(dtype)
            self.name = name
            self.data = data

    data = np.array([b"value"])
    assert typeutils.check_unicode(Dataset("S5", "/bytes", data)) is data

    with pytest.raises(TypeError, match=r"/unicode.*unicode type"):
        typeutils.check_unicode(Dataset("U5", "/unicode", None))
