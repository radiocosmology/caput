"""Tests for dynamic import helpers."""

import pytest

from caput.util import importtools


def test_import_class_loads_qualified_class():
    """Resolve a class from its fully qualified import path."""
    assert importtools.import_class("builtins.dict") is dict


def test_import_class_propagates_missing_attribute():
    """Report a missing class attribute instead of returning a fallback."""
    with pytest.raises(AttributeError):
        importtools.import_class("builtins.NotAType")
