"""Test accessing astronomical (and other) units."""

import pytest

from caput.astro import units


def test_scipy_units():
    """Check a handful of scipy units."""
    assert units.c == 299792458.0
    assert units.k == 1.380649e-23
    assert units.year == 31536000.0
    assert units.G == 6.6743e-11


def test_special_units():
    """Check that we get the correct values for custom units."""
    assert units.solar_mass == 1.98892e30
    assert units.second == 1.0
    assert units.t_sidereal == 86164.09056
    assert units.a_rad == 7.565733250280007e-16
    assert units.nu21 == 1420.40575177


def test_prefix_units():
    """Check that prefixes are applied correctly."""
    assert units.kilo_second == 1000.0
    assert units.pico_year == 3.1536e-05
    assert units.kilo_gram == 1.0


@pytest.mark.parametrize("name", ["nan", "sceond", "year_"])
def test_invalid_unit(name):
    """Check that invalid units are caught."""
    with pytest.raises(AttributeError) as err:
        getattr(units, name)

    assert str(err.value) == f"Cannot find constant `{name}`."


@pytest.mark.parametrize("name", ["gigo_year"])
def test_invalid_prefix(name):
    """Check that invalid units are caught."""
    with pytest.raises(AttributeError) as err:
        getattr(units, name)

    # This will fail _after_ splitting off the prefix
    assert str(err.value) == f"Cannot find constant `{name.split('_')[0]}`."


@pytest.mark.parametrize("name", ["pico_giga_year"])
def test_too_many_prefixes(name):
    with pytest.raises(AttributeError) as err:
        getattr(units, name)

    assert str(err.value) == f"Only single-prefix values are allowed. Got `{name}`."
