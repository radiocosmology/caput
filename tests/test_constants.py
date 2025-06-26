"""Test accessing astronomical (and other) constants."""

import pytest

from caput.astro import constants


def test_scipy_constants():
    """Check a handful of scipy constants."""
    assert constants.c == 299792458.0
    assert constants.k == 1.380649e-23
    assert constants.year == 31536000.0
    assert constants.G == 6.6743e-11


def test_special_constants():
    """Check that we get the correct values for custom constants."""
    assert constants.solar_mass == 1.98892e30
    assert constants.second == 1.0
    assert constants.t_sidereal == 86164.09056
    assert constants.a_rad == 7.565733250280007e-16
    assert constants.nu21 == 1420.40575177


def test_prefix_constants():
    """Check that prefixes are applied correctly."""
    assert constants.kilo_second == 1000.0
    assert constants.pico_year == 3.1536e-05
    assert constants.kilo_gram == 1.0


@pytest.mark.parametrize("name", ["nan", "sceond", "year_"])
def test_invalid_constant(name):
    """Check that invalid constants are caught."""
    with pytest.raises(AttributeError) as err:
        getattr(constants, name)

    assert str(err.value) == f"Cannot find constant `{name}`."


@pytest.mark.parametrize("name", ["gigo_year"])
def test_invalid_prefix(name):
    """Check that invalid constants are caught."""
    with pytest.raises(AttributeError) as err:
        getattr(constants, name)

    # This will fail _after_ splitting off the prefix
    assert str(err.value) == f"Cannot find constant `{name.split('_')[0]}`."


@pytest.mark.parametrize("name", ["pico_giga_year"])
def test_too_many_prefixes(name):
    with pytest.raises(AttributeError) as err:
        getattr(constants, name)

    assert str(err.value) == f"Only single-prefix values are allowed. Got `{name}`."
