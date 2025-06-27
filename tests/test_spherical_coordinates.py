"""Test interferometry routines."""

import pytest
from math import pi, sqrt

from caput.astro.coordinates import spherical


def test_sphdist():
    from skyfield.units import Angle

    # 90 degrees
    assert spherical.sphdist(
        Angle(radians=0), Angle(radians=0), Angle(radians=pi / 2), Angle(radians=0)
    ).radians == pytest.approx(pi / 2)
    assert spherical.sphdist(
        Angle(radians=0), Angle(radians=0), Angle(radians=0), Angle(radians=pi / 2)
    ).radians == pytest.approx(pi / 2)

    # 60 degrees
    assert spherical.sphdist(
        Angle(radians=0), Angle(radians=0), Angle(radians=pi / 4), Angle(radians=pi / 4)
    ).radians == pytest.approx(pi / 3)
    assert spherical.sphdist(
        Angle(radians=pi / 4), Angle(radians=pi / 4), Angle(radians=0), Angle(radians=0)
    ).radians == pytest.approx(pi / 3)


def test_rotate_ypr():

    # No rotation
    basis = spherical.rotate_ypr([0, 0, 0], 1, 2, 3)
    assert basis == pytest.approx([1, 2, 3])

    # Rotate into +Y two ways
    basis = spherical.rotate_ypr([pi / 2, 0, 0], 1, 0, 0)
    assert basis == pytest.approx([0, 1, 0])

    basis = spherical.rotate_ypr([0, pi / 2, 0], 0, 0, 1)
    assert basis == pytest.approx([0, 1, 0])

    # General rotation
    x0 = sqrt(2) / 2
    y0 = 0.5
    z0 = 0.5
    basis = spherical.rotate_ypr([pi / 2, pi / 3, pi / 4], x0, y0, z0)

    # The calculation goes like this ("v" denotes a square root):

    # After yawing by 90 degrees:
    # x1 = -y0               y1 = x0                     z1 = z0

    # Then pitching by 60 degrees:
    # x2 = x1                y2 = 0.5 * y1 + v3/2 z1     z2 = -v3/2 y1 + 0.5 z1

    # Then rolling by 45 degrees:
    # x3 = v2/2 (x2 - z2)    y3 = y2                     z3 = v2/2 (x2 + z2)

    assert basis[0] == pytest.approx(sqrt(2) / 2 * (-y0 + sqrt(3) * x0 / 2 - z0 / 2))
    assert basis[1] == pytest.approx(x0 / 2 + sqrt(3) * z0 / 2)
    assert basis[2] == pytest.approx(sqrt(2) / 2 * (-y0 - sqrt(3) * x0 / 2 + z0 / 2))
