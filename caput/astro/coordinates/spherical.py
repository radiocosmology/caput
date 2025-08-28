"""Spherical coordinate transformations and projections."""

from __future__ import annotations

import numpy as np
from skyfield.units import Angle

from ._spherical import *  # noqa: F403

__all__ = [
    "cart_to_sph",
    "cosine_rule",
    "great_circle_points",
    "ground_to_sph",
    "norm_vec2",
    "projected_distance",
    "rotate_ypr",
    "sph_dot",
    "sph_to_cart",
    "sph_to_ground",
    "sphdist",
    "thetaphi_plane",
    "thetaphi_plane_cart",
]


def sph_to_ground(ha, lat, dec):
    """Get the ground based XYZ coordinates.

    Parameters
    ----------
    ha : array_like
        The CIRS hour angle of the source in radians.
    lat : array_like
        The latitude of the observatory in radians.
    dec : array_like
        The CIRS declination of the source in radians.

    Returns
    -------
    x, y, x : array_like
        The projected angular position in ground-fixed XYZ coordinates.
    """
    x = -1 * np.cos(dec) * np.sin(ha)
    y = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(ha)
    z = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)

    return x, y, z


def ground_to_sph(x, y, lat):
    """Compute CIRS coordinates from ground-fixed coordinates.

    Parameters
    ----------
    x : array_like
        The East projection of the angular position.
    y : array_like
        The North projection of the angular position.
    lat : array_like
        The latitude of the observatory in radians.

    Returns
    -------
    ha, dec : array_like
        CIRS hour angle and declination in radians.
    """
    z = np.sqrt(1 - x**2 - y**2)

    xe = z * np.cos(lat) - y * np.sin(lat)
    ye = x
    ze = y * np.cos(lat) + z * np.sin(lat)

    ha = -1 * np.arctan2(ye, xe)
    dec = np.arctan2(ze, np.sqrt(xe**2 + ye**2))

    return ha, dec


def projected_distance(ha, lat, dec, x, y, z=0.0):
    """Compute distance projected in the direction of a source.

    Parameters
    ----------
    ha : array_like
        The CIRS hour angle of the source in radians.
    lat : array_like
        The latitude of the observatory in radians.
    dec : array_like
        The CIRS declination of the source in radians.
    x : array_like
        The East-West coordinate in wavelengths (increases to the East).
    y : array_like
        The North-South coordinate in wavelengths (increases to the North).
    z : array_like
        The vertical coordinate in wavelengths (increases upwards).

    Returns
    -------
    distance : ndarray
        The projected distance. Has whatever units `x`, `y`, `z` had.
    """
    # We could use `sph_to_ground` here, but it's likely to be more memory
    # efficient to do this directly
    dist = x * (-1 * np.cos(dec) * np.sin(ha))
    dist += y * (np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(ha))
    dist += z * (np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha))

    return dist


def rotate_ypr(rot, xhat, yhat, zhat):
    """Rotate a basis by a yaw, pitch and roll.

    Parameters
    ----------
    rot : tuple
        Angles of rotation, in radians. This is a 3-tuple in order: `yaw`,
        `pitch`, `roll`.
    xhat : array_like
        X-component of the basis.  X is the axis of rotation for pitch.
    yhat : array_like
        Y-component of the basis.  Y is the axis of rotation for roll.
    zhat : array_like
        Z-component of the basis.  Z is the axis of rotation for yaw.

    Returns
    -------
    xhat, yhat, zhat : array_like
        New basis vectors.
    """
    yaw, pitch, roll = rot

    # Yaw rotation
    xhat1 = np.cos(yaw) * xhat - np.sin(yaw) * yhat
    yhat1 = np.sin(yaw) * xhat + np.cos(yaw) * yhat
    zhat1 = zhat

    # Pitch rotation
    xhat2 = xhat1
    yhat2 = np.cos(pitch) * yhat1 + np.sin(pitch) * zhat1
    zhat2 = -np.sin(pitch) * yhat1 + np.cos(pitch) * zhat1

    # Roll rotation
    xhat3 = np.cos(roll) * xhat2 - np.sin(roll) * zhat2
    yhat3 = yhat2
    zhat3 = np.sin(roll) * xhat2 + np.cos(roll) * zhat2

    return xhat3, yhat3, zhat3


def sphdist(long1, lat1, long2, lat2):
    """Return the angular distance between two coordinates on the sphere.

    Parameters
    ----------
    long1, lat1 : Angle
        longitude and latitude of the first coordinate. Each should be the
        same length; can be one or longer.

    long2, lat2 : Angle
        longitude and latitude of the second coordinate. Each should be the
        same length. If long1, lat1 have length longer than 1, long2 and
        lat2 should either have the same length as coordinate 1 or length 1.

    Returns
    -------
    angle : Angle
        Angle between the two coordinates.
    """
    dsinb = np.sin((lat1.radians - lat2.radians) / 2.0) ** 2

    dsinl = (
        np.cos(lat1.radians)
        * np.cos(lat2.radians)
        * (np.sin((long1.radians - long2.radians) / 2.0)) ** 2
    )

    dist = np.arcsin(np.sqrt(dsinl + dsinb))

    return Angle(radians=2 * dist)
