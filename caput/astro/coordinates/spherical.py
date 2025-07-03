"""Functions for dealing with spherical coordinates.

Coordinates
-----------
- :py:meth:`sphdist`
- :py:meth:`sph_to_ground`
- :py:meth:`ground_to_sph`
- :py:meth:`project_distance`
- :py:meth:`rotate_ypr`
"""

import numpy as np

from . import _spherical
from ._spherical import *  # noqa: F403

__all__ = [
    "ground_to_sph",
    "projected_distance",
    "rotate_ypr",
    "sph_to_ground",
    "sphdist",
    *_spherical.__all__,
]


def sphdist(long1, lat1, long2, lat2):
    """Return the angular distance between two coordinates on the sphere.

    Parameters
    ----------
    long1, lat1 : Skyfield Angle objects
        longitude and latitude of the first coordinate. Each should be the
        same length; can be one or longer.

    long2, lat2 : Skyfield Angle objects
        longitude and latitude of the second coordinate. Each should be the
        same length. If long1, lat1 have length longer than 1, long2 and
        lat2 should either have the same length as coordinate 1 or length 1.

    Returns
    -------
    dist : Skyfield Angle object
        angle between the two coordinates
    """
    from skyfield.units import Angle

    dsinb = np.sin((lat1.radians - lat2.radians) / 2.0) ** 2

    dsinl = (
        np.cos(lat1.radians)
        * np.cos(lat2.radians)
        * (np.sin((long1.radians - long2.radians) / 2.0)) ** 2
    )

    dist = np.arcsin(np.sqrt(dsinl + dsinb))

    return Angle(radians=2 * dist)


def sph_to_ground(ha, lat, dec):
    """Get the ground based XYZ coordinates.

    All input angles are radians. HA, DEC should be in CIRS coordinates.

    Parameters
    ----------
    ha : array_like
        The Hour Angle of the source to fringestop too.
    lat : array_like
        The latitude of the observatory.
    dec : array_like
        The declination of the source.

    Returns
    -------
    x, y, z : array_like
        The projected angular position in ground fixed XYZ coordinates.
    """
    x = -1 * np.cos(dec) * np.sin(ha)
    y = np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(ha)
    z = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)

    return x, y, z


def ground_to_sph(x, y, lat):
    """Get the CIRS coordinates.

    Latitude is given in radians. Assumes z is positive

    Parameters
    ----------
    x : array_like
        The East projection of the angular position
    y : array_like
        The North projection of the angular position
    lat : array_like
        The latitude of the observatory.

    Returns
    -------
    ha, dec: array_like
        Hour Angle and declination in radians
    """
    z = np.sqrt(1 - x**2 - y**2)

    xe = z * np.cos(lat) - y * np.sin(lat)
    ye = x
    ze = y * np.cos(lat) + z * np.sin(lat)

    ha = -1 * np.arctan2(ye, xe)
    dec = np.arctan2(ze, np.sqrt(xe**2 + ye**2))

    return ha, dec


def projected_distance(ha, lat, dec, x, y, z=0.0):
    """Return the distance project in the direction of a source.

    Parameters
    ----------
    ha : array_like
        The Hour Angle of the source to fringestop too.
    lat : array_like
        The latitude of the observatory.
    dec : array_like
        The declination of the source.
    x : array_like
        The EW coordinate in wavelengths (increases to the E)
    y : array_like
        The NS coordinate in wavelengths (increases to the N)
    z : array_like, optional
        The vertical coordinate on wavelengths (increases to the sky!)

    Returns
    -------
    dist : np.ndarray
        The projected distance. Has whatever units x, y, z did.
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
    rot : [yaw, pitch, roll]
        Angles of rotation, in radians.
    xhat: np.ndarray
        X-component of the basis.  X is the axis of rotation for pitch.
    yhat: np.ndarray
        Y-component of the basis.  Y is the axis of rotation for roll.
    zhat: np.ndarray
        Z-component of the basis.  Z is the axis of rotation for yaw.

    Returns
    -------
    xhat, yhat, zhat : np.ndarray[3]
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
