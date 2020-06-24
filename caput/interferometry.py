"""Useful functions for radio interferometry.

Coordinates
===========

.. autosummary::
    :toctree: generated/

    sph_to_ground
    ground_to_sph
    project_distance


Interferometry
==============

.. autosummary::
    :toctree: generated/

    fringestop_phase
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np


def sph_to_ground(ha, lat, dec):
    """Get the ground based XYZ coordinates.

    All input angles are radians. HA, DEC should be in CIRS coordinates.

    Parameter
    ---------
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

    Parameter
    ---------
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

    Parameter
    ---------
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


def fringestop_phase(ha, lat, dec, u, v, w=0.0):
    """Return the phase required to fringestop. All angle inputs are radians.

    Note that for a visibility V_{ij} = < E_i E_j^*>, this expects the u, v,
    w coordinates are the components of (d_i - d_j) / lambda.

    Parameter
    ---------
    ha : array_like
        The Hour Angle of the source to fringestop too.
    lat : array_like
        The latitude of the observatory.
    dec : array_like
        The declination of the source.
    u : array_like
        The EW separation in wavelengths (increases to the E)
    v : array_like
        The NS separation in wavelengths (increases to the N)
    w : array_like, optional
        The vertical separation on wavelengths (increases to the sky!)

    Returns
    -------
    phase : np.ndarray
        The phase required to *correct* the fringeing. Shape is
        given by the broadcast of the arguments together.
    """

    phase = -2.0j * np.pi * projected_distance(ha, lat, dec, u, v, w)

    return np.exp(phase, out=phase)
