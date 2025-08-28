import numpy as np
import numpy.typing as npt

__all__ = [
    "cart_to_sph",
    "cosine_rule",
    "great_circle_points",
    "norm_vec2",
    "sph_dot",
    "sph_to_cart",
    "thetaphi_plane",
    "thetaphi_plane_cart",
]

#
def sph_to_cart(
    sph_coords: npt.ArrayLike[np.integer | np.floating],
) -> np.ndarray[np.float64]:
    """Convert a vector in spherical polar coordinates to Cartesian coordinates.

    This routine is OpenMP-parallel.

    Parameters
    ----------
    sph_coords : array_like
        A vector (or array) of spherical polar coordinates. Values should be
        packed as [r, theta, phi] along the last axis. Alternatively they can be
        packed as [theta, phi] in which case r is assumed to be one.

    Returns
    -------
    cart_vector : array_like
        Array of equivalent vectors in cartesian coordinartes.
    """
    ...

#
def cart_to_sph(
    cart_arr: npt.ArrayLike[np.integer | np.floating],
) -> np.ndarray[np.floating]:
    """Convert a vector of Cartesian coordinates into spherical polar coordinates.

    Uses the same convention as `sph_to_cart`.

    Parameters
    ----------
    cart_arr : array_like
        Vector (or array) of cartesian coordinates.

    Returns
    -------
    polar_vector : array_like
        Array of spherical polars (packed as [[ r1, theta1, phi1], [r2, theta2,
        phi2], ...]
    """
    ...

#
def sph_dot(
    arr1: npt.ArrayLike[np.integer | np.floating],
    arr2: npt.ArrayLike[np.integer | np.floating],
) -> np.ndarray[np.floating]:
    """Take the scalar product in spherical polar coordinates.

    Parameters
    ----------
    arr1, arr2 : array_like
        Two arrays of vectors in spherical polar coordinates ``[r, theta, phi]``,
        (or alternatively as ``[theta, phi]`` with `r` assumed to be 1). The two arrays
        should be broadcastable against each other.

    Returns
    -------
    dotted_vectors : array_like
        An array of the scalar products.
    """
    ...

#
def thetaphi_plane(
    sph_arr: npt.ArrayLike[np.integer | np.floating],
) -> tuple[np.ndarray[np.floating], np.ndarray[np.floating]]:
    """Compute unit vectors from spherical polar coordinate positions.

    Parameters
    ----------
    sph_arr : array_like
        Angular positions (in spherical polar coordinates).

    Returns
    -------
    theta : array_like
        `theta` unit vector.
    phi : array_like
        `phi` unit vector.
    """
    ...

#
def thetaphi_plane_cart(
    sph_coords: npt.ArrayLike[np.integer | np.floating],
) -> np.ndarray[np.float64]:
    """Convert unit vectors in spherical polar coordinates to Cartesian coordinates.

    This routine is OpenMP-parallel.

    Parameters
    ----------
    sph_coords : array_like
        Angular unit vectors (in spherical polar coordinates) packed
        as ``[r, theta, phi]`` along the last axis.

    Returns
    -------
    cert_vector : array_like
        Unit vectors in the theta and phi directions now in cartesian coordinates.
    """
    ...

#
def norm_vec2(vec2: npt.NDArray[np.floating]) -> None:
    """For an array of 2D vectors, normalise each to unit length *in-place*.

    Parameters
    ----------
    vec2 : ndarray
        An array of 2D vectors, which will be updated in-place.
    """
    ...

#
def great_circle_points(
    sph1: npt.ArrayLike[np.integer | np.floating],
    sph2: npt.ArrayLike[np.integer | np.floating],
    npoints: int,
) -> np.ndarray[np.floating]:
    """Compute intermediate points on the great circle between endpoints..

    Parameters
    ----------
    sph1, sph2 : array_like
        Endpoints on sphere, packed as [theta, phi] along the last axis. The
        endpoints may not be antipodal (since there's no unique great circle
        in that case).
    npoints : int
        Number of intermediate points to compute.

    Returns
    -------
    points : ndarray
        Intermediate points, packed as ``[theta1, phi1]`` along the last axis.
    """
    ...

#
def cosine_rule(
    mu: npt.ArrayLike[np.float64],
    x1: npt.ArrayLike[np.float64],
    x2: npt.ArrayLike[np.float64],
) -> np.ndarray[np.float64]:
    """Calculate the distances between a grid of points.

    This is a somewhat niche implementation intended to help calculate multi-distance
    angular power spectra.

    Parameters
    ----------
    mu : array_like
        Angular separations in ``cos(theta)``.
    x1, x2 : array_like
        Positions of the first and second points.

    Returns
    -------
    separation : array_like
        The separation of the points for all combinations of mu, x1, x2. This is
        a 3-dimensional array with axis-lengths equal to the lengths of the
        three input vectors.
    """
    ...
