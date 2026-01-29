"""Spherical co-ordinate utilities

A module of useful functions when dealing with Spherical Polar co-ordinates.
"""

cimport cython
from cython.parallel cimport prange

import numpy as np
import numpy.typing as npt

from libc.math cimport sin, cos, hypot, M_PI_2


def sph_to_cart(sph_coords):
    cdef double[:, ::1] sph_view, cart_view
    cdef Py_ssize_t i, n, nd
    cdef double r

    if not isinstance(sph_coords, np.ndarray):
        raise ValueError("Argument must be a numpy array.")

    nd = sph_coords.shape[-1]

    # Make sure we are dealing with double precision types
    sph_coords = np.ascontiguousarray(sph_coords).astype(np.float64, copy=False)
    sph_view = sph_coords.reshape(-1, nd)

    # Create an array of the correct size for the output.
    cart_coords = np.empty(sph_coords.shape[:-1] + (3,), dtype=np.float64)
    cart_view = cart_coords.reshape(-1, 3)

    n = sph_view.shape[0]

    with cython.wraparound(False), cython.boundscheck(False):

        if nd == 2:
            for i in prange(n, nogil=True):
                _s2c(sph_view[i, 0], sph_view[i, 1], &cart_view[i, 0])

        elif nd == 3:
            for i in prange(n, nogil=True):
                _s2c(sph_view[i, 1], sph_view[i, 2], &cart_view[i, 0])
                r = sph_view[i, 0]
                cart_view[i, 0] *= r
                cart_view[i, 1] *= r
                cart_view[i, 2] *= r

        else:
            raise ValueError(
                f"Last axis of `sph_view` has unsupported length {nd}."
            )


    return cart_coords


def cart_to_sph(cart_arr):
    sph_arr = np.empty_like(cart_arr)

    sph_arr[..., 2] = np.arctan2(cart_arr[..., 1], cart_arr[..., 0])

    sph_arr[..., 0] = np.sum(cart_arr ** 2, axis=-1) ** 0.5

    sph_arr[..., 1] = np.arccos(cart_arr[..., 2] / sph_arr[..., 0])

    return sph_arr


def sph_dot(arr1, arr2):
    return np.inner(sph_to_cart(arr1), sph_to_cart(arr2))


def thetaphi_plane(sph_arr):
    thetahat = sph_arr.copy()
    thetahat[..., -2] += np.pi / 2.0

    phihat = sph_arr.copy()
    phihat[..., -2] = np.pi / 2.0
    phihat[..., -1] += np.pi / 2.0

    if sph_arr.shape[-1] == 3:
        thetahat[..., 0] = 1.0
        phihat[..., 0] = 1.0

    return thetahat, phihat


def thetaphi_plane_cart(sph_coords):
    
    cdef double[:, ::1] sph_view
    cdef double[:, :, ::1] tp_view
    cdef Py_ssize_t i, n
    cdef double theta, phi

    if not isinstance(sph_coords, np.ndarray) or sph_coords.shape[-1] != 2:
        raise ValueError("Argument must be a numpy array with last axis of length 2.")

    sph_coords = np.ascontiguousarray(sph_coords).astype(np.float64, copy=False)
    sph_view = sph_coords.reshape(-1, 2)

    # Create an array of the correct size for the output.
    tp_cart = np.empty((2,) + sph_coords.shape[:-1] + (3,), dtype=np.float64)
    tp_view = tp_cart.reshape(2, -1, 3)

    n = sph_view.shape[0]

    with cython.wraparound(False), cython.boundscheck(False):
        for i in prange(n, nogil=True):

            theta = sph_view[i, 0]
            phi = sph_view[i, 1]

            _s2c(theta + M_PI_2, phi, &tp_view[0, i, 0])
            _s2c(M_PI_2, phi + M_PI_2, &tp_view[1, i, 0])

    return tp_cart


def norm_vec2(vec2):

    cdef double[:, ::1] vec_view
    cdef Py_ssize_t i, n
    cdef double norm

    if not isinstance(vec2, np.ndarray) or vec2.shape[-1] != 2:
        raise ValueError("Argument must be a numpy array with last axis of length 2.")

    vec_view = vec2.reshape(-1, 2)

    n = vec_view.shape[0]

    with cython.wraparound(False), cython.boundscheck(False):
        for i in prange(n, nogil=True):

            norm = hypot(vec_view[i, 0], vec_view[i, 1])
            vec_view[i, 0] /= norm
            vec_view[i, 1] /= norm


def great_circle_points(sph1, sph2, npoints):

    # Turn points on circle into Cartesian vectors
    c1 = sph_to_cart(sph1)
    c2 = sph_to_cart(sph2)

    # Construct the difference vector
    dc = c2 - c1
    if np.sum(dc ** 2) == 4.0:
        raise Exception("Points antipodal, path undefined.")

    # Create difference unit vector
    dcn = dc / np.dot(dc, dc) ** 0.5

    # Calculate central angle, and angle on corner O C1 C2
    thc = np.arccos(np.dot(c1, c2))
    alp = np.arccos(-np.dot(c1, dcn))

    # Map regular angular intervals into distance along the displacement between
    # C1 and C2
    f = np.linspace(0.0, 1.0, npoints)
    x = np.sin(f * thc) / np.sin(f * thc + alp)

    # Generate the Cartesian vectors for intermediate points
    cv = x[:, np.newaxis] * dcn[np.newaxis, :] + c1[np.newaxis, :]

    # Return the angular poitions of the points
    return cart_to_sph(cv)[:, 1:]


@cython.boundscheck(False)
@cython.wraparound(False)
def cosine_rule(double[::1] mu, double[::1] x1, double[::1] x2):

    cdef int nm = len(mu)
    cdef int n1 = len(x1)
    cdef int n2 = len(x2)

    cdef double[:, :, ::1] rview

    r = np.zeros((nm, n1, n2), dtype=np.float64)

    rview = r

    cdef int i, j, k

    for i in prange(nm, nogil=True):
        for j in range(n1):
            for k in range(n2):
                rview[i, j, k] = ((x1[j] - x2[k]) ** 2 + 2 * x1[j] * x2[k] * (1 - mu[i])) ** 0.5

    return r


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _s2c(double theta, double phi, double cart[3]) nogil:
    # Worker function for calculating the cartesian version of a unit vector in
    # spherical polars
    cdef double sintheta = sin(theta)
    cart[0] = sintheta * cos(phi)
    cart[1] = sintheta * sin(phi)
    cart[2] = cos(theta)