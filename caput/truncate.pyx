"""Routines for truncating data to a specified precision."""

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as cnp


cdef extern from "truncate.hpp":
    inline float bit_truncate_float(float val, float err) nogil


ctypedef double complex complex128

cdef extern from "complex.h" nogil:
    double cabs(complex128)


def bit_truncate(float val, float err):
    """Truncate using a fixed error.

    Parameters
    ----------
    val
        The value to truncate.
    err
        The absolute precision to allow.

    Returns
    -------
    val
        The truncated value.
    """

    return bit_truncate_float(val, err)


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_weights(float[::1] val, float[::1] inv_var, float fallback):
    """Truncate using a set of inverse variance weights.

    Giving the error as an inverse variance is particularly useful for data analysis.

    Parameters
    ----------
    val
        The array of values to truncate the precision of. These values are modified in place.
    inv_var
        The acceptable precision expressed as an inverse variance.
    fallback
        A relative precision to use for cases where the inv_var is zero.

    Returns
    -------
    val
        The modified array. This shares the same underlying memory as the input.
    """
    cdef Py_ssize_t n = val.shape[0]
    cdef Py_ssize_t i = 0

    if val.ndim != 1:
        raise ValueError("Input array must be 1-d.")
    if inv_var.shape[0] != n:
        raise ValueError(
            f"Weight and value arrays must have same shape ({inv_var.shape[0]} != {n})"
        )

    for i in prange(n, nogil=True):
        if inv_var[i] != 0:
            val[i] = bit_truncate_float(val[i], 1.0 / inv_var[i]**0.5)
        else:
            val[i] = bit_truncate_float(val[i], fallback * val[i])

    return np.asarray(val)


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_relative(float[::1] val, float prec):
    """Truncate using a relative tolerance.

    Parameters
    ----------
    val
        The array of values to truncate the precision of. These values are modified in place.
    prec
        The fractional precision required.

    Returns
    -------
    val
        The modified array. This shares the same underlying memory as the input.
    """
    cdef Py_ssize_t n = val.shape[0]
    cdef Py_ssize_t i = 0

    for i in prange(n, nogil=True):
        val[i] = bit_truncate_float(val[i], prec * val[i])

    return np.asarray(val)


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_max_complex(complex128[:, ::1] val, float prec, float prec_max_row):
    """Truncate using a relative per element and per the maximum of the last dimension.

    This scheme allows elements to be truncated based on their own value and a
    measure of their relative importance compared to other elements. In practice the
    per element absolute precision for an element `val[i, j]` is given by `max(prec *
    val[i, j], prec_max_dim * val[i].max())`

    Parameters
    ----------
    val
        The array of values to truncate the precision of. These values are modified in place.
    prec
        The fractional precision on each elements.
    prec_max_row
        The precision to use relative to the maximum of the of each row.

    Returns
    -------
    val
        The modified array. This shares the same underlying memory as the input.
    """
    cdef Py_ssize_t n = val.shape[0]
    cdef Py_ssize_t m = val.shape[1]
    cdef Py_ssize_t i = 0, j = 0
    cdef float abs_prec
    cdef double vr, vi
    cdef double max_abs
    cdef double abs2

    for i in prange(n, nogil=True):

        max_abs = 0.0

        # Find the largest abs**2 value in the row, store in max_abs, but note that it is the *square*
        for j in range(m):
            vr = val[i, j].real
            vi = val[i, j].imag
            abs2 = vr * vr + vi * vi

            if abs2 > max_abs:
                max_abs = abs2

        max_abs = max_abs**0.5

        for j in range(m):
            # Get the precision to apply
            abs_prec = max(<float>cabs(val[i, j]) * prec, prec_max_row * max_abs)

            vr = val[i, j].real
            vi = val[i, j].imag
            val[i, j].real = bit_truncate_float(<float>vr, abs_prec)
            val[i, j].imag = bit_truncate_float(<float>vi, abs_prec)

    return np.asarray(val)