"""Routines for truncating data to a specified precision."""

# cython: language_level=3

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as cnp

cdef extern from "truncate.hpp":
    inline int bit_truncate(int val, int err) nogil

cdef extern from "truncate.hpp":
    inline long bit_truncate_64(long val, long err) nogil

cdef extern from "truncate.hpp":
    inline float _bit_truncate_float(float val, float err) nogil


cdef extern from "truncate.hpp":
    inline double _bit_truncate_double(double val, double err) nogil

ctypedef double complex complex128

cdef extern from "complex.h" nogil:
    double cabs(complex128)


def bit_truncate_int(int val, int err):
    """
    Bit truncation of a 32bit integer.

    Truncate the precision of `val` by rounding to a multiple of a power of
    two, keeping error less than or equal to `err`.

    Made available for testing.
    """
    return bit_truncate(val, err)

def bit_truncate_long(long val, long err):
    """
    Bit truncation of a 64bit integer.

    Truncate the precision of `val` by rounding to a multiple of a power of
    two, keeping error less than or equal to `err`.

    Made available for testing.
    """
    return bit_truncate_64(val, err)


def bit_truncate_float(float val, float err):
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

    return _bit_truncate_float(val, err)


def bit_truncate_double(double val, double err):
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

    return _bit_truncate_double(val, err)


def bit_truncate_weights(val, inv_var, fallback):
    if val.dtype == np.float32 and inv_var.dtype == np.float32:
        return bit_truncate_weights_float(val, inv_var, fallback)
    if val.dtype == np.float64 and inv_var.dtype == np.float64:
        return bit_truncate_weights_double(val, inv_var, fallback)
    else:
        raise RuntimeError(f"Can't truncate data of type {val.dtype}/{inv_var.dtype} "
                           f"(expected float32 or float64).")


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_weights_float(float[:] val, float[:] inv_var, float fallback):
    """Truncate using a set of inverse variance weights.

    Giving the error as an inverse variance is particularly useful for data analysis.

    N.B. non-contiguous arrays are supported in order to allow real and imaginary parts
    of numpy arrays to be truncated without making a copy.

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
            val[i] = _bit_truncate_float(val[i], 1.0 / inv_var[i]**0.5)
        else:
            val[i] = _bit_truncate_float(val[i], fallback * val[i])

    return np.asarray(val)

@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_weights_double(double[:] val, double[:] inv_var, double fallback):
    """Truncate array of doubles using a set of inverse variance weights.

    Giving the error as an inverse variance is particularly useful for data analysis.

    N.B. non-contiguous arrays are supported in order to allow real and imaginary parts
    of numpy arrays to be truncated without making a copy.

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
            val[i] = _bit_truncate_double(val[i], 1.0 / inv_var[i]**0.5)
        else:
            val[i] = _bit_truncate_double(val[i], fallback * val[i])

    return np.asarray(val)

def bit_truncate_relative(val, prec):
    if val.dtype == np.float32:
        return bit_truncate_relative_float(val, prec)
    if val.dtype == np.float64:
        return bit_truncate_relative_double(val, prec)
    else:
        raise RuntimeError(f"Can't truncate data of type {val.dtype} (expected float32 or float64).")


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_relative_float(float[:] val, float prec):
    """Truncate using a relative tolerance.

    N.B. non-contiguous arrays are supported in order to allow real and imaginary parts
    of numpy arrays to be truncated without making a copy.

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
        val[i] = _bit_truncate_float(val[i], prec * val[i])

    return np.asarray(val)


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_relative_double(cnp.float64_t[:] val, cnp.float64_t prec):
    """Truncate doubles using a relative tolerance.

    N.B. non-contiguous arrays are supported in order to allow real and imaginary parts
    of numpy arrays to be truncated without making a copy.

    Parameters
    ----------
    val
        The array of double values to truncate the precision of. These values are modified in place.
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
        val[i] = _bit_truncate_double(val[i], prec * val[i])

    return np.asarray(val, dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_max_complex(complex128[:, :] val, float prec, float prec_max_row):
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
            val[i, j].real = _bit_truncate_float(<float>vr, abs_prec)
            val[i, j].imag = _bit_truncate_float(<float>vi, abs_prec)

    return np.asarray(val)
