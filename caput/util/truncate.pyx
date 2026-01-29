# cython: language_level=3
"""Routines for truncating data to a specified precision."""

cimport cython
from cython.parallel import prange

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs

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


__all__ = [
    "bit_truncate_float",
    "bit_truncate_double",
    "bit_truncate_weights",
    "bit_truncate_relative",
    "bit_truncate_max_complex",
]


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
    if err != err:
        raise ValueError(f"Error {err} is invalid.")

    return _bit_truncate_float(val, err)


def bit_truncate_double(double val, double err):
    if err != err:
        raise ValueError(f"Error {err} is invalid.")

    return _bit_truncate_double(val, err)


def bit_truncate_weights(val, inv_var, fallback):
    if val.dtype == np.float32 and inv_var.dtype == np.float32:
        return _bit_truncate_weights_float(val, inv_var, fallback)
    if val.dtype == np.float64 and inv_var.dtype == np.float64:
        return _bit_truncate_weights_double(val, inv_var, fallback)
    else:
        raise RuntimeError(
            f"Can't truncate data of type {val.dtype}/{inv_var.dtype} "
            "(expected float32 or float64)."
        )

@cython.boundscheck(False)
@cython.wraparound(False)
def _bit_truncate_weights_float(float[:] val, float[:] inv_var, float fallback):
    """Truncate array of floats using a set of inverse variance weights."""
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
def _bit_truncate_weights_double(double[:] val, double[:] inv_var, double fallback):
    """Truncate array of doubles using a set of inverse variance weights."""
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
        return _bit_truncate_relative_float(val, prec)
    if val.dtype == np.float64:
        return _bit_truncate_relative_double(val, prec)
    else:
        raise RuntimeError(
            f"Can't truncate data of type {val.dtype} (expected float32 or float64)."
        )

@cython.boundscheck(False)
@cython.wraparound(False)
def _bit_truncate_relative_float(float[:] val, float prec):
    """Truncate array of floats using a relative tolerance."""
    cdef Py_ssize_t n = val.shape[0]
    cdef Py_ssize_t i = 0

    for i in prange(n, nogil=True):
        val[i] = _bit_truncate_float(val[i], fabs(prec * val[i]))

    return np.asarray(val)

@cython.boundscheck(False)
@cython.wraparound(False)
def _bit_truncate_relative_double(cnp.float64_t[:] val, cnp.float64_t prec):
    """Truncate array of doubles using a relative tolerance."""
    cdef Py_ssize_t n = val.shape[0]
    cdef Py_ssize_t i = 0

    for i in prange(n, nogil=True):
        val[i] = _bit_truncate_double(val[i], fabs(prec * val[i]))

    return np.asarray(val, dtype=np.float64)


@cython.boundscheck(False)
@cython.wraparound(False)
def bit_truncate_max_complex(complex128[:, :] val, float prec, float prec_max_row):
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
