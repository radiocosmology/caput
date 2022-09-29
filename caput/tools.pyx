from cython.parallel import prange
cimport cython

from libc.math cimport fabs

cdef extern from "float.h" nogil:
    double DBL_MAX
    double FLT_MAX

ctypedef fused real_or_complex:
    double
    double complex
    float
    float complex

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cpdef invert_no_zero(real_or_complex [:] array, real_or_complex [:] out):

    cdef bint cond
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t n = array.shape[0]
    cdef double thresh, ar, ai
    if (real_or_complex is cython.doublecomplex) or (real_or_complex is cython.double):
        thresh = 1.0 / DBL_MAX
    else:
        thresh = 1.0 / FLT_MAX

    if (real_or_complex is cython.doublecomplex) or (real_or_complex is cython.floatcomplex):
        for i in prange(n, nogil=True):
            cond = (fabs(array[i].real) < thresh) and (fabs(array[i].imag) < thresh)
            out[i] = 0.0 if cond else 1.0 / array[i]
    else:
        for i in prange(n, nogil=True):
            cond = fabs(array[i]) < thresh
            out[i] = 0.0 if cond else 1.0 / array[i]