import numpy as np

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
cpdef _invert_no_zero(real_or_complex [:] array, real_or_complex [:] out):

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


def invert_no_zero(x, out=None):
    """Return the reciprocal, but ignoring zeros.

    Where `x != 0` return 1/x, or just return 0. Importantly this routine does
    not produce a warning about zero division.

    Parameters
    ----------
    x : np.ndarray
        Array to invert
    out : np.ndarray, optional
        Output array to insert results

    Returns
    -------
    r : np.ndarray
        Return the reciprocal of x. Where possible the output has the same memory layout
        as the input, if this cannot be preserved the output is C-contiguous.
    """
    if not isinstance(x, np.generic | np.ndarray) or np.issubdtype(x.dtype, np.integer):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.where(x == 0, 0.0, 1.0 / x)

    if out is not None:
        if x.shape != out.shape:
            raise ValueError(
                f"Input and output arrays don't have same shape: {x.shape} != {out.shape}."
            )
    else:
        # This works even for MPIArrays, producing a correctly shaped MPIArray
        out = np.empty_like(x, order="A")

    # In order to be able to flatten the arrays to do element by element operations, we
    # need to ensure the inputs are numpy arrays, and so we take a view which will work
    # even if `x` (and thus `out`) are MPIArray's
    _invert_no_zero(
        x.view(np.ndarray).ravel(order="A"), out.view(np.ndarray).ravel(order="A")
    )

    return out
