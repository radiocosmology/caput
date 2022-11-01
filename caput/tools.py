""" Collection of assorted tools.
"""

import numpy as np
from caput._fast_tools import _invert_no_zero


def invert_no_zero(x, out=None):
    """Return the reciprocal, but ignoring zeros.

    Where `x != 0` return 1/x, or just return 0. Importantly this routine does
    not produce a warning about zero division.

    Parameters
    ----------
    x : np.ndarray
    out : np.ndarray

    Returns
    -------
    r : np.ndarray
        Return the reciprocal of x. Where possible the output has the same memory layout
        as the input, if this cannot be preserved the output is C-contiguous.
    """
    if not isinstance(x, (np.generic, np.ndarray)) or np.issubdtype(
        x.dtype, np.integer
    ):
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
