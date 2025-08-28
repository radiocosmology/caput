"""Routines for truncating data to a specified precision.

Provides absolute and relative bit truncation routines for integers,
floating point numbers, and numpy arrays.
"""

from typing import overload

import numpy as np

__all__ = [
    "bit_truncate_double",
    "bit_truncate_float",
    "bit_truncate_max_complex",
    "bit_truncate_relative",
    "bit_truncate_weights",
]

#
def bit_truncate_float(val: np.float32, err: np.float32) -> np.float32:
    """Truncate using a fixed error.

    Parameters
    ----------
    val : float32
        The value to truncate.
    err : float32
        The absolute precision to allow.

    Returns
    -------
    truncated : float32
        The truncated value.

    Raises
    ------
    ValueError
        If `err` is a NaN.
    """
    ...

#
def bit_truncate_double(val: np.float64, err: np.float64) -> np.float64:
    """Truncate using a fixed error.

    Parameters
    ----------
    val : float
        The value to truncate.
    err : float
        The absolute precision to allow.

    Returns
    -------
    truncated : float
        The truncated value.

    Raises
    ------
    ValueError
        If `err` is a NaN.
    """
    ...

#
@overload
def bit_truncate_weights(
    val: np.ndarray[np.float32], inv_var: np.ndarray[np.float32], fallback: np.float32
) -> np.ndarray[np.float32]: ...
@overload
def bit_truncate_weights(
    val: np.ndarray[np.float64], inv_var: np.ndarray[np.float64], fallback: np.float64
) -> np.ndarray[np.float64]: ...
def bit_truncate_weights(val, inv_var, fallback):
    """Truncate using a set of inverse variance weights.

    Giving the error as an inverse variance is particularly useful for data analysis.

    N.B. non-contiguous arrays are supported in order to allow real and imaginary parts
    of numpy arrays to be truncated without making a copy.

    Parameters
    ----------
    val : array_like
        The array of values to truncate the precision of. These values are modified in place.
    inv_var : array_like
        The acceptable precision expressed as an inverse variance.
    fallback : array_like
        A relative precision to use for cases where the inv_var is zero.

    Returns
    -------
    truncated : ndarray
        The modified array. This shares the same underlying memory as the input.
    """
    ...

#
@overload
def bit_truncate_relative(
    val: np.ndarray[np.float32], prec: np.float32
) -> np.ndarray[np.float32]: ...
@overload
def bit_truncate_relative(
    val: np.ndarray[np.float64], prec: np.float64
) -> np.ndarray[np.float64]: ...
def bit_truncate_relative(val, prec):
    """Truncate using a relative tolerance.

    N.B. non-contiguous arrays are supported in order to allow real and imaginary parts
    of numpy arrays to be truncated without making a copy.

    Parameters
    ----------
    val : array_like
        The array of values to truncate the precision of. These values are modified in place.
    prec : float
        The fractional precision required.

    Returns
    -------
    truncated : array_like
        The modified array. This shares the same underlying memory as the input.
    """
    ...

#
def bit_truncate_max_complex(
    val: np.ndarray[np.complex128], prec: np.float32, prec_max_row: np.float32
) -> np.ndarray[np.complex128]:
    """Truncate using a relative per element and per the maximum of the last dimension.

    This scheme allows elements to be truncated based on their own value and a
    measure of their relative importance compared to other elements. In practice the
    per element absolute precision for an element `val[i, j]` is given by `max(prec *
    val[i, j], prec_max_dim * val[i].max())`

    Parameters
    ----------
    val : complex array_like
        The array of values to truncate the precision of. These values are modified in place.
    prec : float
        The fractional precision on each elements.
    prec_max_row : float
        The precision to use relative to the maximum of the of each row.

    Returns
    -------
    truncated : complex array_like
        The modified array. This shares the same underlying memory as the input.
    """
    ...
