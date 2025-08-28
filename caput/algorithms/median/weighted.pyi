from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["moving_weighted_median", "quantile", "weighted_median"]

#
def quantile(
    A: npt.NDArray[np.int_ | np.float64],
    W: npt.NDArray[np.int_ | np.float64],
    q: float,
    method: Literal["lower", "higher", "split"] = "split",
) -> np.ndarray[np.floating | np.integer] | np.integer | np.floating:
    """Calculate the weighted quantile of a set of data.

    The weighted quantile is always calculated along the last axis.

    The weights must be postive or zero for the calculation to make sense. This is
    not checked within this routine, and so you must sanitize the input before
    calling.

    In the case that all elements have zero weight, a standard uniformly weighted
    quantile calculation is performed. If there is one, and only one non-zero
    weighted element that is always returned. For two or more non-zero weighted
    elements, we can proceed as expected.

    If the quantile is "split", i.e. it lies exactly on the boundary between two
    elements, we use the bounding non-zero weighted elements to calculate the
    quantile according to the chosen method.

    Examples of special cases:

    >>> quantile([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 2], 0.5)
    3.0
    >>> quantile([1.0, 2.0, 3.0, 4.0], [1, 1, 0, 2], 0.5)
    3.0
    >>> quantile([1.0, 2.0, 4.0, 3.0], [0, 0, 0, 0], 0.5)
    2.5

    Parameters
    ----------
    A : ndarray[int | float]
        The array of data. Only 32 and 64 bit integers and floats are supported.
    W : ndarray[int | float]
        The array of weights. Only 32 and 64 bit integers and floats are supported.
    q : float
        The quantile as a number from zero to one.
    method : {"lower", "higher", "split"}
        Method to use if the requested quantile is exactly between two elements. Must
        be one of "lower", "higher" or "split". Default is "split".

    Returns
    -------
    quantile : ndarray
        The calculated quantile. This has the same shape as A with the last axis
        removed. If A was one dimensional the value returned is a scalar.

    Raises
    ------
    ValueError
        If the array shapes do not match, or the quantile value is invalid.
    TypeError
        If the array types are not supported.
    """
    ...

#
def weighted_median(
    A: npt.NDArray[np.int_ | np.float64],
    W: npt.NDArray[np.int_ | np.float64],
    method: Literal["lower", "higher", "split"] = "split",
) -> np.ndarray[np.floating | np.integer] | np.integer | np.floating:
    """Calculate the weighted median of a set of data.

    The weighted median is always calculated along the last axis.

    See `quantile` for more information on the behaviour for some special cases.

    Parameters
    ----------
    A : ndarray
        The array of data.
    W : ndarray
        The array of weights.
    method : {"lower", "higher", "split"}
        Method to use if the requested quantile is exactly between two elements. Must
        be one of "lower", "higher" or "split". Default is "split".

    Returns
    -------
    median : array_like
        The calculated median. This has the same shape as A with the last axis
        removed. If A was one dimensional the value returned is a scalar.
    """
    ...

#
def moving_weighted_median(
    data: npt.ArrayLike[np.float64],
    weights: npt.ArrayLike[np.float64],
    size: int | tuple[int, int],
    method: Literal["lower", "higher", "split"] = "split",
) -> np.ndarray[np.float64]:
    """Compute moving weighted median for 1 and 2 dimensional arrays.

    Parameters
    ----------
    data : array_like
        The data to move the window over. Can have 1 or 2 dimensions. The data type should be
        float64 or something that can be converted to float64.
    weights : array_like
        The weights for the data. Can have 1 or 2 dimensions. The data type should be
        float64 or something that can be converted to float64.
    size : int | tuple[int, int]
        Size of the window. All values must be uneven.
    method : {"lower", "higher", "split"}
        Either 'split', 'lower' or 'higher'. If multiple values sastisfy the conditions to be the
        weighted median of a window, this decides what is returned:

        - `split`: The average of all candidate values is returned.
        - `lower`: The lowest of all candidate values is returned.
        - `higher`: The highest of all candidate values is returned.

        Default is "split".

    Returns
    -------
    median : array_like
        An array containing the weighted median values.
        The size is the same as the given data and weights.

    Raises
    ------
    ValueError
        If the value of the window size was not odd.
    RuntimeError
        If there was an internal error in the C++ implementation.
    NotImplementedError
        If the data has more than two dimensions.
    """
    ...
