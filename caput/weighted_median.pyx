"""Weighted Median Functions"""

# distutils: language = c++
# cython: language_level = 2
import numpy as np

from libcpp.memory cimport shared_ptr
from libcpp.deque cimport deque
from cython.parallel import prange, parallel
from MedianTree cimport Tree, Data
cimport numpy as np
cimport cython


def _check_arrays(data, weights):

    # make sure this is numpy arrays
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    if not isinstance(weights, np.ndarray):
        weights = np.array(weights, dtype=np.float64)

    if data.dtype != np.dtype(np.float64):
        raise ValueError('Expected data to be numpy.float64 (got {}).'.format(data.dtype))
    if weights.dtype != np.dtype(np.float64):
        raise ValueError('Expected weights to be numpy.float64 (got {}).'
                         .format(weights.dtype))
    if data.ndim != weights.ndim:
        raise ValueError('Expected data and weights to have same dimensions (is {} and {}).'
                         .format(data.ndim, weights.ndim))

    return data, weights


def weighted_median(data, weights, method="split"):
    """Compute weighted median for 1 and 2 dimensional arrays.

    Parameters
    ----------
    data : array_like
        The data to compute weighted median for. Can have 1 or 2 dimensions. The data type should
        be float64 or something that can be converted to float64.
    weights : array_like
        The weights for the data. Can have 1 or 2 dimensions. The data type should be
        float64 or something that can be converted to float64.
    method : str
        Either 'split', 'lower' or 'higher'. If multiple values sastisfy the conditions to be the
        weighted median of a window, this decides what is returned:
        split: The average of all candidate values is returned.
        lower: The lowest of all candidate values is returned.
        higher: The highest of all candidate values is returned.

    Returns
    -------
    float64
        The weighted median.

    Raises
    ------
    NotImplementedError
        If the number of dimensions is not 1 or 2.
    RuntimeError
        If there was an internal error in the C++ implementation.
    """
    data, weights = _check_arrays(data, weights)
    cdef char c_method = _check_method(method)

    if data.ndim is 1:
        return _weighted_median_1D(data, weights, c_method)
    if data.ndim is 2:
        return _weighted_median_2D(data, weights, c_method)
    raise NotImplementedError('weighted_median is only implemented for 1 and 2 dimensions, not {}'
                              .format(data.ndim))


def _weighted_median_1D(np.ndarray[np.float64_t, ndim=1] data,
                        np.ndarray[np.float64_t, ndim=1] weights, method):
    cdef Tree[double] avl = Tree[double]()

    for d, w in zip(data, weights):
        if w != 0:
            avl.insert(d, w)

    return avl.weighted_median(method)


def _weighted_median_2D(np.ndarray[np.float64_t, ndim=2] data,
                        np.ndarray[np.float64_t, ndim=2] weights, method):
    cdef Tree[double] avl = Tree[double]()

    for d, w in zip(np.nditer(data), np.nditer(weights)):
        if w != 0:
            avl.insert(d, w)

    return avl.weighted_median(method)


METHODS = ['split', 'lower', 'higher']
METHODS_C = ['s', 'l', 'h']


def _check_method(method):
    if method not in METHODS:
        raise ValueError('Method should be one of {}, found {}'.format(METHODS, method))
    return ord(METHODS_C[METHODS.index(method)])

def moving_weighted_median(data, weights, size, method="split"):
    """Compute moving weighted median for 1 and 2 dimensional arrays.

    Parameters
    ----------
    data : array_like
        The data to move the window over. Can have 1 or 2 dimensions. The data type should be
        float64 or something that can be converted to float64.
    weights : array_like
        The weights for the data. Can have 1 or 2 dimensions. The data type should be
        float64 or something that can be converted to float64.
    size : int or tuple of int
        Size of the window. All values must be uneven.
    method : str
        Either 'split', 'lower' or 'higher'. If multiple values sastisfy the conditions to be the
        weighted median of a window, this decides what is returned:
        split: The average of all candidate values is returned.
        lower: The lowest of all candidate values is returned.
        higher: The highest of all candidate values is returned.

    Returns
    -------
    :class:'numpy.ndarray'
        An array containing the weighted median values. The size is the same as the given data and
        weights, the data type is float64.

    Raises
    ------
    ValueError
        If the value of the window size was not odd.
    RuntimeError
        If there was an internal error in the C++ implementation.
    NotImplementedError
        If the data has more than two dimensions.
    """
    data, weights = _check_arrays(data, weights)
    cdef char c_method = _check_method(method)

    if size == 0:
        raise ValueError('Got size=0, what do you expect me to do with that?')
    if size == 1:
        return np.where(weights != 0, data, weights)

    if data.ndim == 1:
        if isinstance(size, (tuple, list)):
            if len(size) > 1:
                raise ValueError('Size ({}) has too many dimensions for 1D data.'.format(size))
            size = size[0]

        if size % 2 == 0:
            raise ValueError('Need an uneven window size (got {}).'.format(size))
        return _mwm_1D(data, weights, size, c_method)

    if data.ndim == 2:
        if any(np.asarray(size) % 2 == 0):
            raise ValueError('Need an uneven window size (got {}).'.format(size))

        return _mwm_2D(data, weights, size, c_method)
    raise NotImplementedError('weighted_median() is only implemented for 1 and 2 dimensions, not {}'
                         .format(data.ndim))


def _mwm_1D(np.ndarray[np.float64_t, ndim=1] data, np.ndarray[np.float64_t, ndim=1] weights, size,
            method):

    cdef Py_ssize_t len_data = data.shape[0]
    medians = np.ndarray(len_data, dtype=np.float64)
    cdef Tree[double] avl
    cdef deque[shared_ptr[Data[double]]] fifo
    cdef shared_ptr[Data[double]] node

    # Dummy node to enter into the fifo but not in the tree
    cdef shared_ptr[Data[double]] dummy = shared_ptr[Data[double]](new Data[double](0, 0))

    # Add all elements that are in the window on start
    for i in range(min(size // 2 + 1, len_data)):
        if weights[i] != 0:
            node = avl.insert(data[i], weights[i])
        else:
            node = dummy
        fifo.push_back(node)

    medians[0] = avl.weighted_median(method)

    # move window over the rest of the data
    for i in range(1, len_data):
        edge_r = i + size // 2
        edge_l = i - size // 2 - 1
        if edge_l >= 0:
            node = fifo.front()
            if node.get()[0].weight != 0:
                if avl.remove(node) is False:
                    raise RuntimeError('Error computing moving weighted median: Tried to remove a '
                                       'node that doesn`t exist from the tree.')
            fifo.pop_front()
        if edge_r < len_data:
            if weights[edge_r] != 0:
                node = avl.insert(data[edge_r], weights[edge_r])
            else:
                node = dummy
            fifo.push_back(node)

        # Get the median giving NaN if the tree if empty (all zero weights)
        medians[i] = avl.weighted_median(method) if avl.size() else np.nan
    return medians

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def _mwm_2D(np.ndarray[np.float64_t, ndim=2] data, np.ndarray[np.float64_t, ndim=2] weights, size,
            char method):

    # The 2D moving window goes through the matrix row-by-row, to simplify keeping track of
    # elements in the window with a fifo queue. But moving the window through the data in a zig-zag
    # could be more effizient, because it would reuse more elements when it jumps to the next line.
    cdef int len_data_y = data.shape[0]
    cdef int len_data_x = data.shape[1]
    cdef int len_queue = len_data_x*len_data_y

    cdef int size_y = size[0]
    cdef int size_x = size[1]
    cdef int rad_y = size_y // 2
    cdef int rad_x = size_x // 2

    medians = np.ndarray((len_data_y, len_data_x), dtype=np.float64)
    cdef Tree[double] avl
    cdef deque[shared_ptr[Data[double]]] fifo
    cdef shared_ptr[Data[double]] node
    cdef shared_ptr[Data[double]] dummy

    cdef int upper_limit
    cdef int lower_limit
    cdef int edge_r
    cdef int edge_l

    cdef double[:, ::1] median_view = medians

    cdef int row, x, col, y

    with nogil, parallel():
        # Get a FIFO for each thread
        fifo = deque[shared_ptr[Data[double]]]()

        # Dummy node to enter into the fifo but not in the tree
        dummy = shared_ptr[Data[double]](new Data[double](0, 0))

        # Loop over cols (in parallel using OpenMP).
        for col in prange(len_data_x, schedule='static'):
            # Too bad we have to throw away the tree to jump to the next row (see the comment under
            # the function definition)
            avl = Tree[double]()
            fifo.clear()

            # find upper (low x) and lower (high x) window edges
            upper_limit = col - rad_x
            if upper_limit < 0:
                upper_limit = 0
            lower_limit = col + rad_x + 1
            if lower_limit > len_data_x:
                lower_limit = len_data_x

            # Add all elements that are in the window on start the col
            for y in range(min(size_y // 2 + 1, len_data_y)):
                for x in range(upper_limit, lower_limit):
                    if weights[y,x] != 0:
                        node = avl.insert(data[y, x], weights[y, x])
                    else:
                        node = dummy
                    fifo.push_back(node)

            median_view[0, col] = avl.weighted_median(method)

            # move window over the col
            for row in range(1, len_data_y):
                edge_r = row + rad_y
                edge_l = row - rad_y - 1

                if edge_l >= 0:
                    # pop from fifo/tree what falls out of the window on the top
                    for x in range(upper_limit, lower_limit):
                        node = fifo.front()

                        # only remove from tree if it's not the dummy node
                        if node.get()[0].weight != 0:
                            if avl.remove(node) is False:
                                with gil:
                                    raise RuntimeError('Error computing moving weighted median: '
                                                       'Tried to remove a node that doesn`t exist '
                                                       'from the tree.')
                        fifo.pop_front()
                if edge_r < len_data_y:
                    # add a row of elements to the window from the bottom
                    for x in range(upper_limit, lower_limit):
                        if weights[edge_r, x] != 0:
                            node = avl.insert(data[edge_r, x], weights[edge_r, x])
                        else:
                            node = dummy
                        fifo.push_back(node)
                median_view[row, col] = avl.weighted_median(method)

    return medians
