# distutils: language = c++
# cython: language_level = 2
import numpy as np

from libcpp.memory cimport shared_ptr
from libcpp.deque cimport deque
from cython.parallel import prange, parallel
from MedianTree cimport Tree, Data
cimport numpy as np
cimport cython

# Define the fused types that can be used for the data or weights in the median routine
ctypedef fused data_t:
    int
    long
    float
    double

ctypedef fused weight_t:
    int
    long
    float
    double


@cython.wraparound(False)
@cython.boundscheck(False)
cdef data_t _quickselect_weight(data_t[::1] A, weight_t[::1] W, double qs, char method):
    # A, W: data and weights respectively, both are modified in place
    # qs: the amount of weight that we are selecting the element at
    # method: one of l, h or s to indicate whether to take the lower, higher or split
    #         (average) value if there is a tie

    cdef int i = 0, j = len(A) - 1
    cdef int ii, jj
    cdef double weights_left
    cdef data_t pivot

    cdef data_t min_d, max_d
    cdef data_t min_w, max_w

    while True:
        ii = i
        jj = j
        pivot = A[(ii + jj) // 2]
        weights_left = 0

        # Construct a Hoare partition of the array, while summing up the weights of the
        # left hand partition as we go. Although it would be a bit neater if this was
        # an external function, explicitly inlining it allows us to calculate the
        # weight sum as we go
        while True:

            # Move the left pointer forward until we find a list element which doesn't
            # satisfy the criterion. For every element that satisfies we need to
            # accumulate its weight
            while A[ii] < pivot:
                weights_left += W[ii]
                ii += 1

            # Move the right pointer back until we find a list element which doesn't
            # satisfy the criterion
            while A[jj] > pivot:
                jj -= 1

            # If the pointers meet or overlap then we're done
            if ii >= jj:

                # If the pointers match exactly then this means the last element didn't
                # get its weight added into the total
                if ii == jj:
                    weights_left += W[ii]

                break

            # If, not we need swap the elements and start again, the new left hand
            # element has it's weight added
            A[ii], A[jj] = A[jj], A[ii]
            W[ii], W[jj] = W[jj], W[ii]
            weights_left += W[ii]

            # Advance to the next elements
            ii += 1
            jj -= 1

        # If the total sum of the weights on the left equals the weight target that
        # means that the target element lies exactly on the boundary. In this case the
        # standard behaviour is to find the mean of the bounding elements. The easiest
        # way to do that is to find the maximum element in the left partition and the
        # minimum of the right
        if qs == weights_left:

            i = _max_non_zero(A, W, 0, jj)
            j = _min_non_zero(A, W, jj + 1, len(A) - 1)

            if (W[j] == 0 and W[i] > 0) or method == "l":
                return A[i]
            elif (W[i] == 0 and W[j] > 0) or method == "h":
                return A[j]
            else:  # method == "s":
                return <data_t>((A[i] + A[j]) / 2)

        # Otherwise the target element is in the left partition...
        elif qs <= weights_left:
            j = jj

        # ... or in the right
        else:
            i = jj + 1
            qs -= weights_left

        # When the list is a single element long, we're done and can just return it
        if i == j:
            return A[i]




@cython.wraparound(False)
@cython.boundscheck(False)
cdef int _max_non_zero(data_t[::1] d, weight_t[::1] w, int i, int j):
    # Find the maximum element in the array with non-zero weight. If all elements have
    # non-zero weight return the maximum of those
    cdef int ii
    cdef int cur = i

    for ii in range(i+1, j+1):
        if (w[cur] > 0) == (w[ii] > 0):
            if d[ii] > d[cur]:
                cur = ii
        elif w[ii] > 0:
            cur = ii

    return cur


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int _min_non_zero(data_t[::1] d, weight_t[::1] w, int i, int j):
    # Find the minimum element in the array with non-zero weight. If all elements have
    # non-zero weight return the minimum of those
    cdef int ii
    cdef int cur = i

    for ii in range(i+1, j+1):
        if (w[cur] > 0) == (w[ii] > 0):
            if d[ii] < d[cur]:
                cur = ii
        elif w[ii] > 0:
            cur = ii

    return cur


@cython.wraparound(False)
@cython.boundscheck(False)
cpdef void _quickselect(
    data_t[:, ::1] A,
    weight_t[:, ::1] W,
    double q,
    char method,
    data_t[::1] quantile,
    data_t[::1] At,
    weight_t[::1] Wt
):
    # Worker routine for the weighted quickselect

    cdef double wt, qs
    cdef int i, j

    for i in range(A.shape[0]):

        wt = 0

        for j in range(A.shape[1]):
            At[j] = A[i, j]
            Wt[j] = W[i, j]
            wt += W[i, j]

        # If all the weights are zero just ensures to do a regular quantile calculation
        # by using uniform non-zero weights
        if wt == 0:
            for j in range(W.shape[1]):
                Wt[j] = 1
            wt = W.shape[1]

        qs = wt * q


        quantile[i] = _quickselect_weight(At, Wt, qs, method)


def quantile(A, W, q, method="split"):
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
    A : array_like
        The array of data. Only 32 and 64 bit integers and floats are supported.
    W : array_like
        The array of weights. Only 32 and 64 bit integers and floats are supported.
    q : float
        The quantile as a number from zero to one.
    method : {"lower", "higher", "split"}:
        Method to use if the requested quantile is exactly between two elements.

    Returns
    -------
    r : np.ndarray or scalar
        The calculated quantile. This has the same shape as A with the last axis
        removed. If A was one dimensional the value returned is a scalar.

    Raises
    ------
    ValueError
        If the array shapes do not match, or the quantile value is invalid.
    TypeError
        If the array types are not supported.
    """
    cdef char methodc = _check_method(method)

    # Ensure that the inputs are numpy arrays
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(W, np.ndarray):
        W = np.array(W)


    if A.shape != W.shape:
        raise ValueError("Shapes of A and W much match.")

    if q < 0 or q > 1:
        raise ValueError("Quantile must be a number between zero and one.")

    # Reshape the arrays to 2D. The last axis is the one along which the quantile will
    # be calculated
    Ar = A.reshape(-1, A.shape[-1])
    Wr = W.reshape(-1, W.shape[-1])

    # Allocate the temporaries and output arrays
    At = np.ndarray(Ar.shape[-1], dtype=A.dtype)
    Wt = np.ndarray(Wr.shape[-1], dtype=W.dtype)
    res = np.ndarray(Ar.shape[0], dtype=Ar.dtype)

    # NOTE: this is super annoying. In theory this is what Cython fused types are meant
    # to be for, but in practice it seems to be impossible to combine arbitrarily sized
    # arrays (which require you to treat the arrays as Python objects), and fused
    # types. Explicitly dispatching to the correct routines seems to be the lesser of
    # all evils here
    if A.dtype == np.intc:
        if W.dtype == np.intc:
            _quickselect[int, int](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.int:
            _quickselect[int, long](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.single:
            _quickselect[int, float](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.double:
            _quickselect[int, double](Ar, Wr, q, methodc, res, At, Wt)
        else:
            raise TypeError("Type of weight array is not supported.")
    elif A.dtype == np.int:
        if W.dtype == np.intc:
            _quickselect[long, int](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.int:
            _quickselect[long, long](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.single:
            _quickselect[long, float](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.double:
            _quickselect[long, double](Ar, Wr, q, methodc, res, At, Wt)
        else:
            raise TypeError("Type of weight array is not supported.")
    elif A.dtype == np.single:
        if W.dtype == np.intc:
            _quickselect[float, int](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.int:
            _quickselect[float, long](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.single:
            _quickselect[float, float](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.double:
            _quickselect[float, double](Ar, Wr, q, methodc, res, At, Wt)
        else:
            raise TypeError("Type of weight array is not supported.")
    elif A.dtype == np.double:
        if W.dtype == np.intc:
            _quickselect[double, int](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.int:
            _quickselect[double, long](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.single:
            _quickselect[double, float](Ar, Wr, q, methodc, res, At, Wt)
        elif W.dtype == np.double:
            _quickselect[double, double](Ar, Wr, q, methodc, res, At, Wt)
        else:
            raise TypeError("Type of weight array is not supported.")
    else:
        raise TypeError("Type of data array is not supported.")

    # If the input was 1D dimensional, the output should be scalar. If it wasn't then
    # it the output has the same shape as the input without the last axis.
    if A.ndim == 1:
        return res[0]
    else:
        return res.reshape(A.shape[:-1])


def weighted_median(A, W, method="split"):
    """Calculate the weighted median of a set of data.

    The weighted median is always calculated along the last axis.

    See `quantile` for more information on the behaviour for some special cases.

    Parameters
    ----------
    A : np.ndarray
        The array of data.
    W : np.ndarray
        The array of weights.
    method : {"lower", "higher", "split"}:
        Method to use if the requested quantile is exactly between two elements.

    Returns
    -------
    r : np.ndarray or scalar
        The calculated median. This has the same shape as A with the last axis
        removed. If A was one dimensional the value returned is a scalar.
    """
    return quantile(A, W, 0.5, method=method)


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


def _check_method(method):
    METHODS = ['split', 'lower', 'higher']
    METHODS_C = ['s', 'l', 'h']

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
