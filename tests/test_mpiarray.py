"""Tests of MPI Array functionality.

Designed to be run as an MPI job with four processes like::

    $ mpirun -np 4 python test_mpiarray.py
"""

from typing import Union
from packaging import version
import pytest
from pytest_lazy_fixtures import lf
import h5py
import numpy as np
import zarr

from caput.memdata import fileformats
from caput.util import mpitools
from caput import darray


def _arange_dtype(N: int, dtype: Union[np.dtype, str]) -> np.ndarray:
    dtype = np.dtype(dtype)

    if dtype.kind == "U":
        return np.array([f"{ii}" for ii in range(N)], dtype=dtype)
    else:
        return np.arange(N, dtype=dtype)


def test_construction():
    """Test local/global shape construction of MPIArray."""
    arr = darray.MPIArray((10, 11), axis=1)

    l, s, _ = mpitools.split_local(11)

    # Check that global shape is set correctly
    assert arr.global_shape == (10, 11)

    assert arr.shape == (10, l)

    assert arr.local_offset == (0, s)

    assert arr.local_shape == (10, l)


@pytest.mark.parametrize(
    "dtype",
    [np.int64, np.float32, "U16"],
)
def test_redistribution(dtype):
    """Test redistributing an MPIArray."""
    gshape = (1, 11, 2, 14, 3, 4)
    nelem = np.prod(gshape)
    garr = _arange_dtype(nelem, dtype=dtype).reshape(gshape)

    _, s0, e0 = mpitools.split_local(11)
    _, s1, e1 = mpitools.split_local(14)
    _, s2, e2 = mpitools.split_local(4)

    arr = darray.MPIArray(gshape, axis=1, dtype=dtype)
    arr[:] = garr[:, s0:e0]

    arr2 = arr.redistribute(axis=3)
    assert (arr2.local_array == garr[:, :, :, s1:e1]).all()

    arr3 = arr.redistribute(axis=5)
    assert (arr3.local_array == garr[:, :, :, :, :, s2:e2]).all()

    assert arr.local_array.flags.c_contiguous
    assert arr2.local_array.flags.c_contiguous
    assert arr3.local_array.flags.c_contiguous


@pytest.mark.parametrize(
    "dtype",
    [np.int64, np.float32, "U16"],
)
def test_gather(dtype):
    """Test MPIArray.gather() including for unicode data which doesn't work natively."""
    rank = mpitools.rank
    size = mpitools.size
    block = 2

    global_shape = (2, 3, size * block)
    global_array = np.zeros(global_shape, dtype=dtype)
    global_array[..., :] = _arange_dtype(size * block, dtype=dtype)

    arr = darray.MPIArray(global_shape, dtype=dtype, axis=2)
    arr[:] = global_array[..., (rank * block) : ((rank + 1) * block)]

    assert (arr.allgather() == global_array).all()

    gather_rank = 1 if size > 1 else 0
    ga = arr.gather(rank=gather_rank)

    if rank == gather_rank:
        assert (ga == global_array).all()
    else:
        assert ga is None


def test_wrap():
    """Test MPIArray.wrap()."""
    ds = darray.MPIArray((10, 17))

    df = np.fft.rfft(ds, axis=1)

    assert isinstance(df, np.ndarray)

    da = darray.MPIArray.wrap(df, axis=0)

    assert isinstance(da, darray.MPIArray)
    assert da.global_shape == (10, 9)
    assert da.axis == 0

    l0, _, _ = mpitools.split_local(10)

    assert da.local_shape == (l0, 9)

    if mpitools.rank0:
        df = df[:-1]

    if mpitools.size > 1:
        with pytest.raises(Exception):
            darray.MPIArray.wrap(df, axis=0)


@pytest.mark.parametrize(
    "filename, file_open_function, file_format",
    [
        (lf("h5_file_distributed"), h5py.File, fileformats.HDF5),
        (
            lf("zarr_file_distributed"),
            zarr.open_group,
            fileformats.Zarr,
        ),
    ],
)
def test_io(filename, file_open_function, file_format):
    """Test I/O of MPIArray."""
    gshape = (19, 17)

    ds = darray.MPIArray(gshape, dtype=np.int64)

    ga = np.arange(np.prod(gshape)).reshape(gshape)

    _, s0, e0 = mpitools.split_local(gshape[0])
    ds[:] = ga[s0:e0]

    ds.redistribute(axis=1).to_file(
        filename, "testds", create=True, file_format=file_format
    )

    if mpitools.rank0:
        with file_open_function(filename, "r") as f:
            h5ds = f["testds"][:]

            assert (h5ds == ga).all()

    ds2 = darray.MPIArray.from_file(filename, "testds", file_format=file_format)

    assert (ds2 == ds).all()

    mpitools.barrier()

    # Check that reading over another distributed axis works
    ds3 = darray.MPIArray.from_file(filename, "testds", axis=1, file_format=file_format)
    assert ds3.shape[0] == gshape[0]
    assert ds3.shape[1] == mpitools.split_local(gshape[1])[0]
    ds3 = ds3.redistribute(axis=0)
    assert (ds3 == ds).all()
    mpitools.barrier()

    # Check a read with an arbitrary slice in there. This only checks the shape is correct.
    ds4 = darray.MPIArray.from_file(
        filename,
        "testds",
        axis=1,
        sel=(np.s_[3:10:2], np.s_[1:16:3]),
        file_format=file_format,
    )
    assert ds4.shape[0] == 4
    assert ds4.shape[1] == mpitools.split_local(5)[0]
    mpitools.barrier()

    # Check the read with a slice along the axis being read
    ds5 = darray.MPIArray.from_file(
        filename,
        "testds",
        axis=1,
        sel=(np.s_[:], np.s_[3:15:2]),
        file_format=file_format,
    )
    assert ds5.shape[0] == gshape[0]
    assert ds5.shape[1] == mpitools.split_local(6)[0]
    ds5 = ds5.redistribute(axis=0)
    assert (ds5 == ds[:, 3:15:2]).all()
    mpitools.barrier()

    # Check the read with a slice along the axis being read
    ds6 = darray.MPIArray.from_file(
        filename,
        "testds",
        axis=0,
        sel=(np.s_[:], np.s_[3:15:2]),
        file_format=file_format,
    )
    ds6 = ds6.redistribute(axis=0)
    assert (ds6 == ds[:, 3:15:2]).all()
    mpitools.barrier()


def test_transpose():
    gshape = (1, 11, 2, 14)

    l0, s0, _ = mpitools.split_local(11)

    arr = darray.MPIArray(gshape, axis=1, dtype=np.int64)

    arr2 = arr.transpose(1, 3, 0, 2)

    # Check type
    assert isinstance(arr2, darray.MPIArray)

    # Check global shape
    assert arr2.global_shape == (11, 14, 1, 2)

    # Check local shape
    assert arr2.local_shape == (l0, 14, 1, 2)

    # Check local offset
    assert arr2.local_offset == (s0, 0, 0, 0)

    # Check axis
    assert arr2.axis == 0

    # Do the same test with a tuple as argument to transpose
    arr3 = arr.transpose((1, 3, 0, 2))

    # Check type
    assert isinstance(arr3, darray.MPIArray)

    # Check global shape
    assert arr3.global_shape == (11, 14, 1, 2)

    # Check local shape
    assert arr3.local_shape == (l0, 14, 1, 2)

    # Check local offset
    assert arr3.local_offset == (s0, 0, 0, 0)

    # Check axis
    assert arr3.axis == 0

    # Do the same test with None as argument to transpose
    arr4 = arr.transpose()

    # Check type
    assert isinstance(arr4, darray.MPIArray)

    # Check global shape
    assert arr4.global_shape == (14, 2, 11, 1)

    # Check local shape
    assert arr4.local_shape == (14, 2, l0, 1)

    # Check local offset
    assert arr4.local_offset == (0, 0, s0, 0)

    # Check axis
    assert arr4.axis == 2


def test_copy():
    # Test that standard numpy.copy() method works
    # for MPIArrays
    size = mpitools.size

    arr = darray.ones((3, size, 14), axis=1, dtype=np.float32)
    arr2 = arr.copy()

    assert (arr == arr2).all()

    # Check type
    assert isinstance(arr2, darray.MPIArray)

    # Check global shape
    assert arr.global_shape == arr2.global_shape

    # Check axis
    assert arr2.axis == 1


def test_reshape():
    gshape = (1, 11, 2, 14)

    # Redistribute with the axis in the middle
    arr = darray.MPIArray(gshape, axis=1, dtype=np.int64)
    arr2 = arr.reshape((None, 28))
    l0, s0, _ = mpitools.split_local(11)

    # Check the type, global_shape, local_shape, local_offset and axis as are expected
    assert isinstance(arr2, darray.MPIArray)
    assert arr2.global_shape == (11, 28)
    assert arr2.local_shape == (l0, 28)
    assert arr2.local_offset == (s0, 0)
    assert arr2.axis == 0

    # Another test but now with the axis far at the end, this catches a bug where if the
    # number of axes shrunk enough the distributed axis would index off the end
    arr = darray.MPIArray(gshape, axis=3, dtype=np.int64)
    arr2 = arr.reshape((22, None))
    l0, s0, _ = mpitools.split_local(14)

    # Check the type, global_shape, local_shape, local_offset and axis as are expected
    assert isinstance(arr2, darray.MPIArray)
    assert arr2.global_shape == (22, 14)
    assert arr2.local_shape == (22, l0)
    assert arr2.local_offset == (0, s0)
    assert arr2.axis == 1

    # Check a reshape with a wildcard works
    arr2 = arr.reshape(-1, None)

    # Check the type, global_shape, local_shape, local_offset and axis as are expected
    assert isinstance(arr2, darray.MPIArray)
    assert arr2.global_shape == (22, 14)
    assert arr2.local_shape == (22, l0)
    assert arr2.local_offset == (0, s0)
    assert arr2.axis == 1


# pylint: disable=too-many-statements
def test_global_getslice():
    rank = mpitools.rank
    size = mpitools.size

    darr = darray.MPIArray((size * 5, 20), axis=0)

    # Initialise the distributed array
    for li, _ in darr.enumerate(axis=0):
        darr.local_array[li] = 10 * (10 * rank + li) + np.arange(20)

    # Construct numpy array which should be equivalent to the global array
    whole_array = (
        10
        * (
            10 * np.arange(4.0)[:, np.newaxis] + np.arange(5.0)[np.newaxis, :]
        ).flatten()[:, np.newaxis]
        + np.arange(20)[np.newaxis, :]
    )

    # Extract the section for each rank distributed along axis=0
    local_array = whole_array[(rank * 5) : ((rank + 1) * 5)]

    # Extract the correct section for each rank distributed along axis=0
    local_array_T = whole_array[:, (rank * 5) : ((rank + 1) * 5)]

    # Check that these are the same
    assert (local_array == darr.local_array).all()

    # Check a simple slice on the non-parallel axis
    arr = darr.global_slice[:, 3:5]
    res = local_array[:, 3:5]

    assert isinstance(arr, darray.MPIArray)
    assert arr.axis == 0
    assert (arr.local_array == res).all()

    # Check a single element extracted from the non-parallel axis
    arr = darr.global_slice[:, 3]
    res = local_array[:, 3]
    assert (arr.local_array == res).all()

    # Check that slices contain MPIArray attributes
    assert hasattr(arr, "comm") and (arr.comm == darr.comm)

    # These tests depend on the size being at least 2.
    if size > 1:
        # Check a slice on the parallel axis
        arr = darr.global_slice[:7, 3:5]

        res = {0: local_array[:, 3:5], 1: local_array[:2, 3:5], 2: None, 3: None}

        assert arr == res[rank] if arr is None else (arr == res[rank]).all()

        # Check a single element from the parallel axis
        arr = darr.global_slice[7, 3:5]

        res = {0: None, 1: local_array[2, 3:5], 2: None, 3: None}

        assert arr == res[rank] if arr is None else (arr == res[rank]).all()

        # Check a slice on the redistributed parallel axis
        darr_T = darr.redistribute(axis=1)
        arr = darr_T.global_slice[3:5, :7]

        res = {
            0: local_array_T[3:5, :],
            1: local_array_T[3:5, :2],
            2: None,
            3: None,
        }

        assert arr == res[rank] if arr is None else (arr == res[rank]).all()

    # Check a slice that removes an axis
    darr = darray.MPIArray((10, 20, size * 5), axis=2)
    dslice = darr.global_slice[:, 0, :]

    assert dslice.global_shape == (10, size * 5)
    assert dslice.local_shape == (10, 5)
    assert dslice.axis == 1

    # Check that directly indexing into distributed axis returns a numpy array equal to
    # local array indexing
    darr = darray.MPIArray((size,), axis=0)
    with pytest.warns(UserWarning):
        assert (darr[0] == darr.local_array[0]).all()

    # Check that a single index into a non-parallel axis works
    darr = darray.MPIArray((4, size), axis=1)
    darr[:] = rank
    assert (darr[0] == rank).all()
    assert darr[0].axis == 0
    # check that direct slicing into distributed axis returns a numpy array for local array slicing
    with pytest.warns(UserWarning):
        assert (darr[2, 0] == darr.local_array[2, 0]).all()

    darr = darray.MPIArray((20, size * 5), axis=1)
    darr[:] = rank
    # But, you can directly index with global_slice
    if size > 1:
        dslice = darr.global_slice[2, 6]
        if rank != 1:
            assert dslice is None
        else:
            assert (dslice == rank).all()

    # Check that printing the array does not trigger
    # an exception
    assert str(darr)

    # more direct indexing into distributed with global_slice
    # the global slice should return a numpy array on rank=1, and None everywhere else
    if size >= 2:
        darr = darray.MPIArray((20, 10, size * 5), axis=2)
        darr[:] = rank
        dslice = darr.global_slice[:, :, 6]
        if rank != 1:
            assert dslice is None
        else:
            nparr = np.ndarray((20, 10))
            nparr[:] = rank
            assert isinstance(dslice, np.ndarray)
            assert (dslice == nparr).all()

        # check that directly slicing a distributed axis returns a local array
        with pytest.warns(UserWarning):
            assert (darr[:, :, 2:3] == darr.local_array[:, :, 2:3]).all()

    # Check ellipsis and slice at the end
    darr = darray.MPIArray((size * 5, 20, 10), axis=0)
    dslice = darr.global_slice[..., 4:9]

    assert dslice.global_shape == (size * 5, 20, 5)
    assert dslice.local_shape == (5, 20, 5)

    # Check slice that goes off the end of the axis
    darr = darray.MPIArray((size, 136, 2048), axis=0)
    dslice = darr.global_slice[..., 2007:2087]

    assert dslice.global_shape == (size, 136, 41)
    assert dslice.local_shape == (1, 136, 41)


def test_global_setslice():
    rank = mpitools.rank
    size = mpitools.size

    darr = darray.MPIArray((size * 5, 20), axis=0)

    # Initialise the distributed array
    for li, _ in darr.enumerate(axis=0):
        darr[li] = 10 * (10 * rank + li) + np.arange(20)

    # Construct numpy array which should be equivalent to the global array
    whole_array = (
        10
        * (
            10 * np.arange(4.0)[:, np.newaxis] + np.arange(5.0)[np.newaxis, :]
        ).flatten()[:, np.newaxis]
        + np.arange(20)[np.newaxis, :]
    )

    # Extract the section for each rank distributed along axis=0
    local_array = whole_array[(rank * 5) : ((rank + 1) * 5)]
    # Set slice

    # Check a simple assignment to a slice along the non-parallel axis
    darr.global_slice[:, 6] = -2.0
    local_array[:, 6] = -2.0

    assert (darr.local_array == local_array).all()

    # Check a partial assignment along the parallel axis
    darr.global_slice[7:, 7:9] = -3.0
    whole_array[7:, 7:9] = -3.0

    assert (darr.local_array == local_array).all()

    # Check assignment of a single index on the parallel axis
    darr.global_slice[6] = np.arange(20.0)
    whole_array[6] = np.arange(20.0)

    assert (darr.local_array == local_array).all()

    # Check copy of one column into the other
    darr.global_slice[:, 8] = darr.global_slice[:, 9]
    whole_array[:, 8] = whole_array[:, 9]

    assert (darr.local_array == local_array).all()

    # test setting complex dtypes

    darr_complex = darray.MPIArray((size * 5, 20), axis=0, dtype=np.complex64)

    darr_complex[:] = 4
    assert darr_complex.dtype == np.complex64
    assert (darr_complex == 4.0 + 0.0j).all()

    darr_complex[:] = 2.0 + 1.345j

    assert darr_complex.dtype == np.complex64
    assert (darr_complex == 2.0 + 1.345j).all()

    darr_float = darray.MPIArray((size * 5, 20), axis=0, dtype=np.float64)

    darr_float[:] = 4.0
    assert darr_float.dtype == np.float64
    assert (darr_float == 4.0).all()


def test_ufunc_call():
    rank = mpitools.rank
    size = mpitools.size

    dist_arr = darray.MPIArray((size, 4), axis=0)
    dist_arr[:] = rank

    dist_arr_add = dist_arr + dist_arr

    dist_arr_add = dist_arr + dist_arr
    dist_arr_mul = 2 * dist_arr

    # check that you can add two MPIArrays with the same shape
    assert (dist_arr_add == 2 * rank).all()

    # Check that you can multiply an MPIArray against a scalar
    assert (dist_arr_mul == 2 * rank).all()

    # Check that basic output MPI attributes are correct
    assert hasattr(dist_arr_add, "axis") and hasattr(dist_arr_add, "comm")

    assert dist_arr_add.axis == 0

    assert dist_arr_add.comm is dist_arr.comm

    # add differently shaped MPIArrays, with broadcasting
    dist_arr_2 = darray.MPIArray((size, 1), axis=0)
    dist_arr_2[:] = rank - 1
    assert (dist_arr + dist_arr_2 == 2 * rank - 1).all()
    assert (dist_arr + dist_arr_2).axis == 0

    # check that subtracting arrays with two different distributed axis fails
    # pylint: disable=expression-not-assigned
    with pytest.raises(darray.AxisException):
        darray.MPIArray((size, 4), axis=0) - darray.MPIArray((size, 4), axis=1)

    # check that outer ufunc on arrays that cannot be broadcast fails
    with pytest.raises(ValueError):
        np.multiply(
            darray.MPIArray((size, 3), axis=0), darray.MPIArray((size, 4), axis=0)
        )
    # pylint: enable=expression-not-assigned

    # test ufuncs with complex dtypes

    dist_complex = darray.MPIArray((size, 4), axis=0, dtype=np.complex64)
    dist_complex_add = dist_complex + dist_complex

    assert (dist_complex_add == dist_complex + dist_complex).all()
    assert dist_complex_add.dtype == np.complex64


def test_ufunc_broadcast():
    # Test a call ufunc where one of the arguments will get broadcasted to a higher
    # dimensionality

    rank = mpitools.rank
    size = mpitools.size

    dist_arr1 = darray.MPIArray((4, size), axis=1)
    dist_arr1.local_array[:] = rank

    dist_arr2 = darray.MPIArray((size,), axis=0)
    dist_arr2.local_array[:] = rank

    dist_arr3 = dist_arr1 + dist_arr2
    assert (dist_arr3 == 2 * dist_arr1).all()

    # Test a broadcast against a numpy array of the same dimensionality
    nondist_arr = 2 * np.ones((4, 1))
    assert (dist_arr1 * nondist_arr == 2 * rank).all()

    # Test a broadcast against a numpy array of the same dimensionality, but with a
    # mismatch on the "distributed" axis. This should fail
    nondist_arr = np.ones((4, 3))
    with pytest.raises(ValueError):
        _ = dist_arr1 * nondist_arr

    # Test a broadcast against a numpy array of lower dimensionality, with the
    # distributed axis being one of the axes on the non distributed array
    nondist_arr = 2 * np.ones((1,))
    assert (dist_arr1 * nondist_arr == 2 * rank).all()

    # Test a broadcast against a numpy array of lower dimensionality, with the
    # distributed axis being one of the axes implicitly added to the numpy array
    dist_arr3 = darray.MPIArray((size, 4), axis=0)
    dist_arr3.local_array[:] = rank
    nondist_arr = 2 * np.ones((4,))
    assert (dist_arr3 * nondist_arr == 2 * rank).all()

    # Test a broadcast against a numpy array of higher dimensionality, with the
    # distributed axis being one of the axes on the non distributed array
    nondist_arr = 2 * np.ones((1, 4, 1))
    t = dist_arr1 * nondist_arr
    assert t.global_shape == (1, 4, size)
    assert (t == 2 * rank).all()


def test_ufunc_2output():
    # Test a ufunc (divmod) which will return two outputs simultaneously
    rank = mpitools.rank
    size = mpitools.size

    dist_arr = darray.MPIArray((size, 4), axis=0)
    dist_arr[:] = rank

    quotient, remainder = np.divmod(dist_arr, 2)

    # Check that the arrays come back as MPIArrays with the correct structure
    assert isinstance(quotient, darray.MPIArray)
    assert isinstance(remainder, darray.MPIArray)

    assert quotient.shape == dist_arr.shape
    assert remainder.shape == dist_arr.shape

    assert quotient.axis == dist_arr.axis
    assert remainder.axis == dist_arr.axis

    assert quotient.local_offset == dist_arr.local_offset
    assert remainder.local_offset == dist_arr.local_offset

    # Check they have the expected values
    assert (quotient.local_array == (rank // 2)).all()
    assert (remainder.local_array == (rank % 2)).all()


def test_ufunc_reduce():
    rank = mpitools.rank
    size = mpitools.size

    dist_array = darray.MPIArray((size, 4, 3), axis=0)
    dist_array[:] = rank

    # sums across non-distributed axes should be permitted, and work as usual
    assert (dist_array.sum(axis=1) == 4 * rank).all()

    # sum() should reduce across all non-distributed axes
    sum_all = dist_array.sum()
    assert (sum_all == 4 * 3 * rank).all()

    assert sum_all.local_shape == (1,)

    assert sum_all.global_shape == (size,)

    # Reductions should fail across the distributed axis
    with pytest.raises(darray.AxisException):
        dist_array.sum(axis=0)

    # sum across a smaller numbered axes
    # this will result in an axes reduction
    dist_array = darray.MPIArray((5, size, 3), axis=1)
    dist_array[:] = rank

    sum_array_0 = dist_array.sum(axis=0)
    assert (sum_array_0 == 5 * rank).all()

    # check that the new axes was calculated accordingly
    assert sum_array_0.axis == 0
    assert sum_array_0.global_shape == (size, 3)

    assert dist_array.sum(axis=0, keepdims=True).axis == 1

    sum_all = dist_array.sum()
    assert (sum_all == 5 * 3 * rank).all()

    assert sum_all.local_shape == (1,)

    assert sum_all.global_shape == (size,)

    assert darray.MPIArray((size, 4), axis=1).sum(axis=0).axis == 0

    # test AllReduce
    if size > 1:
        from mpi4py import MPI

        # Test comm.Allreduce
        dist_array = darray.MPIArray((size, 4), axis=1)
        dist_array[:] = 1

        df_sum = np.sum(dist_array, axis=0)

        df_total = np.zeros_like(df_sum)

        dist_array.comm.Allreduce(df_sum, df_total, op=MPI.SUM)

        assert (df_total == 4 * size).all()

        # Test MPIArray.allreduce()

        # MPIArray.sum().allreduce() should give the scalar sum of
        # all entries
        df_sum = dist_array.sum()
        df_total = np.zeros_like(df_sum)

        assert dist_array.sum().allreduce() == 4 * size

        df_sum.comm.Allreduce(df_sum, df_total, op=MPI.SUM)

        assert df_total == dist_array.sum().allreduce()

        with pytest.raises(ValueError):
            dist_array.allreduce()


def test_ufunc_reduce_multi():
    # Test that reductions over multiple axes don't break things
    size = mpitools.size

    dist_array = darray.MPIArray((size, 4, 4), axis=0, dtype=np.float64)
    dist_array.local_array[:] = 1.0

    with pytest.raises(darray.AxisException):
        dist_array.sum(axis=(0, 1))

    a = dist_array.sum(axis=(1, 2))
    assert np.all(a.local_array == 16.0)

    dist_array = darray.MPIArray((4, 4, size), axis=2, dtype=np.float64)
    dist_array.local_array[:] = 1.0

    b = dist_array.sum(axis=(0, 1))
    assert np.all(b.local_array == 16.0)


def test_ufunc_accumulate():
    # Test that reductions over multiple axes don't break things
    size = mpitools.size

    dist_array = darray.MPIArray((size, 4, 4), axis=0, dtype=np.float64)
    dist_array.local_array[:] = 1.0

    dist_array_cumsum = np.cumsum(dist_array, axis=2)

    assert dist_array_cumsum.shape == (1, 4, 4)
    assert dist_array_cumsum.axis == 0

    assert (dist_array_cumsum.local_array == np.arange(1.0, 5.0)).all()


def test_ufunc_misc():
    # Test miscellaneous ufunc related issues

    size = mpitools.size

    dist_array1 = darray.MPIArray((size, 4, 4), axis=0, dtype=np.float64)
    dist_array2 = darray.MPIArray((size, 4, 4), axis=0, dtype=np.float64)
    dist_array1.local_array[:] = 1.0
    dist_array2.local_array[:] = 1.0

    # Check that unsupported operations types fail
    # Check that an outer call fails
    with pytest.raises(darray.UnsupportedOperation):
        np.multiply.outer(dist_array1, dist_array2)

    # Check that a reduceat call fails
    with pytest.raises(darray.UnsupportedOperation):
        np.multiply.reduceat(dist_array1, [0, 2], axis=1)

    # Check that an at call fails
    with pytest.raises(darray.UnsupportedOperation):
        np.exp.at(dist_array1, [(0, 2, 1)])

    # Check that where arguments are not supported
    with pytest.raises(darray.UnsupportedOperation):
        np.exp(dist_array1, where=(dist_array2 == 1.0))

    # Check that feeding in one positional argument is interpreted as an out argument
    dist_array3 = np.exp(dist_array1, dist_array2)
    assert dist_array2 is dist_array3
    assert (dist_array2.local_array == np.exp(dist_array1.local_array)).all()

    # Check that feeding in too many arguments actually does fail
    with pytest.raises(TypeError):
        np.exp(dist_array1, dist_array2, dist_array1)


def test_slice_newaxis():
    rank = mpitools.rank
    size = mpitools.size

    # Test inserting an axis before a normal axis
    dist_array = darray.MPIArray((4, size, 3), axis=1)
    dist_array[:] = rank
    new_dist_array = dist_array[np.newaxis, :, :, :]
    assert new_dist_array.shape == (1, 4, 1, 3)
    assert new_dist_array.global_shape == (1, 4, size, 3)
    assert (new_dist_array[:] == rank).all()

    # Test inserting an axis before the distributed axis
    new_dist_array = dist_array[:, np.newaxis, :, :]
    assert new_dist_array.shape == (4, 1, 1, 3)
    assert new_dist_array.global_shape == (4, 1, size, 3)
    assert (new_dist_array[:] == rank).all()


def test_slice_ellipsis():
    rank = mpitools.rank
    size = mpitools.size

    dist_array = darray.MPIArray((4, size, 3), axis=1)
    dist_array[:] = rank

    # Test selecting an axis at the end via an ellipsis
    new_dist_array = dist_array[..., 0]
    assert new_dist_array.shape == (4, 1)
    assert new_dist_array.global_shape == (4, size)
    assert (new_dist_array[:] == rank).all()

    # Test inserting an axis at the end via an ellipsis
    new_dist_array = dist_array[..., np.newaxis]
    assert new_dist_array.shape == (4, 1, 3, 1)
    assert new_dist_array.global_shape == (4, size, 3, 1)
    assert (new_dist_array[:] == rank).all()

    bool_sel = np.ones(3, dtype=bool)
    bool_sel[2:] = False
    new_dist_array = dist_array[..., bool_sel]
    assert new_dist_array.shape == (4, 1, 2)
    assert new_dist_array.global_shape == (4, size, 2)
    assert (new_dist_array == rank).all()


def test_slice_npint64():
    rank = mpitools.rank
    size = mpitools.size

    # Test inserting an axis before a normal axis
    dist_array = darray.MPIArray((4, size, 3), axis=1)
    dist_array[:] = rank
    new_dist_array = dist_array[np.int64(2), :, :]
    assert new_dist_array.shape == (1, 3)
    assert new_dist_array.global_shape == (size, 3)
    assert (new_dist_array[:] == rank).all()


def test_mpi_array_fill():
    rank = mpitools.rank
    size = mpitools.size

    arr_zero = darray.zeros((4, size, 17), axis=1, dtype=np.float32)
    assert (arr_zero == 0).all()
    assert arr_zero.axis == 1
    assert arr_zero.global_shape == (4, size, 17)
    assert arr_zero.shape == (4, 1, 17)
    assert arr_zero.local_offset == (0, rank, 0)
    assert arr_zero.dtype == np.float32

    arr_ones = darray.ones((4, size, 17), axis=1, dtype=int)
    assert (arr_ones == 1).all()
    assert arr_ones.axis == 1
    assert arr_ones.global_shape == (4, size, 17)
    assert arr_ones.shape == (4, 1, 17)
    assert arr_ones.local_offset == (0, rank, 0)
    assert arr_ones.dtype == int


def test_call_ravel():
    size = mpitools.size

    arr_ones = darray.ones((4, size, 17), axis=1)
    with pytest.raises(NotImplementedError):
        arr_ones.ravel()


def test_call_median():
    size = mpitools.size

    arr = darray.ones((4, size, 17), axis=1)
    arr[..., 0] = 1700.0

    # Check that this will fail correctly when trying to
    # take median across the distributed axis
    with pytest.raises(darray.AxisException):
        np.median(arr, axis=1)

    if version.parse(np.__version__) >= version.parse("1.25.0"):
        # Check that the median is correct along the last axis
        assert np.all(np.median(arr, axis=-1) == 1)

        # Check that the median is correct along the first axis
        res = np.ones(17)
        res[0] = 1700.0
        assert np.all(np.median(arr, axis=0) == res)
    else:
        # Median does not work on earlier numpy versions because of an internal use of
        # .ravel()
        with pytest.raises(NotImplementedError):
            np.median(arr, axis=0)

    # Check that median call with local array works
    assert (np.median(arr.local_array, axis=-1) == 1).all()
