"""Tests of MPI Array functionality.

Designed to be run as an MPI job with four processes like::

    $ mpirun -np 4 python test_mpiarray.py
"""
import pytest
from pytest_lazyfixture import lazy_fixture
import h5py
import numpy as np
import zarr

from caput import mpiutil, mpiarray, fileformats


def test_construction():
    """Test local/global shape construction of MPIArray."""
    arr = mpiarray.MPIArray((10, 11), axis=1)

    l, s, _ = mpiutil.split_local(11)

    # Check that global shape is set correctly
    assert arr.global_shape == (10, 11)

    assert arr.shape == (10, l)

    assert arr.local_offset == (0, s)

    assert arr.local_shape == (10, l)


def test_redistribution():
    """Test redistributing an MPIArray."""
    gshape = (1, 11, 2, 14, 3, 4)
    nelem = np.prod(gshape)
    garr = np.arange(nelem).reshape(gshape)

    _, s0, e0 = mpiutil.split_local(11)
    _, s1, e1 = mpiutil.split_local(14)
    _, s2, e2 = mpiutil.split_local(4)

    arr = mpiarray.MPIArray(gshape, axis=1, dtype=np.int64)
    arr[:] = garr[:, s0:e0]

    arr2 = arr.redistribute(axis=3)
    assert (arr2 == garr[:, :, :, s1:e1]).view(np.ndarray).all()

    arr3 = arr.redistribute(axis=5)
    assert (arr3 == garr[:, :, :, :, :, s2:e2]).view(np.ndarray).all()


def test_gather():
    """Test MPIArray.gather()."""
    rank = mpiutil.rank
    size = mpiutil.size
    block = 2

    global_shape = (2, 3, size * block)
    global_array = np.zeros(global_shape, dtype=np.float64)
    global_array[..., :] = np.arange(size * block)

    arr = mpiarray.MPIArray(global_shape, dtype=np.float64, axis=2)
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
    ds = mpiarray.MPIArray((10, 17))

    df = np.fft.rfft(ds, axis=1)

    assert isinstance(df, np.ndarray)

    da = mpiarray.MPIArray.wrap(df, axis=0)

    assert isinstance(da, mpiarray.MPIArray)
    assert da.global_shape == (10, 9)
    assert da.axis == 0

    l0, _, _ = mpiutil.split_local(10)

    assert da.local_shape == (l0, 9)

    if mpiutil.rank0:
        df = df[:-1]

    if mpiutil.size > 1:
        with pytest.raises(Exception):
            mpiarray.MPIArray.wrap(df, axis=0)


@pytest.mark.parametrize(
    "filename, file_open_function, file_format",
    [
        (lazy_fixture("h5_file_distributed"), h5py.File, fileformats.HDF5),
        (
            lazy_fixture("zarr_file_distributed"),
            zarr.open_group,
            fileformats.Zarr,
        ),
    ],
)
def test_io(filename, file_open_function, file_format):
    """Test I/O of MPIArray."""
    gshape = (19, 17)

    ds = mpiarray.MPIArray(gshape, dtype=np.int64)

    ga = np.arange(np.prod(gshape)).reshape(gshape)

    _, s0, e0 = mpiutil.split_local(gshape[0])
    ds[:] = ga[s0:e0]

    ds.redistribute(axis=1).to_file(
        filename, "testds", create=True, file_format=file_format
    )

    if mpiutil.rank0:

        with file_open_function(filename, "r") as f:
            h5ds = f["testds"][:]

            assert (h5ds == ga).all()

    ds2 = mpiarray.MPIArray.from_file(filename, "testds", file_format=file_format)

    assert (ds2 == ds).all()

    mpiutil.barrier()

    # Check that reading over another distributed axis works
    ds3 = mpiarray.MPIArray.from_file(
        filename, "testds", axis=1, file_format=file_format
    )
    assert ds3.shape[0] == gshape[0]
    assert ds3.shape[1] == mpiutil.split_local(gshape[1])[0]
    ds3 = ds3.redistribute(axis=0)
    assert (ds3 == ds).all()
    mpiutil.barrier()

    # Check a read with an arbitrary slice in there. This only checks the shape is correct.
    ds4 = mpiarray.MPIArray.from_file(
        filename,
        "testds",
        axis=1,
        sel=(np.s_[3:10:2], np.s_[1:16:3]),
        file_format=file_format,
    )
    assert ds4.shape[0] == 4
    assert ds4.shape[1] == mpiutil.split_local(5)[0]
    mpiutil.barrier()

    # Check the read with a slice along the axis being read
    ds5 = mpiarray.MPIArray.from_file(
        filename,
        "testds",
        axis=1,
        sel=(np.s_[:], np.s_[3:15:2]),
        file_format=file_format,
    )
    assert ds5.shape[0] == gshape[0]
    assert ds5.shape[1] == mpiutil.split_local(6)[0]
    ds5 = ds5.redistribute(axis=0)
    assert (ds5 == ds[:, 3:15:2]).all()
    mpiutil.barrier()

    # Check the read with a slice along the axis being read
    ds6 = mpiarray.MPIArray.from_file(
        filename,
        "testds",
        axis=0,
        sel=(np.s_[:], np.s_[3:15:2]),
        file_format=file_format,
    )
    ds6 = ds6.redistribute(axis=0)
    assert (ds6 == ds[:, 3:15:2]).all()
    mpiutil.barrier()


def test_transpose():

    gshape = (1, 11, 2, 14)

    l0, s0, _ = mpiutil.split_local(11)

    arr = mpiarray.MPIArray(gshape, axis=1, dtype=np.int64)

    arr2 = arr.transpose(1, 3, 0, 2)

    # Check type
    assert isinstance(arr2, mpiarray.MPIArray)

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
    assert isinstance(arr3, mpiarray.MPIArray)

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
    assert isinstance(arr4, mpiarray.MPIArray)

    # Check global shape
    assert arr4.global_shape == (14, 2, 11, 1)

    # Check local shape
    assert arr4.local_shape == (14, 2, l0, 1)

    # Check local offset
    assert arr4.local_offset == (0, 0, s0, 0)

    # Check axis
    assert arr4.axis == 2


def test_reshape():

    gshape = (1, 11, 2, 14)

    l0, s0, _ = mpiutil.split_local(11)

    arr = mpiarray.MPIArray(gshape, axis=1, dtype=np.int64)

    arr2 = arr.reshape((None, 28))

    # Check type
    assert isinstance(arr2, mpiarray.MPIArray)

    # Check global shape
    assert arr2.global_shape == (11, 28)

    # Check local shape
    assert arr2.local_shape == (l0, 28)

    # Check local offset
    assert arr2.local_offset == (s0, 0)

    # Check axis
    assert arr2.axis == 0


# pylint: disable=too-many-statements
def test_global_getslice():

    rank = mpiutil.rank
    size = mpiutil.size

    darr = mpiarray.MPIArray((size * 5, 20), axis=0)

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
    assert (local_array == darr).all()

    # Check a simple slice on the non-parallel axis
    arr = darr.global_slice[:, 3:5]
    res = local_array[:, 3:5]

    assert isinstance(arr, mpiarray.MPIArray)
    assert arr.axis == 0
    assert (arr == res).all()

    # Check a single element extracted from the non-parallel axis
    arr = darr.global_slice[:, 3]
    res = local_array[:, 3]
    assert (arr == res).all()

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
    darr = mpiarray.MPIArray((10, 20, size * 5), axis=2)
    dslice = darr.global_slice[:, 0, :]

    assert dslice.global_shape == (10, size * 5)
    assert dslice.local_shape == (10, 5)
    assert dslice.axis == 1

    # Check that directly indexing into distributed axis returns a numpy array equal to
    # local array indexing
    darr = mpiarray.MPIArray((size,), axis=0)
    with pytest.warns(UserWarning):
        assert (darr[0] == darr.local_array[0]).all()

    # Check that a single index into a non-parallel axis works
    darr = mpiarray.MPIArray((4, size), axis=1)
    darr[:] = rank
    assert (darr[0] == rank).all()
    assert darr[0].axis == 0
    # check that direct slicing into distributed axis returns a numpy array for local array slicing
    with pytest.warns(UserWarning):
        assert (darr[2, 0] == darr.local_array[2, 0]).all()

    darr = mpiarray.MPIArray((20, size * 5), axis=1)
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
        darr = mpiarray.MPIArray((20, 10, size * 5), axis=2)
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
        assert (darr[:, :, 2:3] == darr.local_array[:, :, 2:3]).all()

    # Check ellipsis and slice at the end
    darr = mpiarray.MPIArray((size * 5, 20, 10), axis=0)
    dslice = darr.global_slice[..., 4:9]

    assert dslice.global_shape == (size * 5, 20, 5)
    assert dslice.local_shape == (5, 20, 5)

    # Check slice that goes off the end of the axis
    darr = mpiarray.MPIArray((size, 136, 2048), axis=0)
    dslice = darr.global_slice[..., 2007:2087]

    assert dslice.global_shape == (size, 136, 41)
    assert dslice.local_shape == (1, 136, 41)


def test_global_setslice():
    rank = mpiutil.rank
    size = mpiutil.size

    darr = mpiarray.MPIArray((size * 5, 20), axis=0)

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

    assert (darr == local_array).all()

    # Check a partial assignment along the parallel axis
    darr.global_slice[7:, 7:9] = -3.0
    whole_array[7:, 7:9] = -3.0

    assert (darr == local_array).all()

    # Check assignment of a single index on the parallel axis
    darr.global_slice[6] = np.arange(20.0)
    whole_array[6] = np.arange(20.0)

    assert (darr == local_array).all()

    # Check copy of one column into the other
    darr.global_slice[:, 8] = darr.global_slice[:, 9]
    whole_array[:, 8] = whole_array[:, 9]

    assert (darr == local_array).all()

    # test setting complex dtypes

    darr_complex = mpiarray.MPIArray((size * 5, 20), axis=0, dtype=np.complex64)

    darr_complex[:] = 4
    assert darr_complex.dtype == np.complex64
    assert (darr_complex == 4.0 + 0.0j).all()

    darr_complex[:] = 2.0 + 1.345j

    assert darr_complex.dtype == np.complex64
    assert (darr_complex == 2.0 + 1.345j).all()

    darr_float = mpiarray.MPIArray((size * 5, 20), axis=0, dtype=np.float64)

    darr_float[:] = 4.0
    assert darr_float.dtype == np.float64
    assert (darr_float == 4.0).all()


def test_outer_ufunc():
    rank = mpiutil.rank
    size = mpiutil.size

    dist_arr = mpiarray.MPIArray((size, 4), axis=0)
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
    dist_arr_2 = mpiarray.MPIArray((size, 1), axis=0)
    dist_arr_2[:] = rank - 1
    assert (dist_arr + dist_arr_2 == 2 * rank - 1).all()
    assert (dist_arr + dist_arr_2).axis == 0

    # check that subtracting arrays with two different distributed axis fails
    # pylint: disable=expression-not-assigned
    with pytest.raises(mpiarray.AxisException):
        mpiarray.MPIArray((size, 4), axis=0) - mpiarray.MPIArray((size, 4), axis=1)

    # check that outer ufunc on arrays that cannot be broadcast fails
    with pytest.raises(ValueError):
        np.multiply(
            mpiarray.MPIArray((size, 3), axis=0), mpiarray.MPIArray((size, 4), axis=0)
        )
    # pylint: enable=expression-not-assigned

    # test ufuncs with complex dtypes

    dist_complex = mpiarray.MPIArray((size, 4), axis=0, dtype=np.complex64)
    dist_complex_add = dist_complex + dist_complex

    assert (dist_complex_add == dist_complex + dist_complex).all()
    assert dist_complex_add.dtype == np.complex64


def test_reduce():

    rank = mpiutil.rank
    size = mpiutil.size

    dist_array = mpiarray.MPIArray((size, 4, 3), axis=0)
    dist_array[:] = rank

    # sums across non-distributed axes should be permitted, and work as usual
    assert (dist_array.sum(axis=1) == 4 * rank).all()

    # sum() should reduce across all non-distributed axes
    sum_all = dist_array.sum()
    assert (sum_all == 4 * 3 * rank).all()

    assert sum_all.local_shape == (1,)

    assert sum_all.global_shape == (size,)

    # sum across a smaller numbered axes
    # this will result in an axes reduction
    dist_array = mpiarray.MPIArray((size, 4, 3), axis=1)
    dist_array[:] = rank

    assert (dist_array.sum(axis=0) == 4 * rank).all()

    # check that the new axes was calculated accordingly
    assert (dist_array.sum(axis=0)).axis == 0

    assert dist_array.sum(axis=0, keepdims=True).axis == 1

    sum_all = dist_array.sum()
    assert (sum_all == 4 * 3 * rank).all()

    assert sum_all.local_shape == (1,)

    assert sum_all.global_shape == (size,)

    assert mpiarray.MPIArray((size, 4), axis=1).sum(axis=0).axis == 0

    # test AllReduce
    if size > 1:
        from mpi4py import MPI

        # Test comm.Allreduce
        dist_array = mpiarray.MPIArray((size, 4), axis=1)
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


#
# class Testmpiarray(unittest.TestCase):
#
#     def test_dataset(self):
#
#         fname = 'testdset.hdf5'
#
#         if mpiutil.rank0:
#             if os.path.exists(fname):
#                 os.remove(fname)
#
#         mpiutil.barrier()
#
#         class TestDataset(mpiarray.mpiarray):
#             _common = {'a': None}
#             _distributed = {'b': None}
#
#         td1 = TestDataset()
#         td1.common['a'] = np.arange(12)
#         td1.attrs['message'] = 'meh'
#
#         gshape = (19, 17)
#         ds = mpiarray.MPIArray(gshape, dtype=np.float64)
#         ds[:] = np.random.standard_normal(ds.local_shape)
#
#         td1.distributed['b'] = ds
#         td1.to_hdf5(fname)
#
#         td2 = TestDataset.from_hdf5(fname)
#
#         assert (td1['a'] == td2['a']).all()
#         assert (td1['b'] == td2['b']).all()
#         assert (td1.attrs['message'] == td2.attrs['message'])
