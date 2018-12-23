"""Tests of MPI Array functionality.

Designed to be run as an MPI job with four processes like::

    $ mpirun -np 4 python test_mpiarray.py
"""
# === Start Python 2/3 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
# === End Python 2/3 compatibility

import os
import unittest

import numpy as np

from caput import mpiutil, mpiarray



class TestMPIAray(unittest.TestCase):

    def test_construction(self):

        arr = mpiarray.MPIArray((10, 11), axis=1)

        l, s, e = mpiutil.split_local(11)

        # Check that global shape is set correctly
        assert arr.global_shape == (10, 11)

        assert arr.shape == (10, l)

        assert arr.local_offset == (0, s)

        assert arr.local_shape == (10, l)

    def test_redistribution(self):

        gshape = (1, 11, 2, 14, 3, 4)
        nelem = np.prod(gshape)
        garr = np.arange(nelem).reshape(gshape)

        l0, s0, e0 = mpiutil.split_local(11)
        l1, s1, e1 = mpiutil.split_local(14)
        l2, s2, e2 = mpiutil.split_local(4)

        arr = mpiarray.MPIArray(gshape, axis=1, dtype=np.int64)
        arr[:] = garr[:, s0:e0]

        arr2 = arr.redistribute(axis=3)
        assert (arr2 == garr[:, :, :, s1:e1]).view(np.ndarray).all()

        arr3 = arr.redistribute(axis=5)
        assert (arr3 == garr[:, :, :, :, :, s2:e2]).view(np.ndarray).all()

    def test_wrap(self):

        ds = mpiarray.MPIArray((10, 17))

        df = np.fft.rfft(ds, axis=1)

        assert type(df) == np.ndarray

        da = mpiarray.MPIArray.wrap(df, axis=0)

        assert type(da) == mpiarray.MPIArray
        assert da.global_shape == (10, 9)

        l0, s0, e0 = mpiutil.split_local(10)

        assert da.local_shape == (l0, 9)

        if mpiutil.rank0:
            df = df[:-1]
        
        if mpiutil.size > 1:
            with self.assertRaises(Exception):
                mpiarray.MPIArray.wrap(df, axis=0)

    def test_io(self):

        import h5py

        # Cleanup directories
        fname = 'testdset.hdf5'

        if mpiutil.rank0 and os.path.exists(fname):
            os.remove(fname)

        mpiutil.barrier()

        gshape = (19, 17)

        ds = mpiarray.MPIArray(gshape, dtype=np.int64)

        ga = np.arange(np.prod(gshape)).reshape(gshape)

        l0, s0, e0 = mpiutil.split_local(gshape[0])
        ds[:] = ga[s0:e0]

        ds.redistribute(axis=1).to_hdf5(fname, 'testds', create=True)

        if mpiutil.rank0:

            with h5py.File(fname, 'r') as f:

                h5ds = f['testds'][:]

                assert (h5ds == ga).all()

        ds2 = mpiarray.MPIArray.from_hdf5(fname, 'testds')

        assert (ds2 == ds).all()

        mpiutil.barrier()

        if mpiutil.rank0 and os.path.exists(fname):
            os.remove(fname)

    def test_transpose(self):

        gshape = (1, 11, 2, 14)

        l0, s0, e0 = mpiutil.split_local(11)

        arr = mpiarray.MPIArray(gshape, axis=1, dtype=np.int64)

        arr2 = arr.transpose((1, 3, 0, 2))

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

    def test_reshape(self):

        gshape = (1, 11, 2, 14)

        l0, s0, e0 = mpiutil.split_local(11)

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

    def test_global_getslice(self):

        rank = mpiutil.rank
        size = mpiutil.size


        darr = mpiarray.MPIArray((size*5, 20), axis=0)

        # Initialise the distributed array
        for li, gi in darr.enumerate(axis=0):
            darr[li] = 10*(10*rank+li) + np.arange(20)


        # Construct numpy array which should be equivalent to the global array
        whole_array = 10*(10*np.arange(4.0)[:, np.newaxis]+np.arange(5.0)[np.newaxis, :]).flatten()[:, np.newaxis] + np.arange(20)[np.newaxis, :]

        # Extract the section for each rank distributed along axis=0
        local_array = whole_array[(rank*5):((rank+1)*5)]

        # Extract the correct section for each rank distributed along axis=0
        local_array_T = whole_array[:, (rank*5):((rank+1)*5)]


        # Check that these are the same
        assert (local_array == darr).all()


        # Check a simple slice on the non-parallel axis
        arr = darr.global_slice[:, 3:5]
        res = local_array[:, 3:5]

        assert isinstance(arr, mpiarray.MPIArray)
        assert (arr == res).all()


        # Check a single element extracted from the non-parallel axis
        arr = darr.global_slice[:, 3]
        res = local_array[:, 3]
        assert (arr == res).all()

        # These tests denpend on the size being at least 2.
        if size > 1:
            # Check a slice on the parallel axis
            arr = darr.global_slice[:7, 3:5]

            res = { 0 : local_array[:, 3:5],
                    1 : local_array[:2, 3:5],
                    2 : None,
                    3 : None }

            assert arr == res[rank] if arr is None else (arr == res[rank]).all()


            # Check a single element from the parallel axis
            arr = darr.global_slice[7, 3:5]

            res = { 0 : None,
                    1 : local_array[2, 3:5],
                    2 : None,
                    3 : None }

            assert arr == res[rank] if arr is None else (arr == res[rank]).all()


            # Check a slice on the redistributed parallel axis
            darr_T = darr.redistribute(axis=1)
            arr = darr_T.global_slice[3:5, :7]

            res = { 0 : local_array_T[3:5, :],
                    1 : local_array_T[3:5, :2],
                    2 : None,
                    3 : None }

            assert arr == res[rank] if arr is None else (arr == res[rank]).all()

        # Check a slice that removes an axis
        darr = mpiarray.MPIArray((10, 20, size*5), axis=2)
        dslice = darr.global_slice[:, 0, :]

        assert dslice.global_shape == (10, size*5)
        assert dslice.local_shape == (10, 5)

        # Check ellipsis and slice at the end
        darr = mpiarray.MPIArray((size*5, 20, 10), axis=0)
        dslice = darr.global_slice[..., 4:9]

        assert dslice.global_shape == (size*5, 20, 5)
        assert dslice.local_shape == (5, 20, 5)

        # Check slice that goes off the end of the axis
        darr = mpiarray.MPIArray((size, 136, 2048), axis=0)
        dslice = darr.global_slice[..., 2007:2087]

        assert dslice.global_shape == (size, 136, 41)
        assert dslice.local_shape == (1, 136, 41)


    def test_global_setslice(self):

        rank = mpiutil.rank
        size = mpiutil.size


        darr = mpiarray.MPIArray((size*5, 20), axis=0)

        # Initialise the distributed array
        for li, gi in darr.enumerate(axis=0):
            darr[li] = 10*(10*rank+li) + np.arange(20)


        # Construct numpy array which should be equivalent to the global array
        whole_array = 10*(10*np.arange(4.0)[:, np.newaxis]+np.arange(5.0)[np.newaxis, :]).flatten()[:, np.newaxis] + np.arange(20)[np.newaxis, :]

        # Extract the section for each rank distributed along axis=0
        local_array = whole_array[(rank*5):((rank+1)*5)]
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


if __name__ == '__main__':
    unittest.main()
