import os
import unittest

import numpy as np

from caput import mpiutil, mpidataset



class TestMPIAray(unittest.TestCase):

    def test_construction(self):

        arr = mpidataset.MPIArray((10, 11), axis=1)

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

        arr = mpidataset.MPIArray(gshape, axis=1, dtype=np.int64)
        arr[:] = garr[:, s0:e0]

        arr2 = arr.redistribute(axis=3)
        assert (arr2 == garr[:, :, :, s1:e1]).view(np.ndarray).all()

        arr3 = arr.redistribute(axis=5)
        assert (arr3 == garr[:, :, :, :, :, s2:e2]).view(np.ndarray).all()

    def test_wrap(self):

        ds = mpidataset.MPIArray((10, 17))

        df = np.fft.rfft(ds, axis=1)

        assert type(df) == np.ndarray

        da = mpidataset.MPIArray.wrap(df, axis=0)

        assert type(da) == mpidataset.MPIArray
        assert da.global_shape == (10, 9)

        l0, s0, e0 = mpiutil.split_local(10)

        assert da.local_shape == (l0, 9)

        if mpiutil.rank0:
            df = df[:-1]

        with self.assertRaises(Exception):
            mpidataset.MPIArray.wrap(df, axis=0)

    def test_io(self):

        import h5py

        # Cleanup directories
        fname = 'testdset.hdf5'

        if os.path.exists(fname):
            os.remove(fname)

        gshape = (19, 17)

        ds = mpidataset.MPIArray(gshape, dtype=np.int64)

        ga = np.arange(np.prod(gshape)).reshape(gshape)

        l0, s0, e0 = mpiutil.split_local(gshape[0])
        ds[:] = ga[s0:e0]

        ds.redistribute(axis=1).to_hdf5(fname, 'testds')

        if mpiutil.rank0:

            with h5py.File(fname, 'r') as f:

                h5ds = f['testds'][:]

                assert (h5ds == ga).all()

        ds2 = mpidataset.MPIArray.from_hdf5(fname, 'testds')

        assert (ds2 == ds).all()

    def test_transpose(self):

        gshape = (1, 11, 2, 14)

        l0, s0, e0 = mpiutil.split_local(11)

        arr = mpidataset.MPIArray(gshape, axis=1, dtype=np.int64)

        arr2 = arr.transpose((1, 3, 0, 2))

        # Check type
        assert isinstance(arr2, mpidataset.MPIArray)

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

        arr = mpidataset.MPIArray(gshape, axis=1, dtype=np.int64)

        arr2 = arr.reshape((None, 28))

        # Check type
        assert isinstance(arr2, mpidataset.MPIArray)

        # Check global shape
        assert arr2.global_shape == (11, 28)

        # Check local shape
        assert arr2.local_shape == (l0, 28)

        # Check local offset
        assert arr2.local_offset == (s0, 0)

        # Check axis
        assert arr2.axis == 0


class TestMPIDataset(unittest.TestCase):

    def test_dataset(self):

        fname = 'testdset.hdf5'

        if mpiutil.rank0:
            if os.path.exists(fname):
                os.remove(fname)

        mpiutil.barrier()

        class TestDataset(mpidataset.MPIDataset):
            _common = {'a': None}
            _distributed = {'b': None}

        td1 = TestDataset()
        td1.common['a'] = np.arange(12)
        td1.attrs['message'] = 'meh'

        gshape = (19, 17)
        ds = mpidataset.MPIArray(gshape, dtype=np.float64)
        ds[:] = np.random.standard_normal(ds.local_shape)

        td1.distributed['b'] = ds
        td1.to_hdf5(fname)

        td2 = TestDataset.from_hdf5(fname)

        assert (td1['a'] == td2['a']).all()
        assert (td1['b'] == td2['b']).all()
        assert (td1.attrs['message'] == td2.attrs['message'])


if __name__ == '__main__':
    unittest.main()
