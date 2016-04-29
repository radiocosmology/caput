"""Unit tests for the parallel features of the memh5 module."""

import unittest
import os
import glob

import numpy as np
import h5py

from caput import memh5, mpiarray, mpiutil


comm = mpiutil.world
rank, size = mpiutil.rank, mpiutil.size

class TestMemGroupDistributed(unittest.TestCase):
    """Unit tests for MemGroup."""

    fname = 'tmp_test_memh5_distributed.h5'

    def test_create_dataset(self):

        global_data = np.arange(size*5*10, dtype=np.float32)
        local_data = global_data.reshape(size, -1, 10)[rank]
        d_array = mpiarray.MPIArray.wrap(local_data, axis=0)
        d_array_T = d_array.redistribute(axis=1)

        # Check that we must specify in advance if the dataset is distributed
        g = memh5.MemGroup()
        if comm is not None:
            self.assertRaises(RuntimeError, g.create_dataset, 'data', data=d_array)

        g = memh5.MemGroup(distributed=True)

        # Create an array from data
        g.create_dataset('data', data=d_array, distributed=True)

        # Create an array from data with a different distribution
        g.create_dataset('data_T', data=d_array, distributed=True, distributed_axis=1)

        # Create an empty array with a specified shape
        g.create_dataset('data2', shape=(size*5, 10), dtype=np.float64, distributed=True, distributed_axis=1)
        self.assertTrue(np.allclose(d_array, g['data'][:]))
        self.assertTrue(np.allclose(d_array_T, g['data_T'][:]))
        if comm is not None:
            self.assertEqual(d_array_T.local_shape, g['data2'].local_shape)

        # Test global indexing
        self.assertTrue((g['data'][rank*5] == local_data[0]).all())

    def test_io(self):

        # Create distributed memh5 object
        g = memh5.MemGroup(distributed=True)
        g.attrs['rank'] = rank

        # Create an empty array with a specified shape
        pdset = g.create_dataset('parallel_data', shape=(size, 10), dtype=np.float64, distributed=True, distributed_axis=0)
        pdset[:] = rank
        pdset.attrs['rank'] = rank

        # Create an empty array with a specified shape
        sdset = g.create_dataset('serial_data', shape=(size*5, 10), dtype=np.float64)
        sdset[:] = rank
        sdset.attrs['rank'] = rank

        # Create nested groups
        g.create_group('hello/world')

        g.to_hdf5(self.fname)

        # Test that the HDF5 file has the correct structure
        with h5py.File(self.fname, 'r') as f:

            # Test that the file attributes are correct
            self.assertTrue(f['parallel_data'].attrs['rank'] == 0)

            # Test that the parallel dataset has been written correctly
            self.assertTrue((f['parallel_data'][:, 0] == np.arange(size)).all())
            self.assertTrue(f['parallel_data'].attrs['rank'] == 0)

            # Test that the common dataset has been written correctly (i.e. by rank=0)
            self.assertTrue((f['serial_data'][:] == 0).all())
            self.assertTrue(f['serial_data'].attrs['rank'] == 0)

            # Check group structure is correct
            self.assertIn('hello', f)
            self.assertIn('world', f['hello'])

        # Test that the read in group has the same structure as the original
        g2 = memh5.MemGroup.from_hdf5(self.fname, distributed=True)

        # Check that the parallel data is still the same
        self.assertTrue((g2['parallel_data'][:] == g['parallel_data'][:]).all())

        # Check that the serial data is all zeros (should not be the same as before)
        self.assertTrue((g2['serial_data'][:] == np.zeros_like(sdset[:])).all())

        # Check group structure is correct
        self.assertIn('hello', g2)
        self.assertIn('world', g2['hello'])

        # Check the attributes
        self.assertTrue(g2['parallel_data'].attrs['rank'] == 0)
        self.assertTrue(g2['serial_data'].attrs['rank'] == 0)

    def tearDown(self):
        if rank == 0:
            file_names = glob.glob(self.fname + '*')
            for fname in file_names:
                os.remove(fname)


class TestMemDiskGroupDistributed(unittest.TestCase):

    fname = 'tmp_parallel_dg.h5'

    def test_misc(self):

        dg = memh5.MemDiskGroup(distributed=True)

        pdset = dg.create_dataset('parallel_data', shape=(10,), dtype=np.float64, distributed=True, distributed_axis=0)
        # pdset[:] = dg._data.comm.rank
        pdset[:] = rank
        # Test successfully added
        self.assertIn('parallel_data', dg)

        dg.save(self.fname)

        dg2 = memh5.MemDiskGroup.from_file(self.fname, distributed=True)

        # Test successful load
        self.assertIn('parallel_data', dg2)
        self.assertTrue((dg['parallel_data'][:] == dg2['parallel_data'][:]).all())

        # self.assertRaises(NotImplementedError, dg.to_disk, self.fname)

        # Test refusal to base off a h5py object when distributed
        from caput import mpiutil
        with h5py.File(self.fname, 'r') as f:
            if comm is not None:
                self.assertRaises(ValueError, memh5.MemDiskGroup, data_group=f, distributed=True)
        mpiutil.barrier()

    def tearDown(self):

        if rank == 0:
            file_names = glob.glob(self.fname + '*')
            for fname in file_names:
                os.remove(fname)

if __name__ == '__main__':
    unittest.main()
