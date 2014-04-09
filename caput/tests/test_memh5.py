"""Unit tests for the memh5 module."""

import unittest
import os
import glob

import numpy as np
import h5py

from caput import memh5

class TestRODict(unittest.TestCase):
    """Unit tests for ro_dict."""

    def test_everything(self):
        a = {'a' : 5}
        a = memh5.ro_dict(a)
        self.assertEqual(a['a'], 5)
        self.assertEqual(a.keys(), ['a'])
        # Convoluded test to make sure you can't write to it.
        try: a['b'] = 6
        except TypeError: correct = True
        else: correct = False
        self.assertTrue(correct)


class TestGroup(unittest.TestCase):
    """Unit tests for MemGroup."""

    def test_nested(self):
        root = memh5.MemGroup()
        l1 = root.create_group('level1')
        l2 = l1.require_group('level2')
        self.assertTrue(root['level1'] is l1)
        self.assertTrue(root['level1/level2'] is l2)
        self.assertEqual(root['level1/level2'].name, '/level1/level2')

    def test_create_dataset(self):
        g = memh5.MemGroup()
        data = np.arange(100, dtype=np.float32)
        g.create_dataset('data', data=data)
        self.assertTrue(np.allclose(data, g['data']))

    def test_recursive_create(self):
        g = memh5.MemGroup()
        self.assertRaises(ValueError, g.create_group, '')
        g2 = g.create_group('level2/')
        self.assertRaises(ValueError, g2.create_group, '/')
        g2.create_group('/level22')
        self.assertEqual(set(g.keys()), {'level22', 'level2'}) 
        g.create_group('/a/b/c/d/')
        gd = g['/a/b/c/d/']
        self.assertEqual(gd.name, '/a/b/c/d')

class TestH5Files(unittest.TestCase):
    """Tests that make hdf5 objects, convert to mem and back."""

    def setUp(self):
        self.fname = 'tmp_test_memh5.h5'
        f = h5py.File(self.fname)
        l1 = f.create_group('level1')
        l2 = l1.create_group('level2')
        d1 = l1.create_dataset('large', data=np.arange(100))
        f.attrs['a'] = 5
        d1.attrs['b'] = 6
        l2.attrs['small'] = np.arange(3)
        f.close()

    def assertGroupsEqual(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        self.assertAttrsEqual(a.attrs, b.attrs)
        for key in a.keys():
            this_a = a[key]
            this_b = b[key]
            if not memh5.is_group(a[key]):
                self.assertAttrsEqual(this_a.attrs, this_b.attrs)
                self.assertTrue(np.allclose(this_a, this_b))
            else:
                self.assertGroupsEqual(this_a, this_b)

    def assertAttrsEqual(self, a, b):
        self.assertEqual(a.keys(), b.keys())
        for key in a.keys():
            this_a = a[key]
            this_b = b[key]
            if hasattr(this_a, 'shape'):
                self.assertTrue(np.allclose(this_a, this_b))
            else:
                self.assertEqual(this_a, this_b)

    def test_h5_sanity(self):
        f = h5py.File(self.fname)
        self.assertGroupsEqual(f, f)
        f.close()

    def test_to_from_hdf5(self):
        m = memh5.MemGroup.from_hdf5(self.fname)
        f = h5py.File(self.fname)
        self.assertGroupsEqual(f, m)
        f.close()
        f = m.to_hdf5(self.fname + '.new')
        self.assertGroupsEqual(f, m)
        f.close()

    def tearDown(self):
        file_names = glob.glob(self.fname + '*')
        for fname in file_names:
            os.remove(fname)


if __name__ == '__main__':
    unittest.main()
