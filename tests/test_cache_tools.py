"""Unit tests for the cache_tools module."""

import unittest
import os
from os import path
import subprocess

import numpy as np

import caput
from caput import cache_tools


# Figure out the path of the `.git` directory.
caput_init_file = path.abspath(caput.__file__)
git_dir = caput_init_file.split('/')[:-2]
git_dir = '/'.join(git_dir) + '/.git'
if not path.isdir(git_dir):
    msg = ("Unit tests assume caput code is under git version"
           " control.")
    raise RuntimeError(msg)

class TestGitSHA1(unittest.TestCase):
    
    def test_find_git(self):
        self.assertEqual(cache_tools.get_git_dir(caput), git_dir)
        self.assertEqual(cache_tools.get_git_dir(cache_tools), git_dir)
        self.assertRaises(cache_tools.UnversionedError,
                          cache_tools.get_git_dir, np)
        self.assertRaises(TypeError, cache_tools.get_git_dir, np.array)

    def test_get_SHA1(self):
        # Run `git show` to get the current commit of caput.
        call = ['git', '--git-dir=' + git_dir, 'show'] 
        proc = subprocess.Popen(call, stdout=subprocess.PIPE)
        ret = proc.wait()
        if not ret == 0:
            raise RuntimeError("Couldn't run git.")
        # Parse stout.
        commit_line = proc.stdout.readline()
        commit = commit_line.split()[1]
        self.assertEqual(cache_tools.get_package_commit(caput), commit)
        self.assertEqual(cache_tools.get_package_commit(cache_tools), commit)

class TestHashParams(unittest.TestCase):

    def test_hash_equal(self):
        """Make sure two equal dictionaries hash the same."""
        
        # Two dictionaries that are the same but defined differently.
        a = {'a' : 5, 'b' : (10, 6, 8.000), 'aa' : {'f' : 'fff', 'g': 'gg'},
             5 : {'s', 1, 3}, 10 : 3j}
        b = {'aa' : {'g': 'gg', 'f' : 'fff'}, 'a' : 5, 'b' : [10, 6 + 0j, 8.],
             5 : {3, 's', 1}, 10 : 0 + 3.0j}
        self.assertEqual(cache_tools.hash_obj(a), cache_tools.hash_obj(b))
        # Break Equality.
        b['b'][2] = 8.001
        self.assertNotEqual(cache_tools.hash_obj(a), cache_tools.hash_obj(b))
        # Regression test.
        self.assertEqual(cache_tools.hash_obj(a), 
                         '0dc008e6282cc273ca6d5bfb7b8d1c6998623fbb')

    def test_equivalent(self):
        """Manually construct a few hashes."""

        h = cache_tools.hash_obj
        self.assertEqual(h(5.000 + 0.00j), h(h('num') + '5'))
        self.assertEqual(h(None), h(h('null')))
        self.assertEqual(h([7, 'a', 1]), h(h('seq') + h(7) + h('a') + h(1)))
        self.assertEqual(h({7, 'a', 1, 42}),
                 h(h('set') + ''.join(sorted([h(7), h(42), h('a'), h(1)]))))
        
        d = {4 : 5, 'a' : 'b', 'c' : 42}
        dhash = [[h(4), h(5)], [h('a'), h('b')], [h('c'), h(42)]]
        dhash = sorted(dhash)
        dhash = [ x[0] + x[1] for x in dhash ]
        dhash = h('map') + ''.join(dhash)
        self.assertEqual(h(d), h(dhash))
        


        


if __name__ == '__main__':
    unittest.main()
