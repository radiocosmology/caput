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
        self.assertEqual(cache_tools.get_package_sha1(caput), commit)
        self.assertEqual(cache_tools.get_package_sha1(cache_tools), commit)

        


if __name__ == '__main__':
    unittest.main()
