"""Test the miscellaneous tools."""

import unittest
import tempfile
import os
import pytest
import shutil

from caput import misc


class TestLock(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_lock_new(self):
        """Test the normal behaviour"""

        base = "newfile.dat"
        newfile_name = os.path.join(self.dir, base)
        lockfile_name = os.path.join(self.dir, "." + base + ".lock")

        with misc.lock_file(newfile_name) as fname:

            # Check lock file has been created
            self.assertTrue(os.path.exists(lockfile_name))

            # Create a stub file
            with open(fname, "w+") as fh:
                fh.write("hello")

            # Check the file exists only at the temporary path
            self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(newfile_name))

        # Check the file exists at the final path and the lock file removed
        self.assertTrue(os.path.exists(newfile_name))
        self.assertFalse(os.path.exists(lockfile_name))

    def test_lock_exception(self):
        """Check what happens in an exception"""

        base = "newfile2.dat"
        newfile_name = os.path.join(self.dir, base)
        lockfile_name = os.path.join(self.dir, "." + base + ".lock")

        with pytest.raises(RuntimeError):
            with misc.lock_file(newfile_name) as fname:

                # Create a stub file
                with open(fname, "w+") as fh:
                    fh.write("hello")

                raise RuntimeError("Test error")

        # Check that neither the file, nor its lock exists
        self.assertFalse(os.path.exists(newfile_name))
        self.assertFalse(os.path.exists(lockfile_name))

    def test_lock_exception_preserve(self):
        """Check what happens in an exception when asked to preserve the temp file"""

        base = "newfile3.dat"
        newfile_name = os.path.join(self.dir, base)
        lockfile_name = os.path.join(self.dir, "." + base + ".lock")
        tmpfile_name = os.path.join(self.dir, "." + base)

        with pytest.raises(RuntimeError):
            with misc.lock_file(newfile_name, preserve=True) as fname:

                # Create a stub file
                with open(fname, "w+") as fh:
                    fh.write("hello")

                raise RuntimeError("Test error")

        # Check that neither the file, nor its lock exists, but that the
        # temporary file does
        self.assertTrue(os.path.exists(tmpfile_name))
        self.assertFalse(os.path.exists(newfile_name))
        self.assertFalse(os.path.exists(lockfile_name))

    def tearDown(self):
        shutil.rmtree(self.dir)


if __name__ == "__main__":
    unittest.main()
