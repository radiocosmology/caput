"""Pytest fixtures and simple tasks that can be used by all unit tests."""

import glob
import os
import shutil

import numpy as np
import pytest

from caput.pipeline import PipelineStopIteration, TaskBase, IterBase
from caput import config, mpiutil


@pytest.fixture(scope="session")
def datasets():
    """A couple of simple numpy arrays."""
    len_axis = 8

    dset1 = np.arange(len_axis * len_axis * len_axis)
    dset1 = dset1.reshape((len_axis, len_axis, len_axis))

    dset2 = np.arange(len_axis * len_axis)
    dset2 = dset2.reshape((len_axis, len_axis))

    return dset1, dset2


class PrintEggs(TaskBase):
    """Simple task used for testing."""

    eggs = config.Property(proptype=list)

    def __init__(self, *args, **kwargs):
        self.i = 0
        super().__init__(*args, **kwargs)

    def setup(self, requires=None):
        print("Setting up PrintEggs.")

    def next(self, _input=None):
        if self.i >= len(self.eggs):
            raise PipelineStopIteration()
        print("Spam and %s eggs." % self.eggs[self.i])
        self.i += 1

    def finish(self):
        print("Finished PrintEggs.")


class GetEggs(TaskBase):
    """Simple task used for testing."""

    eggs = config.Property(proptype=list)

    def __init__(self, *args, **kwargs):
        self.i = 0
        super().__init__(*args, **kwargs)

    def setup(self, requires=None):
        print("Setting up GetEggs.")

    def next(self, _input=None):
        if self.i >= len(self.eggs):
            raise PipelineStopIteration()
        egg = self.eggs[self.i]
        self.i += 1
        return egg

    def finish(self):
        print("Finished GetEggs.")


class CookEggs(IterBase):
    """Simple task used for testing."""

    style = config.Property(proptype=str)

    def setup(self, requires=None):
        print("Setting up CookEggs.")

    def process(self, _input):
        print("Cooking %s %s eggs." % (self.style, input))

    def finish(self):
        print("Finished CookEggs.")

    def read_input(self, filename):
        raise NotImplementedError()

    def read_output(self, filename):
        raise NotImplementedError()

    def write_output(self, filename, output):
        raise NotImplementedError()


@pytest.fixture
def h5_file():
    """Provides a file name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5.h5"
    yield fname
    rm_all_files(fname)


@pytest.fixture
def zarr_file():
    """Provides a directory name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5.zarr"
    yield fname
    rm_all_files(fname)


@pytest.fixture
def h5_file_distributed():
    """Provides a file name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5_distributed.h5"
    yield fname
    if mpiutil.rank == 0:
        rm_all_files(fname)


@pytest.fixture
def zarr_file_distributed():
    """Provides a directory name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5.zarr"
    yield fname
    if mpiutil.rank == 0:
        rm_all_files(fname)


def rm_all_files(file_name):
    """Remove all files and directories starting with `file_name`."""
    file_names = glob.glob(file_name + "*")
    for fname in file_names:
        if os.path.isdir(fname):
            try:
                shutil.rmtree(fname)
            except FileNotFoundError:
                pass
        else:
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass
