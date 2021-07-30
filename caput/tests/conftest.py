"""Pytest fixtures and simple tasks that can be used by all unit tests."""

import numpy as np
import pytest

from caput.pipeline import PipelineStopIteration, TaskBase, IterBase
from caput import config


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
