"""Pytest fixtures that can be used by all unit tests."""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def datasets():
    """A couple of simple numpy arrays."""
    len_axis = 8

    dset1 = np.arange(len_axis * len_axis * len_axis)
    dset1 = dset1.reshape((len_axis, len_axis, len_axis))

    dset2 = np.arange(len_axis * len_axis)
    dset2 = dset2.reshape((len_axis, len_axis))

    return dset1, dset2
