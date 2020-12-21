"""Serial version of the selection tests."""
from caput.memh5 import MemGroup

import pytest
import glob
import numpy as np
import os


len_axis = 8

dset1 = np.arange(len_axis * len_axis * len_axis)
dset1 = dset1.reshape((len_axis, len_axis, len_axis))

dset2 = np.arange(len_axis * len_axis)
dset2 = dset2.reshape((len_axis, len_axis))

freqs = np.arange(len_axis)
inputs = np.arange(len_axis)
ra = np.arange(len_axis)

fsel = slice(1, 8, 2)
isel = slice(1, 4)
sel = {"dset1": (fsel, isel, slice(None)), "dset2": (fsel, slice(None))}


@pytest.fixture
def container_on_disk():
    fname = "tmp_test_memh5_select.h5"
    container = MemGroup()
    container.create_dataset("dset1", data=dset1.view())
    container.create_dataset("dset2", data=dset2.view())
    container.to_hdf5(fname)
    yield fname

    # tear down
    file_names = glob.glob(fname + "*")
    for fname in file_names:
        os.remove(fname)


def test_H5FileSelect(container_on_disk):
    """Tests that makes hdf5 objects and tests selecting on their axes."""

    m = MemGroup.from_hdf5(container_on_disk, selections=sel)
    assert np.all(m["dset1"][:] == dset1[(fsel, isel, slice(None))])
    assert np.all(m["dset2"][:] == dset2[(fsel, slice(None))])
