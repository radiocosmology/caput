"""Serial version of the selection tests."""
import glob
import os
import pytest

import numpy as np

from caput.memh5 import MemGroup

fsel = slice(1, 8, 2)
isel = slice(1, 4)
ind = [0, 2, 7]
sel = {"dset1": (fsel, isel, slice(None)), "dset2": (fsel, slice(None))}
index_sel = {"dset1": (fsel, ind, slice(None)), "dset2": (ind, slice(None))}


@pytest.fixture
def container_on_disk(datasets):
    fname = "tmp_test_memh5_select.h5"
    container = MemGroup()
    container.create_dataset("dset1", data=datasets[0].view())
    container.create_dataset("dset2", data=datasets[1].view())
    container.to_hdf5(fname)
    yield fname, datasets

    # tear down
    file_names = glob.glob(fname + "*")
    for fname in file_names:
        os.remove(fname)


def test_H5FileSelect(container_on_disk):
    """Tests that makes hdf5 objects and tests selecting on their axes."""

    m = MemGroup.from_hdf5(container_on_disk[0], selections=sel)
    assert np.all(m["dset1"][:] == container_on_disk[1][0][(fsel, isel, slice(None))])
    assert np.all(m["dset2"][:] == container_on_disk[1][1][(fsel, slice(None))])

    # now test index selection
    m = MemGroup.from_hdf5(container_on_disk[0], selections=index_sel)
    assert np.all(m["dset1"][:] == container_on_disk[1][0][index_sel["dset1"]])
    assert np.all(m["dset2"][:] == container_on_disk[1][1][index_sel["dset2"]])
