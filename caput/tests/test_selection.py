"""Serial version of the selection tests."""
import pytest

import numpy as np

from caput.memh5 import MemGroup
from caput import fileformats
from caput.tests.conftest import rm_all_files

fsel = slice(1, 8, 2)
isel = slice(1, 4)
sel = {"dset1": (fsel, isel, slice(None)), "dset2": (fsel, slice(None))}


@pytest.fixture
def h5_file_select(datasets, h5_file):
    container = MemGroup()
    container.create_dataset("dset1", data=datasets[0].view())
    container.create_dataset("dset2", data=datasets[1].view())
    container.to_hdf5(h5_file)
    yield h5_file, datasets
    rm_all_files(h5_file)


@pytest.fixture
def zarr_file_select(datasets, zarr_file):
    container = MemGroup()
    container.create_dataset("dset1", data=datasets[0].view())
    container.create_dataset("dset2", data=datasets[1].view())
    container.to_file(zarr_file, file_format=fileformats.Zarr)
    yield zarr_file, datasets
    rm_all_files(zarr_file)


@pytest.mark.parametrize(
    "container_on_disk, file_format",
    [
        (pytest.lazy_fixture("h5_file_select"), fileformats.HDF5),
        (pytest.lazy_fixture("zarr_file_select"), fileformats.Zarr),
    ],
)
def test_H5FileSelect(container_on_disk, file_format):
    """Tests that makes hdf5 objects and tests selecting on their axes."""

    m = MemGroup.from_file(
        container_on_disk[0], selections=sel, file_format=file_format
    )
    assert np.all(m["dset1"][:] == container_on_disk[1][0][(fsel, isel, slice(None))])
    assert np.all(m["dset2"][:] == container_on_disk[1][1][(fsel, slice(None))])
