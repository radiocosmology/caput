"""Serial version of the selection tests."""

import pytest
from pytest_lazy_fixtures import lf
import numpy as np

from mpi4py import MPI

from caput import mpiarray
from caput.util import mpitools
from caput.memdata import fileformats, MemGroup


comm = MPI.COMM_WORLD


fsel = slice(1, 8, 2)
isel = slice(1, 4)
ind = [0, 2, 7]
sel = {"dset1": (fsel, isel, slice(None)), "dset2": (fsel, slice(None))}
index_sel = {"dset1": (fsel, ind, slice(None)), "dset2": (ind, slice(None))}


@pytest.fixture
def standard_h5_file_select(datasets, h5_file, rm_all_files):
    """Provides a file with some content for testing."""
    container = MemGroup()
    container.create_dataset("dset1", data=datasets[0].view())
    container.create_dataset("dset2", data=datasets[1].view())
    container.to_file(h5_file, file_format=fileformats.HDF5)

    yield h5_file, datasets

    rm_all_files(h5_file)


@pytest.fixture
def standard_zarr_file_select(datasets, zarr_file, rm_all_files):
    """Provides a Zarr file with some content for testing."""
    container = MemGroup()
    container.create_dataset("dset1", data=datasets[0].view())
    container.create_dataset("dset2", data=datasets[1].view())
    container.to_file(zarr_file, file_format=fileformats.Zarr)
    yield zarr_file, datasets
    rm_all_files(zarr_file)


@pytest.fixture
def distributed_h5_file_select(datasets, h5_file_distributed, rm_all_files):
    """Prepare a file for the select_parallel tests."""
    if comm.rank == 0:
        m1 = mpiarray.MPIArray.wrap(datasets[0], axis=0, comm=MPI.COMM_SELF)
        m2 = mpiarray.MPIArray.wrap(datasets[1], axis=0, comm=MPI.COMM_SELF)
        container = MemGroup(distributed=True, comm=MPI.COMM_SELF)
        container.create_dataset("dset1", data=m1, distributed=True)
        container.create_dataset("dset2", data=m2, distributed=True)
        container.to_file(h5_file_distributed, file_format=fileformats.HDF5)

    comm.Barrier()

    yield h5_file_distributed, datasets

    comm.Barrier()

    if comm.rank == 0:
        rm_all_files(h5_file_distributed)


@pytest.fixture
def distributed_zarr_file_select(datasets, zarr_file_distributed, rm_all_files):
    """Prepare a file for the select_parallel tests."""
    m1 = mpiarray.MPIArray.wrap(datasets[0], axis=0, comm=MPI.COMM_SELF)
    m2 = mpiarray.MPIArray.wrap(datasets[1], axis=0, comm=MPI.COMM_SELF)
    container = MemGroup(distributed=True, comm=MPI.COMM_SELF)
    container.create_dataset("dset1", data=m1, distributed=True)
    container.create_dataset("dset2", data=m2, distributed=True)
    container.to_file(zarr_file_distributed, file_format=fileformats.Zarr)

    comm.Barrier()

    yield zarr_file_distributed, datasets

    comm.Barrier()

    if comm.rank == 0:
        rm_all_files(zarr_file_distributed)


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "container_on_disk, file_format",
    [
        (lf("standard_h5_file_select"), fileformats.HDF5),
        (lf("standard_zarr_file_select"), fileformats.Zarr),
    ],
)
def test_file_select(container_on_disk, file_format):
    """Tests that makes hdf5 objects and tests selecting on their axes."""
    m = MemGroup.from_file(
        container_on_disk[0], selections=sel, file_format=file_format
    )
    assert np.all(m["dset1"][:] == container_on_disk[1][0][(fsel, isel, slice(None))])
    assert np.all(m["dset2"][:] == container_on_disk[1][1][(fsel, slice(None))])


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "container_on_disk, file_format",
    [
        (lf("standard_h5_file_select"), fileformats.HDF5),
        pytest.param(
            lf("standard_zarr_file_select"),
            fileformats.Zarr,
            marks=pytest.mark.xfail(reason="Zarr doesn't support index selections."),
        ),
    ],
)
def test_file_select_index(container_on_disk, file_format):
    """Tests that makes hdf5 objects and tests selecting on their axes."""

    # now test index selection
    m = MemGroup.from_file(
        container_on_disk[0], selections=index_sel, file_format=file_format
    )
    assert np.all(m["dset1"][:] == container_on_disk[1][0][index_sel["dset1"]])
    assert np.all(m["dset2"][:] == container_on_disk[1][1][index_sel["dset2"]])


## Parallel container tests


@pytest.fixture
def xfail_zarr_listsel(request):
    file_format = request.getfixturevalue("file_format")
    ind = request.getfixturevalue("ind")

    if file_format == fileformats.Zarr and isinstance(ind, (list, tuple)):
        request.node.add_marker(
            pytest.mark.xfail(reason="Zarr doesn't support list based indexing.")
        )


@pytest.mark.parametrize(
    "container_on_disk, file_format",
    [
        (lf("distributed_h5_file_select"), fileformats.HDF5),
        # (lf("distributed_zarr_file_select"), fileformats.Zarr),
    ],
)
@pytest.mark.parametrize("fsel", [slice(1, 8, 2), slice(5, 8, 2)])
@pytest.mark.parametrize("isel", [slice(1, 4), slice(5, 8, 2)])
@pytest.mark.parametrize("ind", [slice(None), [0, 2, 7]])
@pytest.mark.usefixtures("xfail_zarr_listsel")
def test_file_select_distributed(container_on_disk, fsel, isel, file_format, ind):
    """Load H5/Zarr file into parallel container while down-selecting axes."""

    if ind == slice(None):
        sel = {"dset1": (fsel, isel, slice(None)), "dset2": (fsel, slice(None))}
    else:
        sel = {"dset1": (fsel, ind, slice(None)), "dset2": (ind, slice(None))}

    # Tests are designed to run for 1, 2 or 4 processes
    assert 4 % comm.size == 0

    m = MemGroup.from_file(
        container_on_disk[0],
        selections=sel,
        distributed=True,
        comm=comm,
        file_format=file_format,
    )

    d1 = container_on_disk[1][0][sel["dset1"]]
    d2 = container_on_disk[1][1][sel["dset2"]]

    _, s, e = mpitools.split_local(d1.shape[0], comm=comm)
    d1slice = slice(s, e)

    _, s, e = mpitools.split_local(d2.shape[0], comm=comm)
    d2slice = slice(s, e)

    # For debugging...
    # Need to dereference datasets as this is collective
    # md1 = m["dset1"][:]
    # md2 = m["dset2"][:]
    # for ri in range(comm.size):
    #     if ri == comm.rank:
    #         print(comm.rank)
    #         print(md1.shape, d1.shape, d1[d1slice].shape)
    #         print(md1[0, :2, :2] if md1.size else "Empty")
    #         print(d1[d1slice][0, :2, :2] if d1[d1slice].size else "Empty")
    #         print()
    #         print(md2.shape, d2.shape, d2[d2slice].shape)
    #         print(md2[0, :2] if md2.size else "Empty")
    #         print(d2[d2slice][0, :2] if d2[d2slice].size else "Empty")
    #     comm.Barrier()

    assert np.all(m["dset1"][:].local_array == d1[d1slice])
    assert np.all(m["dset2"][:].local_array == d2[d2slice])
