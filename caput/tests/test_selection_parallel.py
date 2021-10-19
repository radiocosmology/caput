"""Parallel version of the selection tests.

Needs to be run on 1, 2 or 4 MPI processes.
"""
import glob
import os

from mpi4py import MPI
import numpy as np
import pytest

from caput import mpiutil, mpiarray
from caput.memh5 import MemGroup


comm = MPI.COMM_WORLD


@pytest.fixture(scope="module")
def container_on_disk(datasets):

    fname = "tmp_test_memh5_select_parallel.h5"

    if comm.rank == 0:
        m1 = mpiarray.MPIArray.wrap(datasets[0], axis=0, comm=MPI.COMM_SELF)
        m2 = mpiarray.MPIArray.wrap(datasets[1], axis=0, comm=MPI.COMM_SELF)
        container = MemGroup(distributed=True, comm=MPI.COMM_SELF)
        container.create_dataset("dset1", data=m1, distributed=True)
        container.create_dataset("dset2", data=m2, distributed=True)
        container.to_hdf5(fname)

    comm.Barrier()

    yield fname, datasets

    comm.Barrier()

    # tear down

    if comm.rank == 0:
        file_names = glob.glob(fname + "*")
        for fname in file_names:
            os.remove(fname)


@pytest.mark.parametrize("fsel", [slice(1, 8, 2), slice(5, 8, 2)])
@pytest.mark.parametrize("isel", [slice(1, 4), slice(5, 8, 2)])
@pytest.mark.parametrize("ind", [slice(None), [0, 2, 7]])
def test_H5FileSelect_distributed(container_on_disk, fsel, isel, ind):
    """Load H5 into parallel container while down-selecting axes."""

    if ind == slice(None):
        sel = {"dset1": (fsel, isel, slice(None)), "dset2": (fsel, slice(None))}
    else:
        sel = {"dset1": (fsel, ind, slice(None)), "dset2": (ind, slice(None))}

    # Tests are designed to run for 1, 2 or 4 processes
    assert 4 % comm.size == 0

    m = MemGroup.from_hdf5(
        container_on_disk[0], selections=sel, distributed=True, comm=comm
    )

    d1 = container_on_disk[1][0][sel["dset1"]]
    d2 = container_on_disk[1][1][sel["dset2"]]

    _, s, e = mpiutil.split_local(d1.shape[0], comm=comm)
    d1slice = slice(s, e)

    _, s, e = mpiutil.split_local(d2.shape[0], comm=comm)
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

    assert np.all(m["dset1"][:] == d1[d1slice])
    assert np.all(m["dset2"][:] == d2[d2slice])
