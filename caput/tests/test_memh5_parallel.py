"""Unit tests for the parallel features of the memh5 module."""

import pytest
from pytest_lazyfixture import lazy_fixture
import numpy as np
import h5py
import zarr

from caput import fileformats, memh5, mpiarray, mpiutil


comm = mpiutil.world
rank, size = mpiutil.rank, mpiutil.size


def test_create_dataset():
    """Test for creating datasets in MemGroup."""
    global_data = np.arange(size * 5 * 10, dtype=np.float32)
    local_data = global_data.reshape(size, -1, 10)[rank]
    d_array = mpiarray.MPIArray.wrap(local_data, axis=0)
    d_array_T = d_array.redistribute(axis=1)

    # Check that we must specify in advance if the dataset is distributed
    g = memh5.MemGroup()
    if comm is not None:
        with pytest.raises(RuntimeError):
            g.create_dataset("data", data=d_array)

    g = memh5.MemGroup(distributed=True)

    # Create an array from data
    g.create_dataset("data", data=d_array, distributed=True)

    # Create an array from data with a different distribution
    g.create_dataset("data_T", data=d_array, distributed=True, distributed_axis=1)

    # Create an empty array with a specified shape
    g.create_dataset(
        "data2",
        shape=(size * 5, 10),
        dtype=np.float64,
        distributed=True,
        distributed_axis=1,
    )
    assert np.allclose(d_array, g["data"][:])
    assert np.allclose(d_array_T, g["data_T"][:])
    if comm is not None:
        assert d_array_T.local_shape == g["data2"].local_shape

    # Test global indexing
    assert (g["data"][rank * 5] == local_data[0]).all()


@pytest.mark.parametrize(
    "compression,compression_opts,chunks",
    [
        (None, None, None),
        ("bitshuffle", (None, "lz4"), (size // 2 + ((size // 2) == 0), 3)),
    ],
)
@pytest.mark.parametrize(
    "test_file,file_open_function,file_format",
    [
        (lazy_fixture("h5_file_distributed"), h5py.File, fileformats.HDF5),
        (
            lazy_fixture("zarr_file_distributed"),
            zarr.open_group,
            fileformats.Zarr,
        ),
    ],
)
def test_io(
    test_file, file_open_function, file_format, compression, compression_opts, chunks
):
    """Test for I/O in MemGroup."""

    # Create distributed memh5 object
    g = memh5.MemGroup(distributed=True)
    g.attrs["rank"] = rank

    # Create an empty array with a specified shape
    pdset = g.create_dataset(
        "parallel_data",
        shape=(size, 10),
        dtype=np.float64,
        distributed=True,
        distributed_axis=0,
        compression=compression,
        compression_opts=compression_opts,
        chunks=chunks,
    )
    pdset[:] = rank
    pdset.attrs["const"] = 17

    # Create an empty array with a specified shape
    sdset = g.create_dataset("serial_data", shape=(size * 5, 10), dtype=np.float64)
    sdset[:] = rank
    sdset.attrs["const"] = 18

    # Create nested groups
    g.create_group("hello/world")

    # Test round tripping unicode data
    g.create_dataset("unicode_data", data=np.array(["hello"]))

    g.to_file(
        test_file,
        convert_attribute_strings=True,
        convert_dataset_strings=True,
        file_format=file_format,
    )

    # Test that the HDF5 file has the correct structure
    with file_open_function(test_file, "r") as f:

        # Test that the file attributes are correct
        assert f["parallel_data"].attrs["const"] == 17

        # Test that the parallel dataset has been written correctly
        assert (f["parallel_data"][:, 0] == np.arange(size)).all()
        assert f["parallel_data"].attrs["const"] == 17

        # Test that the common dataset has been written correctly (i.e. by rank=0)
        assert (f["serial_data"][:] == 0).all()
        assert f["serial_data"].attrs["const"] == 18

        # Check group structure is correct
        assert "hello" in f
        assert "world" in f["hello"]

        # Check compression/chunks
        if file_format is fileformats.Zarr:
            if chunks is None:
                assert f["parallel_data"].chunks == f["parallel_data"].shape
                assert f["parallel_data"].compressor is None
            else:
                assert f["parallel_data"].chunks == chunks
                assert f["parallel_data"].compressor is not None
        elif file_format is fileformats.HDF5:
            # compression should be disabled
            # (for some reason .compression is not set...)
            assert str(fileformats.H5FILTER) not in f["parallel_data"]._filters
            assert f["parallel_data"].chunks is None

    # Test that the read in group has the same structure as the original
    g2 = memh5.MemGroup.from_file(
        test_file,
        distributed=True,
        convert_attribute_strings=True,
        convert_dataset_strings=True,
        file_format=file_format,
    )

    # Check that the parallel data is still the same
    assert (g2["parallel_data"][:] == g["parallel_data"][:]).all()

    # Check that the serial data is all zeros (should not be the same as before)
    assert (g2["serial_data"][:] == np.zeros_like(sdset[:])).all()

    # Check group structure is correct
    assert "hello" in g2
    assert "world" in g2["hello"]

    # Check the unicode dataset
    assert g2["unicode_data"].dtype.kind == "U"
    assert g2["unicode_data"][0] == "hello"

    # Check the attributes
    assert g2["parallel_data"].attrs["const"] == 17
    assert g2["serial_data"].attrs["const"] == 18


@pytest.mark.parametrize(
    "test_file,file_open_function,file_format",
    [
        (lazy_fixture("h5_file_distributed"), h5py.File, fileformats.HDF5),
        (
            lazy_fixture("zarr_file_distributed"),
            zarr.open_group,
            fileformats.Zarr,
        ),
    ],
)
def test_misc(test_file, file_open_function, file_format):
    """Misc tests for MemDiskGroupDistributed"""

    dg = memh5.MemDiskGroup(distributed=True)

    pdset = dg.create_dataset(
        "parallel_data",
        shape=(10,),
        dtype=np.float64,
        distributed=True,
        distributed_axis=0,
    )
    # pdset[:] = dg._data.comm.rank
    pdset[:] = rank
    # Test successfully added
    assert "parallel_data" in dg

    dg.save(test_file, file_format=file_format)

    dg2 = memh5.MemDiskGroup.from_file(
        test_file, distributed=True, file_format=file_format
    )

    # Test successful load
    assert "parallel_data" in dg2
    assert (dg["parallel_data"][:] == dg2["parallel_data"][:]).all()

    # self.assertRaises(NotImplementedError, dg.to_disk, self.fname)

    # Test refusal to base off a h5py object when distributed
    with file_open_function(test_file, "r") as f:
        if comm is not None:
            with pytest.raises(ValueError):
                # MemDiskGroup will guess the file format
                memh5.MemDiskGroup(data_group=f, distributed=True)
    mpiutil.barrier()


def test_redistribute():
    """Test redistribute in BasicCont."""

    g = memh5.BasicCont(distributed=True)

    # Create an array from data
    g.create_dataset("data", shape=(10, 10), distributed=True, distributed_axis=0)
    assert g["data"].distributed_axis == 0
    g.redistribute(1)
    assert g["data"].distributed_axis == 1
