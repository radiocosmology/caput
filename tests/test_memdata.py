"""Unit tests for the memdata module."""

import datetime
import gc
import json
from pathlib import Path
import warnings
from zipfile import ZipFile

import h5py
import numpy as np
import pytest
from pytest_lazy_fixtures import lf
import zarr
import copy

from caput import memdata


def test_ro_dict():
    """Test memdata.ro_dict."""
    a = {"a": 5}
    a = memdata.ro_dict(a)
    assert a["a"] == 5
    assert list(a.keys()) == ["a"]
    # Convoluded test to make sure you can't write to it.
    with pytest.raises(TypeError):
        # pylint: disable=unsupported-assignment-operation
        a["b"] = 6


# Unit test for MemDataset


def test_dataset_copy():
    # Check for string types
    x = memdata.MemDatasetCommon(shape=(4, 5), dtype=np.float32)
    x[:] = 0

    # Check a deepcopy using .copy
    y = x.copy()
    assert x == y
    y[:] = 1
    # Check this this is in fact a deep copy
    assert x != y

    # This is a shallow copy
    y = x.copy(shallow=True)
    assert x == y
    y[:] = 1
    assert x == y

    # Check a deepcopy using copy.deepcopy
    y = copy.deepcopy(x)
    assert x == y
    y[:] = 2
    assert x != y


# Unit tests for MemGroup.


def test_memgroup_nested():
    """Test nested groups in MemGroup."""
    root = memdata.MemGroup()
    l1 = root.create_group("level1")
    l2 = l1.require_group("level2")
    assert root["level1"] == l1
    assert root["level1/level2"] == l2
    assert root["level1/level2"].name == "/level1/level2"


def test_memgroup_create_dataset():
    """Test creating datasets in MemGroup."""
    g = memdata.MemGroup()
    data = np.arange(100, dtype=np.float32)
    g.create_dataset("data", data=data)
    assert np.allclose(data, g["data"])


def test_memgroup_recursive_create():
    """Test creating nested groups at once in MemGroup."""
    g = memdata.MemGroup()
    with pytest.raises(ValueError):
        g.create_group("")
    g2 = g.create_group("level2/")
    with pytest.raises(ValueError):
        g2.create_group("/")
    g2.create_group("/level22")
    assert set(g.keys()) == {"level22", "level2"}
    g.create_group("/a/b/c/d/")
    gd = g["/a/b/c/d/"]
    assert gd.name == "/a/b/c/d"


def test_memgroup_recursive_create_dataset():
    """Test creating nested datasets in MemGroup."""
    g = memdata.MemGroup()
    data = np.arange(10)
    g.create_dataset("a/ra", data=data)
    assert memdata.is_group(g["a"])
    assert np.all(g["a/ra"][:] == data)
    g["a"].create_dataset("/ra", data=data)
    assert np.all(g["ra"][:] == data)
    assert isinstance(g["a/ra"].parent, memdata.MemGroup)

    # Check that d keeps g in scope.
    d = g["a/ra"]
    del g
    gc.collect()
    assert np.all(d.file["ra"][:] == data)


def fill_test_file(f):
    """Fill a file with some groups, datasets and attrs for testing."""
    l1 = f.create_group("level1")
    l1.create_group("level2")
    d1 = l1.create_dataset("large", data=np.arange(100))
    f.attrs["a"] = 5
    d1.attrs["b"] = 6


@pytest.fixture
def filled_h5_file(h5_file):
    """Provides an H5 file with some content."""
    with h5py.File(h5_file, "w") as f:
        fill_test_file(f)
        f["level1"]["level2"].attrs["small"] = np.arange(3)
        f["level1"]["level2"].attrs["ndarray"] = np.ndarray([1, 2, 3])
    yield h5_file


@pytest.fixture
def filled_zarr_file(zarr_file):
    """Provides a .zarr file with some content."""
    with zarr.open_group(zarr_file, "w") as f:
        fill_test_file(f)
        f["level1"]["level2"].attrs["small"] = [0, 1, 2]
    yield zarr_file


@pytest.fixture
def filled_zarrzip_file(filled_zarr_file):
    """Provides a .zarr.zip file with some content."""

    zarrzip_file = filled_zarr_file + ".zip"

    zp = Path(filled_zarr_file)
    with ZipFile(zarrzip_file, "w", compresslevel=0) as zfh:
        for f in zp.rglob("*"):
            arcname = str(f.relative_to(zp))
            zfh.write(f, arcname=arcname)

    yield zarrzip_file


def assertGroupsEqual(a, b):
    """Compare two groups."""
    assert list(a.keys()) == list(b.keys())
    assertAttrsEqual(a.attrs, b.attrs)
    for key in a.keys():
        this_a = a[key]
        this_b = b[key]
        if not memdata.is_group(a[key]):
            assertAttrsEqual(this_a.attrs, this_b.attrs)
            assert np.allclose(this_a, this_b)
        else:
            assertGroupsEqual(this_a, this_b)


def assertAttrsEqual(a, b):
    """Compare two attributes."""
    assert list(a.keys()) == list(b.keys())
    for key in a.keys():
        this_a = a[key]
        this_b = b[key]
        if hasattr(this_a, "shape"):
            assert np.allclose(this_a, this_b)
        else:
            assert this_a == this_b


@pytest.mark.parametrize(
    "test_file,file_open_function",
    [
        (lf("filled_h5_file"), h5py.File),
        (lf("filled_zarr_file"), zarr.open_group),
    ],
)
def test_file_sanity(test_file, file_open_function):
    """Compare a file with itself."""
    with file_open_function(test_file, "r") as f:
        assertGroupsEqual(f, f)


@pytest.mark.parametrize(
    "test_file,file_open_function,file_format",
    [
        (lf("filled_h5_file"), h5py.File, None),
        (lf("filled_zarr_file"), zarr.open_group, None),
        (lf("filled_zarrzip_file"), zarr.open_group, None),
        (lf("filled_h5_file"), h5py.File, memdata.fileformats.HDF5),
        (lf("filled_zarr_file"), zarr.open_group, memdata.fileformats.Zarr),
        (lf("filled_zarrzip_file"), zarr.open_group, memdata.fileformats.Zarr),
    ],
)
def test_to_from_file(test_file, file_open_function, file_format):
    """Tests that makes hdf5 objects, convert to mem and back."""
    m = memdata.MemGroup.from_file(test_file, file_format=file_format)

    # Check that read in file has same structure
    with file_open_function(test_file, "r") as f:
        assertGroupsEqual(f, m)

    new_name = f"new.{test_file}"
    m.to_file(new_name, file_format=file_format)

    # Check that written file has same structure
    with file_open_function(new_name, "r") as f:
        assertGroupsEqual(f, m)


@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("filled_h5_file"), memdata.fileformats.HDF5),
        (lf("filled_zarr_file"), memdata.fileformats.Zarr),
    ],
)
def test_memdisk(test_file, file_format):
    """Test MemDiskGroup."""
    f = memdata.MemDiskGroup(test_file, file_format=file_format)
    assert set(f.keys()) == set(f._data.keys())
    m = memdata.MemDiskGroup(memdata.MemGroup.from_file(test_file, file_format=file_format))
    assert set(m.keys()) == set(f.keys())
    # Recursive indexing.
    assert set(f["/level1/"].keys()) == set(m["/level1/"].keys())
    assert set(f.keys()) == set(m["/level1"]["/"].keys())
    assert np.all(f["/level1/large"][:] == m["/level1/large"])
    gf = f.create_group("/level1/level2/level3/")
    gf.create_dataset("new", data=np.arange(5))
    gm = m.create_group("/level1/level2/level3/")
    gm.create_dataset("new", data=np.arange(5))
    assert np.all(
        f["/level1/level2/level3/new"][:] == m["/level1/level2/level3/new"][:]
    )


@pytest.mark.parametrize(
    "compression,compression_opts,chunks",
    [(None, None, None), ("bitshuffle", (None, "lz4"), (2, 3))],
)
@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("filled_h5_file"), memdata.fileformats.HDF5),
        (lf("filled_zarr_file"), memdata.fileformats.Zarr),
    ],
)
def test_compression(test_file, file_format, compression, compression_opts, chunks):
    # add a new compressed dataset
    f = memdata.MemDiskGroup.from_file(test_file, file_format=file_format)
    rng = np.random.default_rng(12345)
    f.create_dataset(
        "new",
        data=rng.random((5, 7)),
        chunks=chunks,
        compression=compression,
        compression_opts=compression_opts,
    )
    # f.flush()
    f.save(
        test_file + ".cmp",
        convert_attribute_strings=True,
        convert_dataset_strings=True,
        file_format=file_format,
    )
    # f.close()

    # read back compression parameters from file
    with file_format.open(test_file + ".cmp") as fh:
        if file_format is memdata.fileformats.HDF5:
            if compression is not None:
                # for some reason .compression doesn't get set...
                assert str(memdata.fileformats.H5FILTER) in fh["new"]._filters
            assert fh["new"].chunks == chunks
        else:
            if compression is None:
                assert fh["new"].compressor is None
                assert fh["new"].chunks == fh["new"].shape
            else:
                assert fh["new"].compressor is not None
                assert fh["new"].chunks == chunks


class TempSubClass(memdata.MemDiskGroup):
    """A subclass of MemDiskGroup for testing."""

    pass


@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_file"), memdata.fileformats.HDF5),
        (lf("zarr_file"), memdata.fileformats.Zarr),
        (lf("h5_file"), None),
        (lf("zarr_file"), None),
    ],
)
def test_io(test_file, file_format):
    """Test I/O of MemDiskGroup."""
    # Save a subclass of MemDiskGroup
    tsc = TempSubClass()
    tsc.create_dataset("dset", data=np.arange(10))
    tsc.save(test_file, file_format=file_format)

    actual_file_format = memdata.fileformats.guess_file_format(test_file)

    # Load it from disk
    tsc2 = memdata.MemDiskGroup.from_file(test_file, file_format=file_format)
    tsc3 = memdata.MemDiskGroup.from_file(test_file, ondisk=True, file_format=file_format)

    # Check that is is recreated with the correct type
    assert isinstance(tsc2, TempSubClass)
    assert isinstance(tsc3, TempSubClass)

    # Check that parent/etc is properly implemented.
    # Turns out this is very hard so give up for now.
    # self.assertIsInstance(tsc2['dset'].parent, TempSubClass)
    # self.assertIsInstance(tsc3['dset'].parent, TempSubClass)
    tsc3.close()

    with memdata.MemDiskGroup.from_file(
        test_file, mode="r", ondisk=True, file_format=file_format
    ):
        # h5py will error if file already open
        if actual_file_format == memdata.fileformats.HDF5:
            with pytest.raises(IOError):
                actual_file_format.open(test_file, "w")
        # ...zarr will not
        else:
            actual_file_format.open(test_file, "w")

    with memdata.MemDiskGroup.from_file(
        test_file, mode="r", ondisk=False, file_format=file_format
    ):
        f = actual_file_format.open(test_file, "w")
        if actual_file_format == memdata.fileformats.HDF5:
            f.close()




@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_file"), memdata.fileformats.HDF5),
        (lf("zarr_file"), memdata.fileformats.Zarr),
    ],
)
def test_to_from__file_unicode(test_file, file_format):
    """Test that a unicode memdata dataset is round tripped correctly."""
    udata = np.array(["Test", "this", "works"])
    sdata = udata.astype("S")
    assert udata.dtype.kind == "U"
    assert sdata.dtype.kind == "S"

    m = memdata.MemGroup()
    udset = m.create_dataset("udata", data=udata)
    sdset = m.create_dataset("sdata", data=sdata)
    assert udset.dtype.kind == "U"
    assert sdset.dtype.kind == "S"

    # Test a write without conversion. This should throw an exception
    with pytest.raises(TypeError):
        m.to_file(test_file, file_format=file_format)

    # Write with conversion
    m.to_file(
        test_file,
        convert_attribute_strings=True,
        convert_dataset_strings=True,
        file_format=file_format,
    )

    with file_format.open(test_file, "r") as fh:
        # pylint warns here that "Instance of 'Group' has no 'dtype' member"
        # pylint: disable=E1101
        assert fh["udata"].dtype.kind == "S"
        assert fh["sdata"].dtype.kind == "S"

    # Test a load without conversion, types should be bytestrings
    m2 = memdata.MemGroup.from_file(test_file, file_format=file_format)
    assert m2["udata"].dtype.kind == "S"
    assert m2["sdata"].dtype.kind == "S"
    # Check the dtype here, for some reason Python 2 thinks the arrays are equal
    # and Python 3 does not even though both agree that the datatypes are different
    assert m["udata"].dtype != m2["udata"].dtype
    assert (m["sdata"].data == m2["sdata"].data).all()

    # Test a load *with* conversion, types should be unicode
    m3 = memdata.MemGroup.from_file(
        test_file,
        convert_attribute_strings=True,
        convert_dataset_strings=True,
        file_format=file_format,
    )
    assert m3["udata"].dtype.kind == "U"
    assert m3["sdata"].dtype.kind == "U"
    assert (m["udata"].data == m3["udata"].data).all()
    assert (m["udata"].data == m3["sdata"].data).all()


@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_file"), memdata.fileformats.HDF5),
        (lf("zarr_file"), memdata.fileformats.Zarr),
    ],
)
def test_failure(test_file, file_format):
    """Test that we fail when trying to write a non ASCII character."""
    udata = np.array(["\u03b2"])

    m = memdata.MemGroup()
    m.create_dataset("udata", data=udata)

    with pytest.raises(TypeError):
        m.to_file(test_file, file_format=file_format)


@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_file"), memdata.fileformats.HDF5),
        (lf("zarr_file"), memdata.fileformats.Zarr),
    ],
)
def test_to_from_hdf5(test_file, file_format):
    """Test that a memdata dataset JSON serialization is done correctly."""
    json_prefix = "!!_memh5_json:"
    data = {"foo": {"bar": [1, 2, 3], "fu": "1"}}
    time = datetime.datetime.now()

    m = memdata.MemGroup()
    m.attrs["data"] = data
    m.attrs["datetime"] = {"datetime": time}
    m.attrs["ndarray"] = np.ndarray([1, 2, 3])

    m.to_file(test_file, file_format=file_format)
    with file_format.open(test_file, "r") as f:
        assert f.attrs["data"] == json_prefix + json.dumps(data)
        assert f.attrs["datetime"] == json_prefix + json.dumps(
            {"datetime": time.isoformat()}
        )

    m2 = memdata.MemGroup.from_file(test_file, file_format=file_format)
    assert m2.attrs["data"] == data
    assert m2.attrs["datetime"] == {"datetime": time.isoformat()}


@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_file"), memdata.fileformats.HDF5),
        (lf("zarr_file"), memdata.fileformats.Zarr),
    ],
)
def test_json_failure(test_file, file_format):
    """Test that we get a TypeError if we try to serialize something else."""
    m = memdata.MemGroup()
    m.attrs["non_serializable"] = {"datetime": object}

    with pytest.raises(TypeError):
        m.to_file(test_file, file_format=file_format)
