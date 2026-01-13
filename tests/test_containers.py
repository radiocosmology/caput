"""Unit tests for the containers module."""

import pytest
import json
import numpy as np
import warnings

from pathlib import Path
from pytest_lazy_fixtures import lf
from zipfile import ZipFile

from caput import containers, memdata


@pytest.fixture(name="history_dict")
def fixture_history_dict():
    """Provides dict with some content for testing."""
    return {"foo": {"bar": {"f": 23}, "foo": "bar"}, "bar": 0}


@pytest.fixture
def h5_basiccont_file(h5_file, history_dict):
    """Provides a Container file written to HDF5."""
    d = containers.Container()
    d.create_dataset("a", data=np.arange(5))
    d.add_history("test", history_dict)
    d.to_disk(h5_file)
    yield h5_file, history_dict


@pytest.fixture
def zarr_basiccont_file(zarr_file, history_dict):
    """Provides a Container file written to Zarr."""
    d = containers.Container()
    d.create_dataset("a", data=np.arange(5))
    d.add_history("test", history_dict)
    d.to_disk(zarr_file, file_format=memdata.fileformats.Zarr)
    yield zarr_file, history_dict


@pytest.fixture
def zarrzip_basiccont_file(zarr_basiccont_file):
    """Provides a Container file as .zarr.zip."""
    zarr_file, history_dict = zarr_basiccont_file

    zarrzip_file = zarr_file + ".zip"

    zp = Path(zarr_file)
    with ZipFile(zarrzip_file, "w", compresslevel=0) as zfh:
        for f in zp.rglob("*"):
            arcname = str(f.relative_to(zp))
            zfh.write(f, arcname=arcname)

    yield zarrzip_file, history_dict


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_basiccont_file"), memdata.fileformats.HDF5),
        (lf("zarr_basiccont_file"), memdata.fileformats.Zarr),
        (lf("zarrzip_basiccont_file"), memdata.fileformats.Zarr),
    ],
)
def test_access(test_file, file_format):
    """Test access to Container content."""
    test_file = test_file[0]
    d = containers.Container.from_file(test_file, file_format=file_format)
    assert "history" in d._data
    assert "index_map" in d._data
    with pytest.raises(KeyError):
        d.__getitem__("history")
    with pytest.raises(KeyError):
        d.__getitem__("index_map")

    with pytest.raises(ValueError):
        d.create_group("a")
    with pytest.raises(ValueError):
        d.create_dataset("index_map/stuff", data=np.arange(5))


@pytest.mark.mpi_skip
@pytest.mark.parametrize(
    "test_file,file_format",
    [
        (lf("h5_basiccont_file"), memdata.fileformats.HDF5),
        (lf("zarr_basiccont_file"), memdata.fileformats.Zarr),
    ],
)
def test_history(test_file, file_format):
    """Test history of Container."""
    basic_cont, history_dict = test_file
    json_prefix = "!!_memh5_json:"

    # Check file for config- and versiondump
    with file_format.open(basic_cont, "r") as f:
        history = f["history"].attrs["test"]
        # if file_format == memdata.fileformats.HDF5:
        assert history == json_prefix + json.dumps(history_dict)
        # else:
        #     assert history == history_dict

    # add old format history
    with file_format.open(basic_cont, "r+") as f:
        f["history"].create_group("old_history_format")
        f["history/old_history_format"].attrs["foo"] = "bar"

    with containers.Container.from_file(basic_cont, file_format=file_format) as m:
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            old_history_format = m.history["old_history_format"]

            # Expect exactly one warning about deprecated history format
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)
            assert "deprecated" in str(w[-1].message)

    assert old_history_format == {"foo": "bar"}


def test_redistribute():
    """Test redistribute in the base :py:class:`~caput.containers.Container`."""

    g = containers.Container(distributed=True)

    g.create_dataset("data", shape=(10, 10), distributed=True, distributed_axis=0)

    assert g["data"].distributed_axis == 0

    g.redistribute(1)

    assert g["data"].distributed_axis == 1
