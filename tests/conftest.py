"""Pytest fixtures and simple tasks that can be used by all unit tests."""

import glob
import tempfile

import numpy as np
import pytest

from caput.pipeline import PipelineStopIteration, TaskBase, IterBase
from caput.scripts.runner import cli
from caput import config, fileformats, mpiutil


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
        """Run setup."""
        print("Setting up PrintEggs.")

    def next(self, _input=None):
        """Run next."""
        if self.i >= len(self.eggs):
            raise PipelineStopIteration()
        print("Spam and %s eggs." % self.eggs[self.i])
        self.i += 1

    def finish(self):
        """Run finish."""
        print("Finished PrintEggs.")


class GetEggs(TaskBase):
    """Simple task used for testing."""

    eggs = config.Property(proptype=list)

    def __init__(self, *args, **kwargs):
        self.i = 0
        super().__init__(*args, **kwargs)

    def setup(self, requires=None):
        """Run setup."""
        print("Setting up GetEggs.")

    def next(self, _input=None):
        """Run next."""
        if self.i >= len(self.eggs):
            raise PipelineStopIteration()
        egg = self.eggs[self.i]
        self.i += 1
        return egg

    def finish(self):
        """Run finish."""
        print("Finished GetEggs.")


class CookEggs(IterBase):
    """Simple task used for testing."""

    style = config.Property(proptype=str)

    def setup(self, requires=None):
        """Run setup."""
        print("Setting up CookEggs.")

    def process(self, _input):
        """Run process."""
        print("Cooking %s %s eggs." % (self.style, _input))

    def finish(self):
        """Run finish."""
        print("Finished CookEggs.")

    def read_input(self, filename):
        """Run read input not implemented."""
        raise NotImplementedError()

    def read_output(self, filename):
        """Run read output not implemented."""
        raise NotImplementedError()

    def write_output(self, filename, output, file_format=None, **kwargs):
        """Run write output not implemented."""
        raise NotImplementedError()


@pytest.fixture
def run_pipeline():
    """Provides the `run_pipeline` function which will run the pipeline."""
    eggs_pipeline_conf = """
---
pipeline:
  tasks:
    - type: tests.conftest.PrintEggs
      params: eggs_params
    - type: tests.conftest.GetEggs
      params: eggs_params
      out: egg
    - type: tests.conftest.CookEggs
      params: cook_params
      in: egg
eggs_params:
  eggs: ['green', 'duck', 'ostrich']
cook_params:
  style: 'fried'
"""

    def _run_pipeline(parameters=None, configstr=eggs_pipeline_conf):
        """Run `caput.scripts.runner run` with given parameters and config.

        Parameters
        ----------
        parameters : List[str]
            Parameters to pass to the cli, for example `["--profile"]` (see `--help`).
        configstr : str
            YAML string to use as a config. This function will write it to a file that is then passed to the cli.

        Returns
        -------
        result : `click.testing.Result`
            Holds the captured result. Try accessing e.g. `result.exit_code`, `result.output`.
        """
        with tempfile.NamedTemporaryFile("w+") as configfile:
            configfile.write(configstr)
            configfile.flush()
            from click.testing import CliRunner

            runner = CliRunner()
            if parameters is None:
                return runner.invoke(cli, ["run", configfile.name])
            else:
                return runner.invoke(cli, ["run", *parameters, configfile.name])

    return _run_pipeline


@pytest.fixture
def h5_file(rm_all_files):
    """Provides a file name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5.h5"
    yield fname
    rm_all_files(fname)


@pytest.fixture
def zarr_file(rm_all_files):
    """Provides a directory name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5.zarr"
    yield fname
    rm_all_files(fname)


@pytest.fixture
def h5_file_distributed(rm_all_files):
    """Provides a file name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5_distributed.h5"
    yield fname
    if mpiutil.rank == 0:
        rm_all_files(fname)


@pytest.fixture
def zarr_file_distributed(rm_all_files):
    """Provides a directory name and removes all files/dirs with the same prefix later."""
    fname = "tmp_test_memh5.zarr"
    yield fname
    if mpiutil.rank == 0:
        rm_all_files(fname)


@pytest.fixture
def rm_all_files():
    """Provides the `rm_all_files` function."""

    def _rm_all_files(file_name):
        """Remove all files and directories starting with `file_name`."""
        file_names = glob.glob(file_name + "*")
        for fname in file_names:
            fileformats.remove_file_or_dir(fname)

    return _rm_all_files
