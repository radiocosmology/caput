"""Pytest fixtures and simple tasks that can be used by all unit tests."""

import tempfile

import numpy as np
import pytest

from caput.pipeline import PipelineStopIteration, TaskBase, IterBase
from caput.scripts.runner import cli
from caput import config


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
        print("Setting up PrintEggs.")

    def next(self, _input=None):
        if self.i >= len(self.eggs):
            raise PipelineStopIteration()
        print("Spam and %s eggs." % self.eggs[self.i])
        self.i += 1

    def finish(self):
        print("Finished PrintEggs.")


class GetEggs(TaskBase):
    """Simple task used for testing."""

    eggs = config.Property(proptype=list)

    def __init__(self, *args, **kwargs):
        self.i = 0
        super().__init__(*args, **kwargs)

    def setup(self, requires=None):
        print("Setting up GetEggs.")

    def next(self, _input=None):
        if self.i >= len(self.eggs):
            raise PipelineStopIteration()
        egg = self.eggs[self.i]
        self.i += 1
        return egg

    def finish(self):
        print("Finished GetEggs.")


class CookEggs(IterBase):
    """Simple task used for testing."""

    style = config.Property(proptype=str)

    def setup(self, requires=None):
        print("Setting up CookEggs.")

    def process(self, _input):
        print("Cooking %s %s eggs." % (self.style, input))

    def finish(self):
        print("Finished CookEggs.")

    def read_input(self, filename):
        raise NotImplementedError()

    def read_output(self, filename):
        raise NotImplementedError()

    def write_output(self, filename, output):
        raise NotImplementedError()


eggs_pipeline_conf = """
---
pipeline:
  tasks:
    - type: caput.tests.conftest.PrintEggs
      params: eggs_params
    - type: caput.tests.conftest.GetEggs
      params: eggs_params
      out: egg
    - type: caput.tests.conftest.CookEggs
      params: cook_params
      in: egg
eggs_params:
  eggs: ['green', 'duck', 'ostrich']
cook_params:
  style: 'fried'
"""


def run_pipeline(parameters=None, configstr=eggs_pipeline_conf):
    """
    Run `caput.scripts.runner run` with given parameters and config.

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
