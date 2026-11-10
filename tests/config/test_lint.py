"""Tests for pipeline configuration linting."""

import tempfile

import pytest
import yaml
from click.testing import CliRunner

from caput.config import Property
from caput.pipeline import Task
from caput.pipeline.runner._cli import cli as caput_script


class DoNothing(Task):
    """Task class used to build valid lint configuration."""

    pass


class DoNothing2(DoNothing):
    """Task with a typed property for configuration validation tests."""

    a_list = Property(proptype=list)


@pytest.fixture()
def simple_config():
    """Return a valid pipeline configuration with typed task parameters."""
    yield {
        "pipeline": {
            "tasks": [
                {
                    "type": "tests.config.test_lint.DoNothing",
                    "out": "out1",
                },
                {
                    "type": "tests.config.test_lint.DoNothing",
                    "out": "out2",
                    "in": "out1",
                },
                {
                    "type": "tests.config.test_lint.DoNothing2",
                    "in": "out2",
                    "out": "out3",
                    "requires": "out1",
                    "params": {
                        "a_list": [1],
                    },
                },
            ]
        }
    }


def write_to_file(config_json):
    """Serialize a pipeline configuration to a temporary YAML file."""
    with tempfile.NamedTemporaryFile(mode="w+t", delete=False) as configfile:
        yaml.safe_dump(config_json, configfile, encoding="utf-8")
        configfile.flush()
        return configfile.name


def test_load_yaml(simple_config):
    """Check load yaml behavior."""
    test_runner = CliRunner()
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script, ["lint", config_file])
    assert result.exit_code == 0


def test_unknown_task(simple_config):
    """Check unknown task behavior."""
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][0]["type"] = "what.was.the.name.of.my.Task"
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script, ["lint", config_file])
    assert result.exit_code != 0


def test_wrong_type(simple_config):
    """Check wrong type behavior."""
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][2]["params"]["a_list"] = 1
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script, ["lint", config_file])
    assert result.exit_code != 0


def test_lonely_in(simple_config):
    """Check lonely in behavior."""
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][2]["in"] = "foo"
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script, ["lint", config_file])
    assert result.exit_code != 0


def test_lonely_requires(simple_config):
    """Check lonely requires behavior."""
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][2]["requires"] = "bar"
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script, ["lint", config_file])
    assert result.exit_code != 0
