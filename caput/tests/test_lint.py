import tempfile
import pytest
import yaml

from click.testing import CliRunner

from caput.config import Property
from caput.pipeline import TaskBase
from caput.scripts import runner as caput_script


class DoNothing(TaskBase):
    pass


class DoNothing2(DoNothing):
    a_list = Property(proptype=list)


@pytest.fixture()
def simple_config():
    yield {
        "pipeline": {
            "tasks": [
                {
                    "type": "caput.tests.test_lint.DoNothing",
                    "out": "out1",
                },
                {
                    "type": "caput.tests.test_lint.DoNothing",
                    "out": "out2",
                    "in": "out1",
                },
                {
                    "type": "caput.tests.test_lint.DoNothing2",
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
    temp = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
    yaml.safe_dump(config_json, temp, encoding="utf-8")
    temp.flush()
    return temp.name


def test_load_yaml(simple_config):
    test_runner = CliRunner()
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script.lint_config, [config_file])
    assert result.exit_code == 0


def test_unknown_task(simple_config):
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][0]["type"] = "what.was.the.name.of.my.Task"
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script.lint_config, [config_file])
    assert result.exit_code != 0


def test_wrong_type(simple_config):
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][2]["params"]["a_list"] = 1
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script.lint_config, [config_file])
    assert result.exit_code != 0


def test_lonely_in(simple_config):
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][2]["in"] = "foo"
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script.lint_config, [config_file])
    assert result.exit_code != 0


def test_lonely_requires(simple_config):
    test_runner = CliRunner()
    simple_config["pipeline"]["tasks"][2]["requires"] = "bar"
    config_file = write_to_file(simple_config)
    result = test_runner.invoke(caput_script.lint_config, [config_file])
    assert result.exit_code != 0
