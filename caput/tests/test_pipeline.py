"""Test running the caput.pipeline."""
import tempfile

from caput.scripts.runner import cli

eggs_conf = """
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


def test_pipeline():
    """Test running a very simple pipeline."""
    with tempfile.NamedTemporaryFile("w+") as configfile:
        configfile.write(eggs_conf)
        configfile.flush()
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(cli, ["run", configfile.name])
        print(result.output)
        assert result.exit_code == 0
