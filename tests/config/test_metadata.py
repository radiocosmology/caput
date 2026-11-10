import unittest

import numpy
import yaml

import caput
from caput.pipeline import Manager


class TestMetadata(unittest.TestCase):
    """Test metadata defaults and configuration overrides."""

    def test_default_params(self):
        """Check default params behavior."""
        testconfig = """
        pipeline:
            bla: "foo"
        """

        man = Manager.from_yaml_str(testconfig)

        self.assertIn("versions", man.all_tasks_params)
        self.assertIn("pipeline_config", man.all_tasks_params)

        self.assertDictEqual(man.all_tasks_params["versions"], {})
        # remove line numbers
        pipeline_config = man.all_tasks_params["pipeline_config"]
        self.assertDictEqual(
            pipeline_config,
            yaml.load(testconfig, Loader=yaml.SafeLoader),
        )

    def test_metadata_params(self):
        """Check metadata params behavior."""
        testconfig = """
        foo: bar
        pipeline:
            save_versions:
                - numpy
                - caput
            bla: "foo"
        """

        man = Manager.from_yaml_str(testconfig)

        self.assertIn("versions", man.all_tasks_params)
        self.assertIn("pipeline_config", man.all_tasks_params)

        self.assertDictEqual(
            man.all_tasks_params["versions"],
            {"numpy": numpy.__version__, "caput": caput.__version__},
        )

        # remove line numbers
        pipeline_config = man.all_tasks_params["pipeline_config"]
        self.assertDictEqual(
            pipeline_config,
            yaml.load(testconfig, Loader=yaml.SafeLoader),
        )

    def test_metadata_params_no_config(self):
        """Check metadata params no config behavior."""
        testconfig = """
        pipeline:
            save_versions: numpy
            save_config: False
        """

        man = Manager.from_yaml_str(testconfig)

        self.assertIn("versions", man.all_tasks_params)
        self.assertIn("pipeline_config", man.all_tasks_params)

        self.assertDictEqual(
            man.all_tasks_params["versions"], {"numpy": numpy.__version__}
        )
        self.assertIsNone(man.all_tasks_params["pipeline_config"], {})
