import unittest

import numpy
import caput
import yaml
from caput import pipeline


class TestConfig(unittest.TestCase):
    def test_default_params(self):
        testconfig = """
        pipeline:
            bla: "foo"
        """

        man = pipeline.Manager.from_yaml_str(testconfig)

        self.assertIn("versions", man.all_tasks_params)
        self.assertIn("pipeline_config", man.all_tasks_params)

        self.assertDictEqual(man.all_tasks_params["versions"], {})
        self.assertDictEqual(
            man.all_tasks_params["pipeline_config"],
            yaml.load(testconfig, Loader=yaml.SafeLoader),
        )

    def test_metadata_params(self):
        testconfig = """
        foo: bar
        pipeline:
            save_versions:
                - numpy
                - caput
            bla: "foo"
        """

        man = pipeline.Manager.from_yaml_str(testconfig)

        self.assertIn("versions", man.all_tasks_params)
        self.assertIn("pipeline_config", man.all_tasks_params)

        self.assertDictEqual(
            man.all_tasks_params["versions"],
            {"numpy": numpy.__version__, "caput": caput.__version__},
        )
        self.assertDictEqual(
            man.all_tasks_params["pipeline_config"],
            yaml.load(testconfig, Loader=yaml.SafeLoader),
        )

    def test_metadata_params_no_config(self):
        testconfig = """
        pipeline:
            save_versions: numpy
            save_config: False
        """

        man = pipeline.Manager.from_yaml_str(testconfig)

        self.assertIn("versions", man.all_tasks_params)
        self.assertIn("pipeline_config", man.all_tasks_params)

        self.assertDictEqual(
            man.all_tasks_params["versions"], {"numpy": numpy.__version__}
        )
        self.assertIsNone(man.all_tasks_params["pipeline_config"], {})
