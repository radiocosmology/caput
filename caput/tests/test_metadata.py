# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

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
        # remove line numbers
        pipeline_config = man.all_tasks_params["pipeline_config"]
        del pipeline_config["__line__"]
        del pipeline_config["pipeline"]["__line__"]
        self.assertDictEqual(
            pipeline_config,
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

        # remove line numbers
        pipeline_config = man.all_tasks_params["pipeline_config"]
        del pipeline_config["__line__"]
        del pipeline_config["pipeline"]["__line__"]
        self.assertDictEqual(
            pipeline_config,
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
