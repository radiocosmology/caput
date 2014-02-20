"""Unit tests for pipeline flow manager.

Also contains the test task classes. These tasks are used for testing the flow 
and execution order of the pipeline. They manipulate strings as thier data and
print to stdout such that execution order can be verified.
"""

import unittest
import os

from ch_analysis.pipeline import Manager


pipe_file = (os.path.dirname(os.path.realpath( __file__ ))
             + "/input_test_pipe.yaml")

class TestPipeline(unittest.TestCase):
    """Just runs a pipeline to check for errors."""

    def test_pipeline(self):
        pipe = Manager.from_yaml_file(pipe_file)
        pipe.run()


if __name__ == '__main__':
    unittest.main()

