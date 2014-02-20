#!/usr/bin/env python
"""Executes a data analysis pipeline given a pipeline YAML file.

This script, when executed on the command line, accepts a single parameter, the
path to a yaml pipeline file.  For an example of a pipeline file, see
documentation for caput.pipeline.

"""

import sys

from caput.pipeline import Manager

try:
    p, file_name = sys.argv
except ValueError:
    print "Takes one argument, the path to a YAML file."
else:
    P = Manager.from_yaml_file(file_name)
    P.run()

