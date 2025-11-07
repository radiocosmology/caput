"""Extensions for :py:class:`~caput.pipeline.Task` and helper code to run pure Python pipelines.

.. warning::
    Most of the task extensions provided in this module are deprecated in
    favour of :py:class:`~caput.pipeline.tasklib.base.ContainerTask`, and
    may be removed in the future.
"""

from . import _configtypes, _extensions, _pytools
from ._configtypes import *
from ._extensions import *
from ._pytools import *


__all__ = [*_configtypes.__all__, *_extensions.__all__, *_pytools.__all__]
