"""
caput.task
==========

Base classes for building tasks for `caput.pipeline`.
"""

from . import _core
from ._core import (
    SingleTask as SingleTask,
    MPILoggedTask as MPILoggedTask,
    group_tasks as group_tasks,
)

from . import (
    debug as debug,
    basic as basic,
    io as io,
    random as random,
)
