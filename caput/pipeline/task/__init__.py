"""caput.pipeline.task.

Base classes for building tasks for `caput.pipeline`.
"""

from ._core import (
    SingleTask as SingleTask,
    MPILoggedTask as MPILoggedTask,
    SetMPILogging as SetMPILogging,
    group_tasks as group_tasks,
)

from . import (
    debug as debug,
    basic as basic,
    io as io,
    random as random,
)
