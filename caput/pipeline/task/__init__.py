"""caput.pipeline.task.

Base classes for building tasks for `caput.pipeline`.

Submodules
----------
.. autosummary::
   :toctree: _autosummary

   debug
   flow
   io
   random
"""

from ._base import (
    SingleTask as SingleTask,
    MPILoggedTask as MPILoggedTask,
    SetMPILogging as SetMPILogging,
    group_tasks as group_tasks,
)

from . import (
    debug as debug,
    flow as flow,
    io as io,
    random as random,
)
