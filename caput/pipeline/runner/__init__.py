r"""Command-line interface and helper functions for running pipelines.

This module contains functions to run and queue pipelines, along
with a command line interface (CLI) for user interaction.

Basic Functions
~~~~~~~~~~~~~~~

These functions just act as convenience wrappers around the core
pipeline management functionality.

- :py:func:`run_pipeline`
- :py:func:`template_run`
- :py:func:`lint_config`

Cluster Functions
~~~~~~~~~~~~~~~~~

These are additional functions to support running pipelines on compute clusters
using job schedulers like SLURM or PBS.

- :py:func:`queue`
- :py:func:`template_queue`
- :py:func:`register_system`
"""

from ._core import (
    run_pipeline as run_pipeline,
    template_run as template_run,
    lint_config as lint_config,
)
from ._scheduler import (
    template_queue as template_queue,
    queue as queue,
    register_system as register_system,
)
