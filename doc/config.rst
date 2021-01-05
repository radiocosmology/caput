.. _config:

Configuration
=============

The caput pipeline runner script accepts a YAML file for configuration. The structure of this file
is documented in :ref:`config`.

General options
---------------


Pipeline
--------

Logging
.......
The log levels can be configured in multiple ways:

- Use the `logging` section directly in the pipeline blog to define the root log level with either
  `DEBUG`, `INFO`, `WARNING` or `ERROR`. You can also also set log levels for single modules here
  and may add a root log level with the key `"root"`. The default is `{"root": "WARNING"}`

Examples:

::

  pipeline:
    logging:
      root: DEBUG
      annoying.module: INFO

would show `DEBUG` messages for everything, but `INFO` only for a module called `annoying.module`.

::

  pipeline:
    logging: ERROR

would reduce all loggin to `ERROR` messages.

- Set the `log_level` parameter of any task of type
  `draco.core.task.LoggedTask <https://github.com/radiocosmology/draco/blob/master/draco/core/task.py>`_.

- Further filter logging by MPI ranks using
  `draco.core.task.SetMPILogging <https://github.com/radiocosmology/draco/blob/master/draco/core/task.py>`_.
