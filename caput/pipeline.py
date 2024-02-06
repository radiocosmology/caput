r"""Data Analysis and Simulation Pipeline.

A data analysis pipeline is completely specified by a YAML file that specifies
both what tasks are to be run and the parameters that go to those tasks.
Included in this package are base classes for simplifying the construction of
data analysis tasks, as well as the pipeline manager which executes them.

Pipelines are most easily executed using the script in `caput_pipeline.py`,
which ships with :mod:`caput`.

Flow control classes
====================
- :py:class:`Manager`
- :py:class:`PipelineRuntimeError`
- :py:class:`PipelineStopIteration`

Task base classes
=================
- :py:class:`TaskBase`
- :py:class:`SingleBase`
- :py:class:`IterBase`
- :py:class:`H5IOMixin`
- :py:class:`BasicContMixin`
- :py:class:`SingleH5Base`
- :py:class:`IterH5Base`



Examples
--------
Basic Tasks
-----------

A pipeline task is a subclass of :class:`TaskBase` intended to perform some small,
modular piece analysis. The developer of the task must specify what input
parameters the task expects as well as code to perform the actual processing
for the task.

Input parameters are specified by adding class attributes whose values are
instances of :class:`config.Property`. For instance a task definition might begin
with

>>> class SpamTask(TaskBase):
...     eggs = config.Property(proptype=str)

This defines a new task named :class:`SpamTask` with a parameter named *eggs*, whose
type is a string.  The class attribute :attr:`SpamTask.eggs` will replaced with an
instance attribute when an instance of the task is initialized, with it's value
read from the pipeline configuration YAML file (see next section).

The actual work for the task is specified by over-ridding any of the
:meth:`~TaskBase.setup`, :meth:`~TaskBase.next` or
:meth:`~TaskBase.finish` methods (:meth:`~TaskBase.__init__` may also be
implemented`).  These are executed in order, with :meth:`~TaskBask.next`
possibly being executed many times.  Iteration of :meth:`next` is halted by
raising a :exc:`PipelineStopIteration`.  Here is a example of a somewhat
trivial but fully implemented task:

>>> class PrintEggs(TaskBase):
...
...     eggs = config.Property(proptype=list)
...
...     def __init__(self):
...         super().__init__()
...         self.i = 0
...
...     def setup(self):
...         print("Setting up PrintEggs.")
...
...     def next(self):
...         if self.i >= len(self.eggs):
...             raise PipelineStopIteration()
...         print("Spam and %s eggs." % self.eggs[self.i])
...         self.i += 1
...
...     def finish(self):
...         print("Finished PrintEggs.")

Any return value of these three pipeline methods can be handled by the pipeline
and provided to subsequent tasks. The methods :meth:`setup` and :meth:`next`
may accept (positional only) arguments which will be received as the outputs of
early tasks in a pipeline chain. The following is an example of a pair of tasks
that are designed to operate in this manner.

>>> class GetEggs(TaskBase):
...
...     eggs = config.Property(proptype=list)
...
...     def __init__(self):
...         super().__init__()
...         self.i = 0
...
...     def setup(self):
...         print("Setting up GetEggs.")
...
...     def next(self):
...         if self.i >= len(self.eggs):
...             raise PipelineStopIteration()
...         egg = self.eggs[self.i]
...         self.i += 1
...         return egg
...
...     def finish(self):
...         print("Finished GetEggs.")

>>> class CookEggs(TaskBase):
...
...     style = config.Property(proptype=str)
...
...     def setup(self):
...         print("Setting up CookEggs.")
...
...     def next(self, egg):
...         print("Cooking %s %s eggs." % (self.style, egg))
...
...     def finish(self):
...         print("Finished CookEggs.")

Note that :meth:`CookEggs.next` never raises a :exc:`PipelineStopIteration`.
This is because there is no way for the task to internally know how long to
iterate.  :meth:`next` will continue to be called as long as there are inputs
for :meth:`next` and will stop iterating when there are none.

Pipeline Configuration
----------------------

To actually run a task or series of tasks, a YAML pipeline configuration is
required.  The pipeline configuration has two main functions: to specify the
the pipeline (which tasks are run, in which order and how to handle the inputs and
outputs of tasks) and to provide parameters to each individual task.  Here is
an example of a pipeline configuration:

>>> spam_config = '''
... pipeline :
...     tasks:
...         -   type:   PrintEggs
...             params: eggs_params
...
...         -   type:   GetEggs
...             params: eggs_params
...             out:    egg
...
...         -   type:   CookEggs
...             params: cook_params
...             in:     egg
...
... eggs_params:
...     eggs: ['green', 'duck', 'ostrich']
...
... cook_params:
...     style: 'fried'
...
... '''

Here the 'pipeline' section contains parameters that pertain to the pipeline as
a whole.  The most important parameter is *tasks*, a list of tasks to be
executed.  Each entry in this list may contain the following keys:

type
    (required) The name of the class relative to the global
    name space. Any required imports will be performed dynamically.  Any
    classes that are not importable (defined interactively) need to be
    registered in the dictionary ``pipeline.local_tasks``.
params
    (required) Key or list of keys referring to sections of the pipeline
    configuration holding parameters for the task.
out
    A 'pipeline product key' or list of keys that label any return values from
    :meth:`setup`, :meth:`next` or :meth:`finish`.
requires
    A 'pipeline product key' or list of keys representing values to be passed
    as arguments to :meth:`setup`.
in\_
    A 'pipeline product key' or list of keys representing values to be passed
    as arguments to :meth:`next`.

The sections other than 'pipeline' in the configuration contain the parameter
for the various tasks, as specified be the 'params' keys.

Execution Order
---------------

There are two options when choosing how to execute a pipeline: standard and legacy.
When the above pipeline is executed in standard mode, it produces the following output.

>>> local_tasks.update(globals())  # Required for interactive sessions.
>>> m = Manager.from_yaml_str(spam_config)
>>> m.run()
Setting up PrintEggs.
Setting up GetEggs.
Setting up CookEggs.
Spam and green eggs.
Spam and duck eggs.
Spam and ostrich eggs.
Finished PrintEggs.
Cooking fried green eggs.
Cooking fried duck eggs.
Cooking fried ostrich eggs.
Finished GetEggs.
Finished CookEggs.

When executed in legacy mode, it produces this output.

>>> local_tasks.update(globals())  # Required for interactive sessions.
>>> m = Manager.from_yaml_str(spam_config)
>>> m.execution_order = "legacy"
>>> m.run()
Setting up PrintEggs.
Setting up GetEggs.
Setting up CookEggs.
Spam and green eggs.
Cooking fried green eggs.
Spam and duck eggs.
Cooking fried duck eggs.
Spam and ostrich eggs.
Cooking fried ostrich eggs.
Finished PrintEggs.
Finished GetEggs.
Finished CookEggs.

To understand the differences, compare the rules for each strategy.
The `standard` method uses a priority system based on the following criteria,
in decreasing importance:

1. Task must be available to execute some step.
2. Task priority. This is set by two factors:

   * Dynamic priority: tasks which have a higher net consumption
     (inputs consumed minus outputs created).
   * Base priority: user-configurable base priority is added to
     the dynamic priority.

3. Pipeline configuration order.

If no tasks are available to run, the `legacy` method is used, which uses the
following execution order rules:

1. One of the methods `setup()`, `next()` or `finish()`, as appropriate, will
   be executed from each task, in order.
2. If the task method is missing its input, as specified by the 'requires' or 'in\_'
   keys, restart at the beginning of the `tasks` list.
3. If the input to `next()` is missing and the task is at the beginning of the
   list there will be no opportunity to generate this input. Stop iterating
   `next()` and proceed to `finish()`.
4. Once a task has executed `finish()`, remove it from the list.
5. Once a method from the last member of the `tasks` list is executed, restart
   at the beginning of the list.

The difference in outputs is because `PrintEggs` will always have higher priority
than `GetEggs`, so it will run to completion _before_ `GetEggs` starts generating
anything. Only once `PrintEggs` is done will the other tasks run. Even though
`CookEggs` has the highest priority, it cannot do anything without `GetEggs` running
first.

If the above `legacy` rules seem somewhat opaque, consider the following example which
illustrates these rules in a pipeline with a slightly more non-trivial flow.

>>> class DoNothing(TaskBase):
...
...     def setup(self):
...         print("Setting up DoNothing.")
...
...     def next(self, input):
...         print("DoNothing next.")
...
...     def finish(self):
...         print("Finished DoNothing.")

>>> local_tasks.update(globals())  # Required for interactive sessions only.
>>> new_spam_config = '''
... pipeline :
...     tasks:
...         -   type:   GetEggs
...             params: eggs_params
...             out:    egg
...
...         -   type:   CookEggs
...             params: cook_params
...             in:     egg
...
...         -   type:   DoNothing
...             params: no_params
...             in:     non_existent_data_product
...
...         -   type:   PrintEggs
...             params: eggs_params
...
... eggs_params:
...     eggs: ['green', 'duck', 'ostrich']
...
... cook_params:
...     style: 'fried'
...
... no_params: {}
... '''

The following would error, because the pipeline config is checked for errors, like an 'in\_' parameter without a
corresponding 'out'::

    m = Manager.from_yaml_str(new_spam_config)
    m.execution_order = "legacy"
    m.run()

But this is what it would produce otherwise::

    Setting up GetEggs.
    Setting up CookEggs.
    Setting up DoNothing.
    Setting up PrintEggs.
    Cooking fried green eggs.
    Cooking fried duck eggs.
    Cooking fried ostrich eggs.
    Finished GetEggs.
    Finished CookEggs.
    Finished DoNothing.
    Spam and green eggs.
    Spam and duck eggs.
    Spam and ostrich eggs.
    Finished PrintEggs.

Notice that :meth:`DoNothing.next` is never called, since the pipeline never
generates its input, 'non_existent_data_product'.  Once everything before
:class:`DoNothing` has been executed the pipeline notices that there is no
opertunity for 'non_existent_data_product' to be generated and forces
`DoNothing` to proceed to :meth:`finish`. This also unblocks :class:`PrintEggs`
allowing it to proceed normally.

Pure Python Pipelines
---------------------

It is possible to construct and run a pipeline purely within Python, which can be
useful for quick prototyping and debugging. This gives direct control over task
construction and configuration, and allows injection and inspection of pipeline
products.

To add a task to the pipeline you need to: create an instance of it; set any
configuration attributes directly (or call :meth:`~TaskBase.read_config` on an
appropriate dictionary); and then added to the pipeline using the
:meth:`~Manager.add_task` to add the instance and specify the queues it connects to.

To inject products into the pipeline, use the :class:`~Input` and supply it an
iterator as an argument. Each item will be fed into the pipeline one by one. To take
outputs from the pipeline, simply use the :class:`~Output` task. By default this
simply saves everything it receives into a list (which can be accessed via the task's
`outputs` attribute, e.g. with `save_output.outputs` after running the example below),
but it can be given a callback function to apply processing to each argument in turn.

>>> m = Manager()
>>> m.add_task(Input(["platypus", "dinosaur"]), out="key1")
>>> cook = CookEggs()
>>> cook.style = "coddled"
>>> m.add_task(cook, in_="key1")
>>> save_output = Output()
>>> m.add_task(save_output, in_="key1")
>>> print_output = Output(lambda x: print("I love %s eggs!" % x))
>>> m.add_task(print_output, in_="key1")
>>> m.execution_order = "legacy"
>>> m.run()
Setting up CookEggs.
Cooking coddled platypus eggs.
I love platypus eggs!
Cooking coddled dinosaur eggs.
I love dinosaur eggs!
Finished CookEggs.

Advanced Tasks
--------------

Several subclasses of :class:`TaskBase` provide advanced functionality for tasks that
conform to the most common patterns. This functionality includes: optionally
reading inputs from disk, instead of receiving them from the pipeline;
optionally writing outputs to disk automatically; and caching the results of a
large computation to disk in an intelligent manner (not yet implemented).

Base classes providing this functionality are :class:`SingleBase` for 'one
shot' tasks and :class:`IterBase` for task that need to iterate.  There are
limited to a single input ('in' key) and a single output ('out' key).  Method
:meth:`~SingleBase.process` should be overwritten instead of :meth:`next`.
Optionally, :meth:`~SingleBase.read_input` and :meth:`~SingleBase.write_output`
may be over-ridden for maximum functionality.  :meth:`setup` and :meth:`finish`
may be overridden as usual.

In addition :class:`SingleH5Base`, :class:`IterH5Base`, provide the
:meth:`read_input` and :meth:`write_output` methods for the most common
formats.

See the documentation for these base classes for more details.

"""

import importlib
import inspect
import logging
import os
import queue
import re
import traceback
import warnings
from copy import deepcopy

import yaml

from . import config, fileformats, misc, mpiutil, tools

# Set the module logger.
logger = logging.getLogger(__name__)


# Search this dictionary for tasks.
# Used for interactive sessions when task can't be specified by absolute path.
local_tasks = {}


# Exceptions
# ----------


class PipelineRuntimeError(Exception):
    """Raised when there is a pipeline related error at runtime."""

    pass


class PipelineStopIteration(Exception):
    """Stop the iteration of `next()` in pipeline tasks.

    Pipeline tasks should raise this excetions in the `next()` method to stop
    the iteration of the task and to proceed to `finish()`.

    Note that if `next()` recieves input data as an argument, it is not
    required to ever raise this exception.  The pipeline will proceed to
    `finish()` once the input data has run out.

    """

    pass


class _PipelineMissingData(Exception):
    """Used for flow control when input data is yet to be produced."""

    pass


class _PipelineFinished(Exception):
    """Raised by tasks that have been completed."""

    pass


# Pipeline Manager
# ----------------


def _get_versions(modules):
    """Get the versions of a list of python modules.

    Parameters
    ----------
    modules : List[str]
        Names of python modules.

    Returns
    -------
    Dict[str, str]
    """
    if isinstance(modules, str):
        modules = [modules]
    if not isinstance(modules, list):
        raise config.CaputConfigError(
            f"Value of 'save_versions' is of type '{type(modules).__name__}' "
            "(expected 'str' or 'list(str)')."
        )
    versions = {}
    for module in modules:
        if not isinstance(module, str):
            raise config.CaputConfigError(
                f"Found value of type '{type(module).__name__}' in list "
                "'save_versions' (expected 'str')."
            )
        try:
            versions[module] = importlib.import_module(module).__version__
        except ModuleNotFoundError as err:
            raise config.CaputConfigError(
                "Failure getting versions requested with config parameter "
                "'save_versions'."
            ) from err
    return versions


class Manager(config.Reader):
    r"""Pipeline manager for setting up and running pipeline tasks.

    The manager is in charge of initializing all pipeline tasks, setting them
    up by providing the appropriate parameters, then executing the methods of
    the each task in the appropriate order. It also handles intermediate data
    products and ensuring that the correct products are passed between tasks.

    Attributes
    ----------
    logging : Dict(str, str)
        Log levels per module. The key "root" stores the root log level.
    multiprocessing : int
        TODO
    cluster : dict
        TODO
    task_specs : list
        Configuration of pipeline tasks.
    execution_order : str
        Set the task execution order for this pipeline instance. `legacy` round-robins
        through all tasks based on the config order, and tries to clear out finished
        tasks as soon as possible. `standard` uses a priority and availability system
        to select the next task to run, and falls back to `legacy` if nothing is available.
    key_pattern : str, optional
        Regex pattern to match on in order to pass a key to subsequent tasks. This is
        useful for controlling which keys are passed in tasks which produce multiple
        outputs. Default is `[^\W_]`, which will cause any key that contains no
        alphanumeric characters to be ignored.
    save_versions : list
        Module names (str). This list together with the version strings from these
        modules are attached to output metadata. Default is [].
    save_config : bool
        If this is True, the global pipeline configuration is attached to output
        metadata. Default is `True`.
    psutil_profiling : bool
        Use psutil to profile CPU and memory usage. Default is `False`.
    """

    logging = config.logging_config(default={"root": "WARNING"})
    multiprocessing = config.Property(default=1, proptype=int)
    cluster = config.Property(default={}, proptype=dict)
    task_specs = config.Property(default=[], proptype=list, key="tasks")
    execution_order = config.enum(["standard", "legacy"], default="standard")
    key_pattern = config.Property(proptype=str, default=r"[^\W_]")

    # Options to be stored in self.all_tasks_params
    versions = config.Property(default=[], proptype=_get_versions, key="save_versions")
    save_config = config.Property(default=True, proptype=bool)

    def __init__(self, psutil_profiling=False):
        # Initialise the list of task instances
        self.tasks = []
        self.all_params = []
        self.all_tasks_params = []

        self._psutil_profiling = psutil_profiling

        # Precompile the key pattern to skip
        self.key_match = re.compile(self.key_pattern)

        logger.debug(
            f"CPU and memory profiling using psutil {'enabled' if self._psutil_profiling else 'disabled'}."
        )

    @classmethod
    def from_yaml_file(cls, file_name, lint=False, psutil_profiling=False):
        """Initialize the pipeline from a YAML configuration file.

        Parameters
        ----------
        file_name: string
            Path to YAML pipeline configuration file.
        lint : bool
            Instantiate Manager only to lint config. Disables debug logging.
        psutil_profiling : bool
            Use psutil to profile CPU and memory usage

        Returns
        -------
        self: Pipeline object
        """
        try:
            with open(file_name) as f:
                yaml_doc = f.read()
        except TypeError as e:
            raise config.CaputConfigError(
                f"Unable to open yaml file ({file_name}): {e}",
                file_=file_name,
            )
        return cls.from_yaml_str(yaml_doc, lint, psutil_profiling)

    @classmethod
    def from_yaml_str(cls, yaml_doc, lint=False, psutil_profiling=False):
        """Initialize the pipeline from a YAML configuration string.

        Parameters
        ----------
        yaml_doc: string
            Yaml configuration document.
        lint : bool
            Instantiate Manager only to lint config. Disables debug logging.
        psutil_profiling : bool
            Use psutil to profile CPU and memory usage.

        Returns
        -------
        self: Pipeline object
        """
        from .config import SafeLineLoader

        yaml_params = yaml.load(yaml_doc, Loader=SafeLineLoader)
        try:
            if not isinstance(yaml_params["pipeline"], dict):
                raise config.CaputConfigError(
                    "Value 'pipeline' in YAML configuration is of type "
                    f"`{type(yaml_params['pipeline']).__name__}` (expected a dict here).",
                    location=yaml_params,
                )
        except TypeError as e:
            raise config.CaputConfigError(
                "Couldn't find key 'pipeline' in YAML configuration document.",
                location=yaml_params,
            ) from e

        self = cls.from_config(
            yaml_params["pipeline"], psutil_profiling=psutil_profiling
        )
        self.all_params = yaml_params
        self.all_tasks_params = {
            "versions": self.versions,
            "pipeline_config": self.all_params if self.save_config else None,
        }

        self._setup_logging(lint)
        self._setup_tasks()

        return self

    def _setup_logging(self, lint=False):
        """Set up logging based on the config.

        Parameters
        ----------
        lint : bool
            Instantiate Manager only to lint config. Disables debug logging.
        """
        # set root log level and set up default formatter
        loglvl_root = self.logging.get("root", "WARNING")

        # Don't allow INFO log level when linting
        if lint and loglvl_root == "DEBUG":
            loglvl_root = "INFO"

        logging.basicConfig(level=getattr(logging, loglvl_root))
        for module, level in self.logging.items():
            if module != "root":
                logging.getLogger(module).setLevel(getattr(logging, level))

    def run(self):
        """Main driver for the pipeline.

        This function initializes all pipeline tasks and runs the pipeline
        through to completion.

        Raises
        ------
        PipelineRuntimeError
            If a task stage returns the wrong number of outputs.

        """
        from .profile import PSUtilProfiler

        # Log MPI information
        if mpiutil._comm is not None:
            logger.debug(f"Running with {mpiutil.size} MPI process(es)")
        else:
            logger.debug("Running in single process without MPI.")

        # Index of first task in the list which has
        # not finished running
        self._task_head = 0
        # Pointer to next task index
        self._task_idx = 0

        # Choose how to order tasks based on the execution order
        next_task = (
            self._iter_tasks if self.execution_order == "legacy" else self._next_task
        )

        logger.debug(f"Using `{self.execution_order}` iteration method.")

        # Run the pipeline.
        while True:
            # Get the next task. `StopIteration` is raised when there are no
            # non-None tasks left in the tasks list
            try:
                task = next_task()
            except StopIteration:
                # No tasks remaining
                break

            with PSUtilProfiler(
                self._psutil_profiling, str(task), logger=getattr(task, "log", logging)
            ):
                try:
                    out = task._pipeline_next()
                # Raised if either `setup` or `next` was called without
                # enough available inputs
                except _PipelineMissingData:
                    # If this is the first task in the task list, it can't receive
                    # any more inputs and should advance its state
                    if self._task_idx == self._task_head:
                        logger.debug(
                            f"{task!s} missing input data and "
                            "is at beginning of task list. Advancing state."
                        )
                        task._pipeline_advance_state()
                    else:
                        # Restart from the beginning of the task list
                        self._task_idx = self._task_head
                    continue
                # Raised if the task has finished
                except _PipelineFinished:
                    # Overwrite the task to maintain task list indices
                    self.tasks[self._task_idx] = None
                    # Update the first available task index
                    for ii, t in enumerate(self.tasks[self._task_head :]):
                        if t is not None:
                            self._task_head += ii
                            break
                    continue

            if self.execution_order == "legacy":
                # Advance the task pointer
                self._task_idx += 1

            # Ensure the output(s) are correctly structured
            out = self._check_task_output(out, task)

            if out is None:
                continue

            # Queue outputs for any associated tasks
            for key, product in zip(task._out_keys, out):
                # Purposefully skip an output that does not match
                # the allowed key pattern
                if not self.key_match.search(key):
                    continue
                # Try to pass this product to each task
                received = [
                    recv._pipeline_queue_product(key, product)
                    for recv in self.tasks
                    if recv is not None
                ]

                if not any(received):
                    # Just warn. This probably shouldn't happen, but there
                    # could be some edge cases to deal with
                    logger.info(
                        f"Task {task!s} tried to pass key {key} "
                        "but no task was found to accept it."
                    )

        # Pipeline is done
        logger.info("FIN")

    def _next_task(self):
        """Get the next task to run from the task list.

        Task is chosen based the following criteria:
        - able to do something in its current state
        - highest priority
        - highest base priority
        - next in pipeline config order

        If no task is available to do anything, restart from the
        pipeline task head.
        """
        # Get a list of tasks which are availble to run
        available = []

        for ii in range(len(self.tasks)):
            # Loop through tasks starting at the current index. Including
            # the current tasks first ensures we clear out completed
            # tasks faster
            jj = (ii + self._task_idx) % len(self.tasks)
            task = self.tasks[jj]

            if task is None:
                continue

            if task._pipeline_is_available:
                available.append(jj)

        if not available:
            # Nothing is currently available, so fall back to a
            # blind loop starting at the first task. If there is
            # nothing left, this will raise StopIteration.
            self._task_idx = self._task_head
            return self._iter_tasks()

        # Reverse sort the available tasks first by priority and second
        # by base priority such that for any two tasks with equal priority,
        # the task with highest base priority will be selected
        new_index = sorted(
            available,
            key=lambda i: (self.tasks[i].priority, self.tasks[i].base_priority),
            reverse=True,
        )[0]

        # Ensure that all ranks are running the same task.
        # This probably should never be needed with the current
        # priority selection. Effectively a no-op if no MPI
        self._task_idx = mpiutil.bcast(new_index, root=0)

        return self.tasks[self._task_idx]

    def _iter_tasks(self):
        """Iterate through tasks in order and return the next in order.

        This method implements the `legacy` execution order, and is used
        as a fallback for the `standard` processing order when no task is
        available to run.
        """
        for ii in range(len(self.tasks)):
            # Iterate starting at the next task
            jj = (ii + self._task_idx) % len(self.tasks)
            task = self.tasks[jj]

            if task is not None:
                # Update the task pointer
                self._task_idx = jj
                return task

        # If all tasks are None, the pipeline is done
        raise StopIteration

    @staticmethod
    def _check_task_output(out, task):
        """Check if task stage's output is as expected.

        Returns
        -------
        out : Same as `TaskBase.next` or None
            Pipeline product, or None if there is no output of the task stage that
            has to be handled further.

        Raises
        ------
        PipelineRuntimeError
            If a task stage returns the wrong number of outputs.
        """
        # This iteration supplied no output, or the output is not
        # meant to be handled by the pipeline
        if out is None or len(task._out_keys) == 0:
            return None

        if len(task._out_keys) == 1:
            # if tuple, in config file written as `out: out_key`, No
            # unpacking if `out` is a length 1 sequence. If list,
            # in config file written as `out: [out_key,]`.
            # `out` must be a length 1 sequence.
            if isinstance(task._out_keys, tuple):
                if not isinstance(out, tuple):
                    out = (out,)

        elif len(task._out_keys) != len(out):
            raise PipelineRuntimeError(
                f"Found unexpected number of outputs in {task!s} "
                f"(got {len(out)} expected {len(task._out_keys)})"
            )

        logger.debug(
            f"{task!s} produced output data product with keys {task._out_keys!s}"
        )

        return out

    def _setup_tasks(self):
        """Create and setup all tasks from the task list."""
        # Validate that all inputs have a corresponding output key.
        self._validate_task_inputs()

        # Setup all tasks in the task list
        for ii, task_spec in enumerate(self.task_specs):
            try:
                # Load the task instance and add it to the pipeline
                task = self._get_task_from_spec(task_spec)
                self.add_task(task, task_spec)
            except config.CaputConfigError as e:
                raise config.CaputConfigError(
                    f"Setting up task {ii} caused an error:\n\t{traceback.format_exc()}",
                    location=task_spec if e.line is None else e.line,
                ) from e

    def _validate_task_inputs(self):
        # Make sure all tasks' in/requires values have corresponding
        # out keys from another task
        all_out_values = []
        for t in self.task_specs:
            if "out" in t:
                if isinstance(t["out"], (list, tuple)):
                    all_out_values.extend(t["out"])
                else:
                    all_out_values.append(t["out"])

        unique_out_values = set(all_out_values)

        # Multiple tasks produce output with the same key
        if len(unique_out_values) != len(all_out_values):
            dup_keys = [k for k in unique_out_values if all_out_values.count(k) > 1]
            raise config.CaputConfigError(
                f"Duplicate output keys: outputs {dup_keys} were found "
                "to come from multiple tasks."
            )

        for task_spec in self.task_specs:
            in_ = task_spec.get("in", None)
            requires = task_spec.get("requires", None)

            for key, value in (["in", in_], ["requires", requires]):
                if value is None:
                    continue
                if not isinstance(value, list):
                    value = [value]
                for v in value:
                    if v not in unique_out_values:
                        raise config.CaputConfigError(
                            f"Value '{key}' for task {task_spec['type']} has no corresponding "
                            f"`out` from another task (Value {v} is not in {unique_out_values})."
                        )

    def _get_task_from_spec(self, task_spec: dict):
        """Set up a pipeline task from the spec given in the tasks list."""
        # Check that only the expected keys are in the task spec.
        for key in task_spec.keys():
            if key not in ["type", "params", "requires", "in", "out"]:
                raise config.CaputConfigError(
                    f"Task got an unexpected key '{key}' in 'tasks' list."
                )

        # 'type' is a required key.
        try:
            task_path = task_spec["type"]
        except KeyError as e:
            raise config.CaputConfigError("'type' not specified for task.") from e

        # Find the tasks class either in the local set, or by importing a fully
        # qualified class name
        if task_path in local_tasks:
            task_cls = local_tasks[task_path]
        else:
            try:
                task_cls = misc.import_class(task_path)
            except (config.CaputConfigError, AttributeError, ModuleNotFoundError) as e:
                raise config.CaputConfigError(
                    f"Loading task `{task_path}` caused an error:\n\t{traceback.format_exc()}"
                ) from e

        # Get the parameters and initialize the class.
        params = {}
        if "params" in task_spec:
            # If params is a dict, assume params are inline
            if isinstance(task_spec["params"], dict):
                params.update(task_spec["params"])

            # Otherwise assume it's a list of keys
            else:
                param_keys = task_spec["params"]

                # Must be a list of keys, convert if only one key was specified.
                if not isinstance(param_keys, list):
                    param_keys = [param_keys]

                # Locate param sections, and add to dict
                for param_key in param_keys:
                    try:
                        params.update(self.all_params[param_key])
                    except KeyError as e:
                        raise config.CaputConfigError(
                            f"Parameter group {param_key} not found in config."
                        ) from e

        # add global params to params
        task_params = deepcopy(self.all_tasks_params)
        task_params.update(params)

        # Create and configure the task instance
        try:
            task = task_cls._from_config(task_params)
        except config.CaputConfigError as e:
            raise config.CaputConfigError(
                f"Failed instantiating {task_cls} from config.\n\t{traceback.format_exc()}",
                location=task_spec.get("params", task_spec),
            ) from e

        return task

    def add_task(self, task, task_spec: dict = {}, **kwargs):
        r"""Add a task instance to the pipeline.

        Parameters
        ----------
        task : TaskBase
            A pipeline task instance.
        task_spec : dict
            include optional argument: requires, in\_, out : list or string
            The names of the task inputs and outputs.
        **kwargs : dict
            Included for legacy purposes. Alternative method to provide
            `requires`, `in\_`, and `out` arguments. These should *only*
            be provided if `task_spec` is not provided - a ValueError
            will be raised otherwise.

        Raises
        ------
        caput.config.CaputConfigError
            If there was an error in the task configuration.
        """

        def _check_duplicate(key0: str, key1: str, d0: dict, d1: dict):
            """Check if an argument has been provided twice."""
            val0 = d0.get(key0, d0.get(key1))
            val1 = d1.get(key0, d1.get(key1))

            # Check that the key has not been provided twice. It's
            # ok to return None, we only care if *both* values are
            # not None
            if val0 is None:
                return val1

            if val1 is None:
                return val0

            raise ValueError(f"Argument `{key0}/{key1}` was provided twice")

        requires = _check_duplicate("requires", "requires", task_spec, kwargs)
        in_ = _check_duplicate("in", "in_", task_spec, kwargs)
        out = _check_duplicate("out", "out", task_spec, kwargs)

        try:
            task._setup_keys(in_, out, requires)
        # Want to blindly catch errors
        except Exception as e:
            raise config.CaputConfigError(
                f"Adding task {task!s} caused an error:\n\t{traceback.format_exc()}"
            ) from e

        self.tasks.append(task)
        logger.debug(f"Added {task!s} to task list.")


# Pipeline Task Base Classes
# --------------------------


class TaskBase(config.Reader):
    """Base class for all pipeline tasks.

    All pipeline tasks should inherit from this class, with functionality and
    analysis added by over-riding `__init__`, `setup`, `next` and/or
    `finish`.

    In addition, input parameters may be specified by adding class attributes
    which are instances of `config.Property`. These will then be read from the
    pipeline yaml file when the pipeline is initialized.  The class attributes
    will be overridden with instance attributes with the same name but with the
    values specified in the pipeline file.

    Attributes
    ----------
    broadcast_inputs : bool
        If true, input queues will be broadcast to process all combinations of
        entries. Otherwise, items in input queues are removed at equal rate.
        NOT CURRENTLY IMPLEMENTED
    limit_outputs : int
        Limits the number of `next` outputs from this task before finishing.
        Default is None, allowing an unlimited number of `next` products.
    base_priority : int
        Base integer priority. Priority only matters relative to other tasks
        in a pipeline, with run order given by `sorted(priorities, reverse=True)`.
        Task priority is also adjusted based on net difference in input and output,
        which will typically adjust priority by +/- (0 to 2). `base_priority` should
        be set accordingly - factors of 10 (i.e. -10, 10, 20, ...) are effective at
        forcing a task to have highest/lowest priority relative to other tasks.
        `base_priority` should be used sparingly when a user wants to enforce a
        specific non-standard pipeline behaviour. See method `priority` for details
        about dynamic priority.
    """

    broadcast_inputs = config.Property(proptype=bool, default=False)
    limit_outputs = config.Property(proptype=int, default=None)
    base_priority = config.Property(proptype=int, default=0)

    # Overridable Attributes
    # -----------------------

    def __init__(self):
        """Initialize pipeline task.

        May be overridden with no arguments.  Will be called after any
        `config.Property` attributes are set and after 'input' and 'requires'
        keys are set up.
        """
        pass

    def __str__(self):
        """Clean string representation of the task and its state.

        If no state has been set yet, the state is None.
        """
        state = getattr(self, "_pipeline_state", None)

        return f"{self.__class__.__name__}.{state}"

    def setup(self, requires=None):
        """First analysis stage of pipeline task.

        May be overridden with any number of positional only arguments
        (defaults are allowed).  Pipeline data-products will be passed as
        specified by `requires` keys in the pipeline setup.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.
        """
        pass

    def validate(self):
        """Validate the task after instantiation."""
        pass

    def next(self, input=None):
        """Iterative analysis stage of pipeline task.

        May be overridden with any number of positional only arguments
        (defaults are allowed).  Pipeline data-products will be passed as
        specified by `in` keys in the pipeline setup.

        Function will be called repetitively until it either raises a
        `PipelineStopIteration` or, if accepting inputs, runs out of input
        data-products.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.
        """
        raise PipelineStopIteration()

    def finish(self):
        """Final analysis stage of pipeline task.

        May be overridden with no arguments.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.
        """
        pass

    @property
    def embarrassingly_parallelizable(self):
        """Override to return `True` if `next()` is trivially parallelizeable.

        This property tells the pipeline that the problem can be parallelized
        trivially. This only applies to the `next()` method, which should not
        change the state of the task.

        If this returns `True`, then the Pipeline will execute `next()` many
        times  in parallel and handle all the intermediate data efficiently.
        Otherwise `next()` must be parallelized internally if at all. `setup()`
        and `finish()` must always be parallelized internally.

        Usage of this has not implemented.
        """
        return False

    @property
    def cacheable(self):
        """Override to return `True` if caching results is implemented.

        No caching infrastructure has yet been implemented.
        """
        return False

    # Pipeline Infrastructure
    # -----------------------

    @property
    def _pipeline_is_available(self):
        """True if this task can be run."""
        if not hasattr(self, "_pipeline_state"):
            # This task hasn't been initialized
            return False

        if self._pipeline_state == "setup":
            # True if all `requires` items have been provided
            # This also returns True is `self._requires` is empty
            return all(r is not None for r in self._requires)

        if self._pipeline_state == "next":
            # True if there is at least one input available
            # in each input queue.
            return bool(min((q.qsize() for q in self._in), default=0))

        # Otherwise, this task is likely done and can be run to
        # see if anything else happens
        return True

    @property
    def priority(self):
        """Return the priority associated with this task.

        If the task is not yet initialized, dynamic priority is zero.

        If the task in in state `setup`, dynamic priority is one
        if all `requires` items are stashed and zero otherwise.

        If the task is in state `next`, dynamic priority is the total
        net consumption of the task.

        For example:
        - A task which consumes 2 items, produces one, and can currently run
        once will have priority (2 - 1) * 1 + base = 1 + base
        - A task which does not consume anything but produces one item
        will have priority (0 - 1) * 1 + base = -1 + base

        In any other state, priority is just net consumption for one
        iteration.

        The priority returned is the sum of `base_priority` and the
        calculated dynamic priority.

        Returns
        -------
        priority : int
            `base_priority` plus dynamic priority calculated based on
            task state and inputs/outputs
        """
        if not hasattr(self, "_pipeline_state"):
            # This task hasn't been initialized
            p = 0

        elif self._pipeline_state == "setup":
            # 1 if all requirements are available or no requirements,
            # zero if requirements are needed but not available
            p = int(all(r is not None for r in self._requires))

        elif self._pipeline_state == "next":
            # Calculate the total net consumption of the task
            p = len(self._in_keys) - len(self._out_keys)
            # How many times can the task run?
            p *= min((q.qsize() for q in self._in), default=1)

        else:
            # If a task has passed the above states, it should be
            # finished quickly so set a very high priority
            p = 1e10

        return p + self.base_priority

    @property
    def mem_used(self):
        """Return the approximate total memory referenced by this task."""
        return tools.total_size(self)

    @classmethod
    def _from_config(cls, config):
        self = cls.__new__(cls)
        # Check for unused keys, but ignore the ones not put there by the user.
        self.read_config(config, compare_keys=["versions", "pipeline_config"])
        self.__init__()

        return self

    def _setup_keys(self, in_=None, out=None, requires=None):
        """Setup the 'requires', 'in' and 'out' keys for this task."""
        # Parse the task spec.
        requires = _format_product_keys(requires)
        in_ = _format_product_keys(in_)
        out = _format_product_keys(out)

        # Inspect the `setup` method to see how many arguments it takes.
        setup_argspec = inspect.getfullargspec(self.setup)

        try:
            len_defaults = len(setup_argspec.defaults)
        # defaults is None
        except TypeError:
            len_defaults = 0

        min_req = len(setup_argspec.args) - len_defaults - 1

        # Make sure it matches `requires` keys list specified in config.
        n_requires = len(requires)

        if n_requires < min_req:
            raise config.CaputConfigError(
                "Didn't get enough 'requires' keys. Expected at least "
                f"{min_req} and only got {n_requires}."
            )

        if n_requires > len(setup_argspec.args) - 1 and setup_argspec.varargs is None:
            raise config.CaputConfigError(
                "Got too many 'requires' keys. Expected at most "
                f"{len(setup_argspec.args) - 1} and got {n_requires}."
            )

        # Inspect the `next` method to see how many arguments it takes.
        next_argspec = inspect.getfullargspec(self.next)

        try:
            len_defaults = len(next_argspec.defaults)
        except TypeError:  # defaults is None
            len_defaults = 0

        min_in = len(next_argspec.args) - len_defaults - 1

        # Make sure it matches `in` keys list specified in config.
        n_in = len(in_)

        if n_in < min_in:
            raise config.CaputConfigError(
                "Didn't get enough 'in' keys. Expected at least "
                f"{min_in} and only got {n_in}."
            )

        if n_in > len(next_argspec.args) - 1 and next_argspec.varargs is None:
            raise config.CaputConfigError(
                "Got too many 'in' keys. Expected at most "
                f"{len(next_argspec.args) - 1} and got {n_in}."
            )

        # Now that all data product keys have been verified to be valid, store
        # them on the instance.
        self._requires_keys = requires
        # Set up a list with the number of required entries
        self._requires = [None] * n_requires
        # Store input keys
        self._in_keys = in_
        # Make a list with one queue for each input. Since any given input can
        # produce multiple values, queue up items which may be used in the
        # future
        self._in = [queue.Queue() for _ in range(n_in)]
        # Store output keys
        self._out_keys = out
        # Keep track of the number of times this task has produced output
        self._num_iters = 0

        if self.broadcast_inputs:
            # Additional queues to help manage inputs when broadcasting
            # self._bcast_queue = [queue.Queue() for _ in range(n_in)]
            raise NotImplementedError

        # Do any extra validation here
        self.validate()
        # Put pipeline in state such that `setup` is the next stage called.
        self._pipeline_advance_state()

    def _pipeline_advance_state(self):
        """Advance this pipeline task to the next stage.

        The task stages are 'setup', 'next', 'finish' or 'raise'.  This
        method sets the state of the task, advancing it to the next stage.

        Also performs some clean up tasks and checks associated with changing
        stages.
        """
        if not hasattr(self, "_pipeline_state"):
            self._pipeline_state = "setup"

        elif self._pipeline_state == "setup":
            # Advance the state to `next`
            self._pipeline_state = "next"
            # Make sure setup received all input. If not, go straight to
            # `finish`, because some input requirement was never generated.
            for req, req_key in zip(self._requires, self._requires_keys):
                if req is None:
                    warnings.warn(
                        f"Task {self!s} tried to advance to `next` "
                        f"without completing `setup`. Input `{req_key}` was never received. "
                        "Advancing to `finish`."
                    )
                    self._pipeline_state = "finish"
            # Overwrite inputs to free memory.
            self._requires = None

        elif self._pipeline_state == "next":
            # Make sure input queues are empty then delete them so no more data
            # can be queued.
            for in_, in_key in zip(self._in, self._in_keys):
                if not in_.empty():
                    warnings.warn(
                        f"Task {self!s} finished iterating `next()` "
                        f"but input queue `{in_key}` isn't empty."
                    )

            self._in = None
            self._pipeline_state = "finish"

        elif self._pipeline_state == "finish":
            self._pipeline_state = "raise"

        elif self._pipeline_state == "raise":
            pass

        else:
            raise PipelineRuntimeError()

    def _pipeline_next(self):
        """Execute the next stage of the pipeline.

        Execute `setup()`, `next()`, `finish()` or raise `PipelineFinished`
        depending on the state of the task.  Advance the state to the next
        stage if applicable.
        """
        if self._pipeline_state == "setup":
            # Check if we have all the required input data.
            for req in self._requires:
                if req is None:
                    raise _PipelineMissingData()

            logger.debug(f"Task {self!s} calling 'setup()'.")

            out = self.setup(*tuple(self._requires))
            self._pipeline_advance_state()

            return out

        if self._pipeline_state == "next":
            # Check if we have all the required input data.
            for in_ in self._in:
                if in_.empty():
                    raise _PipelineMissingData()

            if self.broadcast_inputs:
                raise NotImplementedError
            else:  # noqa RET506
                # Get the next set of data to be run.
                args = tuple(in_.get() for in_ in self._in)

            # Call the next iteration of `next`. If it is done running,
            # advance the task state and continue
            logger.debug(f"Task {self!s} calling 'next()'.")

            try:
                out = self.next(*args)
            except PipelineStopIteration:
                # Finished iterating `next()`.
                self._pipeline_advance_state()
                out = None

            if out is not None:
                self._num_iters += 1
                # If this task has a restricted number of outputs, it should advance
                # if enough output iterations have been executed
                if (
                    self.limit_outputs is not None
                    and self._num_iters >= self.limit_outputs
                ):
                    logger.info(
                        f"Task {self!s} reached maximum number of output "
                        f"iterations ({self.limit_outputs}). Advancing state."
                    )
                    self._pipeline_advance_state()

            return out

        if self._pipeline_state == "finish":
            logger.debug(f"Task {self!s} calling 'finish()'.")

            out = self.finish()
            self._pipeline_advance_state()

            return out

        if self._pipeline_state == "raise":
            raise _PipelineFinished()

        raise PipelineRuntimeError()

    def _pipeline_queue_product(self, key, product):
        """Put a product into an input queue as applicable.

        Add a product to either a `requires` slot or an input queue based
        on the associated key.
        """
        result = False

        # First, check requires keys
        if key in self._requires_keys:
            ii = self._requires_keys.index(key)
            logger.debug(
                f"{self!s} stowing data product with key {key} for `requires`."
            )
            if self._requires is None:
                raise PipelineRuntimeError(
                    "Tried to set 'requires' data product, but `setup()` already run."
                )
            if self._requires[ii] is not None:
                raise PipelineRuntimeError(
                    "'requires' data product set more than once."
                )
            self._requires[ii] = product

            result = True

        if key in self._in_keys:
            ii = self._in_keys.index(key)
            logger.debug(f"{self!s} stowing data product with key {key} for `in`.")
            if self._in is None:
                raise PipelineRuntimeError(
                    "Tried to queue 'in' data product, but `next()` already run."
                )

            self._in[ii].put(product)

            result = True

        return result


class _OneAndOne(TaskBase):
    """Base class for tasks that have (at most) one input and one output.

    This is not a user base class and simply holds code that is common to
    `SingleBase` and `IterBase`.
    """

    input_root = config.Property(default="None", proptype=str)
    output_root = config.Property(default="None", proptype=str)
    output_format = config.file_format()

    def __init__(self):
        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = inspect.getfullargspec(self.process)
        n_args = len(pro_argspec.args) - 1
        if n_args == 0:
            self._no_input = True
        else:
            self._no_input = False

    def process(self, input):
        """Override this method with your data processing task."""
        output = input

        return output  # noqa: RET504

    def validate(self):
        """Validate the task after instantiation.

        May be overriden to add any special task validation before the task is run.
        This is called by the :py:class:`Manager` after it's added to the pipeline and has special attributes like
        `_requires_keys`, `_requires`, `_in_keys`, `_in`, `-_out_keys` set.
        Call `super().validate()` if you overwrite this.

        Raises
        ------
        caput.config.CaputConfigError
            If there was an error in the task configuration.
        """
        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = inspect.getfullargspec(self.process)
        n_args = len(pro_argspec.args) - 1

        if n_args > 1:
            msg = "`process` method takes more than 1 argument, which is not allowed."
            raise config.CaputConfigError(msg)
        if (
            pro_argspec.varargs
            or pro_argspec.varkw
            or pro_argspec.kwonlyargs
            or pro_argspec.defaults
        ):
            msg = (
                "`process` method may not have variable length or optional"
                " arguments."
            )
            raise config.CaputConfigError(msg)

        # Make sure we know where to get the data from.
        if self.input_root == "None":
            if len(self._in) != n_args:
                msg = (
                    "No data to iterate over. 'input_root' is 'None' and"
                    " there are no 'in' keys."
                )
                raise config.CaputConfigError(msg)
        else:
            if len(self._in) != 0:
                msg = (
                    "For data input, supplied both a file path and an 'in'"
                    " key.  If not reading to disk, set 'input_root' to"
                    " 'None'."
                )
                raise config.CaputConfigError(msg)
            if n_args != 1:
                msg = (
                    "Reading input from disk but `process` method takes no"
                    " arguments."
                )
                raise config.CaputConfigError(msg)

    def read_process_write(self, input, input_filename, output_filename):
        """Reads input, executes any processing and writes output."""
        # Read input if needed.
        if input is None and not self._no_input:
            if input_filename is None:
                raise RuntimeError("No file to read from.")
            input_filename = self.input_root + input_filename
            input_filename = os.path.expanduser(input_filename)
            logger.info(
                "%s reading data from file %s.", self.__class__.__name__, input_filename
            )
            input = self.read_input(input_filename)
        # Analyse.
        if self._no_input:
            if input is not None:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set.")
            output = self.process()
        else:
            output = self.process(input)
        # Write output if needed.
        if self.output_root != "None" and output is not None:
            if output_filename is None:
                raise RuntimeError("No file to write to.")
            output_filename = self.output_root + output_filename
            output_filename = os.path.expanduser(output_filename)
            logger.info(
                "%s writing data to file %s.", self.__class__.__name__, output_filename
            )
            output_dirname = os.path.dirname(output_filename)
            if not os.path.isdir(output_dirname):
                os.makedirs(output_dirname)
            self.write_output(
                output_filename,
                output,
                file_format=self.output_format,
            )
        return output

    def read_input(self, filename):
        """Override to implement reading inputs from disk."""
        raise NotImplementedError()

    def cast_input(self, input):
        """Override to support accepting pipeline inputs of variouse types."""
        return input

    def read_output(self, filename):
        """Override to implement reading outputs from disk.

        Used for result cacheing.
        """
        raise NotImplementedError()

    @staticmethod
    def write_output(filename, output, file_format=None, **kwargs):
        """Override to implement reading inputs from disk."""
        raise NotImplementedError()


class SingleBase(_OneAndOne):
    """Base class for non-iterating tasks with at most one input and output.

    Inherits from :class:`TaskBase`.

    Tasks inheriting from this class should override `process` and optionally
    :meth:`setup`, :meth:`finish`, :meth:`read_input`, :meth:`write_output` and
    :meth:`cast_input`.  They should not override :meth:`next`.

    If the value of :attr:`input_root` is anything other than the string "None"
    then the input will be read (using :meth:`read_input`) from the file
    ``self.input_root + self.input_filename``.  If the input is specified both as
    a filename and as a product key in the pipeline configuration, an error
    will be raised upon initialization.


    If the value of :attr:`output_root` is anything other than the string
    "None" then the output will be written (using :meth:`write_output`) to the
    file ``self.output_root + self.output_filename``.

    Attributes
    ----------
    input_root : string
        Pipeline settable parameter giving the first part of the input path.
        If set to 'None' no input is read. Either it is assumed that no input
        is required or that input is recieved from the pipeline.
    input_filename : string
        Pipeline settable parameter giving the last part of input path. The
        full input path is ``self.input_root + self.input_filename``.
    output_root : strig
        Pipeline settable parameter giving the first part of the output path.
        If set to 'None' no output is written.
    output_filename : string
        Pipeline settable parameter giving the last part of output path. The
        full output path is ``self.output_root + self.output_filename``.
    """

    input_filename = config.Property(default="", proptype=str)
    output_filename = config.Property(default="", proptype=str)
    output_format = config.file_format()
    output_compression = config.Property(default=None, proptype=str)
    output_compression_opts = config.Property(default=None)

    def next(self, input=None):
        """Should not need to override."""
        # This should only be called once.
        try:
            if self.done:
                raise PipelineStopIteration()
        except AttributeError:
            self.done = True

        if input:
            input = self.cast_input(input)
        return self.read_process_write(input, self.input_filename, self.output_filename)


class IterBase(_OneAndOne):
    """Base class for iterating tasks with at most one input and one output.

    Tasks inheriting from this class should override :meth:`process` and
    optionally :meth:`setup`, :meth:`finish`, :meth:`read_input`,
    :meth:`write_output` and :meth:`cast_input`. They should not override
    :meth:`next`.

    If the value of :attr:`input_root` is anything other than the string "None"
    then the input will be read (using :meth:`read_input`) from the file
    ``self.input_root + self.file_middles[i] + self.input_ext``.  If the
    input is specified both as a filename and as a product key in the pipeline
    configuration, an error will be raised upon initialization.

    If the value of :attr:`output_root` is anything other than the string "None"
    then the output will be written (using :meth:`write_output`) to the file
    ``self.output_root + self.file_middles[i] + self.output_ext``.

    Attributes
    ----------
    iteration : int
        The current iteration of `process`/`next`.
    file_middles : list of strings
        The unique part of each file path.
    input_root : string
        Pipeline settable parameter giving the first part of the input path.
        If set to 'None' no input is read. Either it is assumed that no input
        is required or that input is recieved from the pipeline.
    input_ext : string
        Pipeline settable parameter giving the last part of input path. The
        full input path is ``self.input_root +
        self.file_middles[self.iteration] + self.input_ext``.
    output_root : strig
        Pipeline settable parameter giving the first part of the output path.
        If set to 'None' no output is written.
    output_ext : string
        Pipeline settable parameter giving the last part of output path. The
        full output path is ``self.output_root +
        self.file_middles[self.iteration] + self.output_ext``.
    """

    file_middles = config.Property(default=[], proptype=list)
    input_ext = config.Property(default="", proptype=str)
    output_ext = config.Property(default="", proptype=str)

    def __init__(self):
        super().__init__()
        self.iteration = 0

    def next(self, input=None):
        """Should not need to override."""
        # Sort out filenames.
        if self.iteration >= len(self.file_middles):
            if not self.input_root == "None":
                # We are iterating over input files and have run out.
                raise PipelineStopIteration()
            # Not iterating over input files, and unable to assign
            # filenames.
            input_filename = None
            output_filename = None
        else:
            # May or may not be iterating over input files, but able to assign
            # filenames.
            middle = self.file_middles[self.iteration]
            input_filename = middle + self.input_ext
            output_filename = middle + self.output_ext

        if input:
            input = self.cast_input(input)
        output = self.read_process_write(input, input_filename, output_filename)
        self.iteration += 1

        return output


class H5IOMixin:
    """Provides hdf5/zarr IO for pipeline tasks.

    As a mixin, this must be combined (using multiple inheritance) with a
    subclass of `TaskBase`, providing the full task API.

    Provides the methods `read_input`, `read_output` and `write_output` for
    hdf5 data.
    """

    # TODO, implement reading on disk (i.e. no copy to memory).
    # ondisk = config.Property(default=False, proptype=bool)

    @staticmethod
    def read_input(filename):
        """Method for reading hdf5 input."""
        from caput import memh5

        return memh5.MemGroup.from_hdf5(filename, mode="r")

    @staticmethod
    def read_output(filename):
        """Method for reading hdf5 output (from caches)."""
        # Replicate code from read_input in case read_input is overridden.
        from caput import memh5

        return memh5.MemGroup.from_hdf5(filename, mode="r")

    @staticmethod
    def write_output(filename, output, file_format=None, **kwargs):
        """Method for writing hdf5/zarr output.

        Parameters
        ----------
        filename : str
            File name
        output : memh5.Group, zarr.Group or h5py.Group
            `output` to be written. If this is a `h5py.Group` (which include `hdf5.File` objects)
            the buffer is flushed if `filename` points to the same file and a copy is made otherwise.
        file_format : fileformats.Zarr, fileformats.HDF5 or None
            File format to use. If this is not specified, the file format is guessed based on the type of
            `output` or the `filename`. If guessing is not successful, HDF5 is used.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        import h5py

        from caput import memh5

        file_format = fileformats.check_file_format(filename, file_format, output)

        try:
            import zarr
        except ImportError:
            if file_format == fileformats.Zarr:
                raise RuntimeError("Can't write to zarr file. Please install zarr.")

        # Ensure parent directory is present.
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                # It's possible the directory was created by another MPI task
                if not os.path.isdir(dirname):
                    raise e
        # Cases for `output` object type.
        if isinstance(output, memh5.MemGroup):
            # Already in memory.

            # Lock file
            with misc.lock_file(filename, comm=output.comm) as fn:
                output.to_file(fn, mode="w", file_format=file_format, **kwargs)
            return

        if isinstance(output, h5py.Group):
            if os.path.isfile(filename) and os.path.samefile(
                output.file.filename, filename
            ):
                # `output` already lives in this file.
                output.flush()

            else:
                # Copy to memory then to disk
                # XXX This can be made much more efficient using a direct copy.
                out_copy = memh5.MemGroup.from_hdf5(output)

                # Lock file as we write
                with misc.lock_file(filename, comm=out_copy.comm) as fn:
                    out_copy.to_hdf5(fn, mode="w")
        elif isinstance(output, zarr.Group):
            if os.path.isdir(filename) and os.path.samefile(
                output.store.path, filename
            ):
                pass
            else:
                logger.debug(f"Copying {output.store}:{output.path} to {filename}.")
                from . import mpiutil

                if mpiutil.rank == 0:
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        output.store,
                        zarr.DirectoryStore(filename),
                        source_path=output.path,
                    )
                logger.debug(
                    f"Copied {n_copied} items ({n_bytes_copied} bytes), skipped {n_skipped} items."
                )


class BasicContMixin:
    """Provides IO for BasicCont objects in pipeline tasks.

    As a mixin, this must be combined (using multiple inheritance) with a
    subclass of `TaskBase`, providing the full task API.

    Provides the methods `read_input`, `read_output` and `write_output` for
    BasicCont data which gets written to HDF5 files.
    """

    # TODO, implement reading on disk (i.e. no copy to memory).
    # ondisk = config.Property(default=False, proptype=bool)

    # Private setting for reading of inputs, should be overriden in sub class.
    _distributed = False
    _comm = None

    def read_input(self, filename):
        """Method for reading hdf5 input."""
        from caput import memh5

        return memh5.BasicCont.from_file(
            filename, distributed=self._distributed, comm=self._comm
        )

    def read_output(self, filename):
        """Method for reading hdf5 output (from caches)."""
        # Replicate code from read_input in case read_input is overridden.
        from caput import memh5

        return memh5.BasicCont.from_file(
            filename, distributed=self._distributed, comm=self._comm
        )

    @staticmethod
    def write_output(filename, output, file_format=None, **kwargs):
        """Method for writing output to disk.

        Parameters
        ----------
        filename : str
            File name.
        output : :class:`memh5.BasicCont`
            Data to be written.
        file_format : `fileformats.FileFormat`
            File format to use. Default `fileformats.HDF5`.
        **kwargs : dict
            Arbitrary keyword arguments.
        """
        from caput import memh5

        file_format = fileformats.check_file_format(filename, file_format, output)

        # Ensure parent directory is present.
        dirname = os.path.dirname(filename)
        if dirname != "" and not os.path.isdir(dirname):
            try:
                os.makedirs(dirname)
            except OSError as e:
                # It's possible the directory was created by another MPI task
                if not os.path.isdir(dirname):
                    raise e
        # Cases for `output` object type.
        if not isinstance(output, memh5.BasicCont):
            raise RuntimeError(
                "Object to write out is not an instance of memh5.BasicCont"
            )

        # Already in memory.
        output.save(filename, file_format=file_format, **kwargs)


class SingleH5Base(H5IOMixin, SingleBase):
    """Base class for tasks with hdf5 input and output.

    Inherits from :class:`H5IOMixin` and :class:`SingleBase`.
    """

    pass


class IterH5Base(H5IOMixin, IterBase):
    """Base class for iterating over hdf5 input and output.

    Inherits from :class:`H5IOMixin` and :class:`IterBase`.
    """

    pass


# Simple tasks for bridging to Python
# -----------------------------------


class Input(TaskBase):
    """Pass inputs into the pipeline from outside."""

    def __init__(self, inputs=None):
        super().__init__()
        self.inputs = inputs or []
        self._iter = None

    def next(self):
        """Pop and return the first element of inputs."""
        if self._iter is None:
            self._iter = iter(self.inputs)

        try:
            return next(self._iter)
        except StopIteration as e:
            raise PipelineStopIteration() from e


class Output(TaskBase):
    """Take outputs from the pipeline and place them in a list.

    To apply some processing to pipeline output (i.e. this tasks input), use the
    `callback` argument which will get passed the item. The return value of the
    callback is placed in the `outputs` attribute. Note that this need not be the
    input, so if pipeline output should be deleted to save memory you can simply
    return `None`.

    Parameters
    ----------
    callback : function, optional
        A function which can apply some processing to the pipeline output.
    """

    def __init__(self, callback=None):
        super().__init__()
        self.outputs = []
        self.callback = callback

    def next(self, in_):
        """Pop and return the first element of inputs."""
        if self.callback:
            in_ = self.callback(in_)

        self.outputs.append(in_)


# Internal Functions
# ------------------


def _format_product_keys(keys):
    """Formats the pipeline task product keys.

    In the pipeline config task list, the values of 'requires', 'in' and 'out'
    are keys representing data products.  This function gets that key from the
    task's entry of the task list, defaults to zero, and ensures it's formated
    as a sequence of strings.
    """
    if keys is None:
        return []

    # Turn into a sequence if only one key was provided.
    if not isinstance(keys, list):
        # Making this a tuple instead of a list is significant.  It only
        # impacts the 'out' product key and affects how return values are
        # unpacked.
        keys = (keys,)

    # Check that all the keys provided are strings.
    for key in keys:
        if not isinstance(key, str):
            msg = "Data product keys must be strings."
            raise config.CaputConfigError(msg)
    return keys


if __name__ == "__main__":
    import doctest

    doctest.testmod()
