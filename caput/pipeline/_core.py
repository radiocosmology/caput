"""Core pipeline task infrastructure."""

import importlib
import inspect
import logging
import queue
import re
import traceback
import warnings
from collections.abc import Generator
from copy import deepcopy

import yaml

from .. import config
from ..util import importtools, mpitools, objecttools

# Set the module logger.
logger = logging.getLogger(__name__)


__all__ = [
    "Manager",
    "PipelineRuntimeError",
    "PipelineStopIteration",
    "TaskBase",
    "local_tasks",
]


# Search this dictionary for tasks.
# Used for interactive sessions when task can't be specified by absolute path.
local_tasks = {}


# Exceptions
# ----------


class PipelineRuntimeError(RuntimeError):
    """Raised when there is a pipeline related error at runtime."""


class PipelineStopIteration(Exception):
    """Stop the iteration of `next()` in pipeline tasks.

    Pipeline tasks should raise this excetions in the `next()` method to stop
    the iteration of the task and to proceed to `finish()`.

    Note that if `next()` recieves input data as an argument, it is not
    required to ever raise this exception.  The pipeline will proceed to
    `finish()` once the input data has run out.
    """


class _PipelineMissingData(Exception):
    """Used for flow control when input data is yet to be produced."""


class _PipelineFinished(Exception):
    """Raised by tasks that have been completed."""


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
    interactive: bool, optional
        If True, the `.run()` method becomes a generator and stops on each iteration after
        selecting the next task to run. This allows a user to interact with the pipeline
        at each step and to probe the internal state of each task. This feature should be
        used as follows:
        - p = Manager()
        - p.interactive = True
        - runner = p.runner()
        - next(runner)
        Default is False.
    enable_breakpoints : bool, optional
        If True, task breakpoints are enabled. If a task requests a breakpoint, a call to
        `breakpoint()` is made every time the task is selected to be run. If `interactive`
        is True, this does nothing. Default is False.
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
    interactive = config.Property(proptype=bool, default=False)
    enable_breakpoints = config.Property(proptype=bool, default=False)
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
        from ..config import SafeLineLoader

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
        """Run the pipeline through to completion.

        Interactive mode will be ignored, but breakpoints are
        still respected.
        """
        runner = self.runner()

        while True:
            try:
                next(runner)
            except StopIteration:
                break

        # Pipeline is done
        logger.info("FIN")

    def runner(self) -> Generator:
        """Main driver for the pipeline.

        This function creates a generator object to initialize and run
        the pipeline. If `self.interactive` is True, or if breakpoint are
        set and enabled, the generator object will yield at the relevant
        pipeline stages. Otherwise, a single generator `next` call will run
        the pipeline to completion.

        Raises
        ------
        PipelineRuntimeError
            If a task stage returns the wrong number of outputs.

        Returns
        -------
        runner : Generator
            Generator object to run the pipeline.

        """
        from ..util.profiler import PSUtilProfiler

        # Log MPI information
        if mpitools._comm is not None:
            logger.debug(f"Running with {mpitools.size} MPI process(es)")
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

            if self.interactive:
                # Freezes the pipeline runner object in its current state
                yield
            elif self.enable_breakpoints and task.breakpoint:
                # Drop into PDB for this specific task
                breakpoint()

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
                    logger.debug(
                        f"Task {task!s} tried to pass key {key} "
                        "but no task was found to accept it."
                    )

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
            key=lambda i: (
                self.tasks[i]._pipeline_priority,
                self.tasks[i].base_priority,
            ),
            reverse=True,
        )[0]

        # Ensure that all ranks are running the same task.
        # This probably should never be needed with the current
        # priority selection. Effectively a no-op if no MPI
        self._task_idx = mpitools.bcast(new_index, root=0)

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
                if isinstance(t["out"], list | tuple):
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
                task_cls = importtools.import_class(task_path)
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


# Pipeline Task Base Class
# ------------------------


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
        about dynamic priority. Default is 0.
    breakpoint: bool
        If true, signals to the pipeline runner to make a call to `breakpoint` each
        time this task is run. This will drop the interpreter into pdb, allowing for
        interactive debugging of the current pipeline and task state. Default is False.
    """

    broadcast_inputs = config.Property(proptype=bool, default=False)
    limit_outputs = config.Property(proptype=int, default=None)
    base_priority = config.Property(proptype=int, default=0)
    breakpoint = config.Property(proptype=bool, default=False)

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
    def _pipeline_priority(self):
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
        return objecttools.total_size(self)

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
                # Let the user know if there's a chance that data has been
                # modified in-place
                for opt in out if isinstance(out, tuple | list) else (out,):
                    if opt in args:
                        logger.debug(
                            f"Task {self!s} may have modified dataset {opt!r} in-place. "
                            "If you encounter unexpected results, check that this is "
                            "the intended behaviour."
                        )

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
            # It's possible that the same key could be passed multiple times
            indices = (ii for ii, k in enumerate(self._requires_keys) if k == key)

            for ii in indices:
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
            indices = (ii for ii, k in enumerate(self._in_keys) if k == key)

            for ii in indices:
                logger.debug(f"{self!s} stowing data product with key {key} for `in`.")
                if self._in is None:
                    raise PipelineRuntimeError(
                        "Tried to queue 'in' data product, but `next()` already run."
                    )

                self._in[ii].put(product)

                result = True

        return result


if __name__ == "__main__":
    import doctest

    doctest.testmod()
