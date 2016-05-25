"""
Data Analysis and Simulation Pipeline.

A data analysis pipeline is completely specified by a YAML file that specifies
both what tasks are to be run and the parameters that go to those tasks.
Included in this package are base classes for simplifying the construction of
data analysis tasks, as well as the pipeline manager which executes them.

Pipelines are most easily executed using the script in ``caput_pipeline.py`,
which ships with :mod:`caput`.

Flow control classes
====================

.. autosummary::
   :toctree: generated/

   Manager
   PipelineConfigError
   PipelineRuntimeError
   PipelineStopIteration


Task base classes
=================

.. autosummary::
    :toctree: generated/

   TaskBase
   SingleBase
   IterBase
   H5IOMixin
   BasicContMixin
   SingleH5Base
   IterH5Base

Examples
========

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
...         self.i = 0
...
...     def setup(self):
...         print "Setting up PrintEggs."
...
...     def next(self):
...         if self.i >= len(self.eggs):
...             raise PipelineStopIteration()
...         print "Spam and %s eggs." % self.eggs[self.i]
...         self.i += 1
...
...     def finish(self):
...         print "Finished PrintEggs."

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
...         self.i = 0
...
...     def setup(self):
...         print "Setting up GetEggs."
...
...     def next(self):
...         if self.i >= len(self.eggs):
...             raise PipelineStopIteration()
...         egg = self.eggs[self.i]
...         self.i += 1
...         return egg
...
...     def finish(self):
...         print "Finished GetEggs."

>>> class CookEggs(TaskBase):
...
...     style = config.Property(proptype=str)
...
...     def setup(self):
...         print "Setting up CookEggs."
...
...     def next(self, egg):
...         print "Cooking %s %s eggs." % (self.style, egg)
...
...     def finish(self):
...         print "Finished CookEggs."

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
in
    A 'pipeline product key' or list of keys representing values to be passed
    as arguments to :meth:`next`.

The sections other than 'pipeline' in the configuration contain the parameter
for the various tasks, as specified be the 'params' keys.

Execution Order
---------------

When the above pipeline is executed is produces the following output.

>>> local_tasks.update(globals())  # Required for interactive sessions.
>>> Manager.from_yaml_str(spam_config).run()
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

The rules for execution order are as follows:

1. One of the methods `setup()`, `next()` or `finish()`, as appropriate, will
   be executed from each task, in order.
2. If the task method is missing its input, as specified by the 'requires' or 'in'
   keys, restart at the beginning of the `tasks` list.
3. If the input to `next()` is missing and the task is at the beginning of the
   list there will be no opportunity to generate this input. Stop iterating
   `next()` and proceed to `finish()`.
4. Once a task has executed `finish()`, remove it from the list.
5. Once a method from the last member of the `tasks` list is executed, restart
   at the beginning of the list.

If the above rules seem somewhat opaque, consider the following example which
illustrates these rules in a pipeline with a slightly more non-trivial flow.

>>> class DoNothing(TaskBase):
...
...     def setup(self):
...         print "Setting up DoNothing."
...
...     def next(self, input):
...         print "DoNothing next."
...
...     def finish(self):
...         print "Finished DoNothing."

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

>>> Manager.from_yaml_str(new_spam_config).run()
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

Notice that :meth:`DoNothing.next` is nerver called, since the pipeline never
generates its input, 'non_existent_data_product'.  Once everything before
:class:`DoNothing` has been executed the pipeline notices that there is no
opertunity for 'non_existent_data_product' to be generated and forces
`DoNothing` to proceed to :meth:`finish`. This also unblocks :class:`PrintEggs`
allowing it to proceed normally.

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


import sys
import inspect
import Queue
import logging
import os
from os import path
import warnings

import yaml

import config


# Set the module logger.
logger = logging.getLogger(__name__)


# Search this dictionary for tasks.
# Used for interactive sessions when task can't be specified by absolute path.
local_tasks = {}


# Exceptions
# ----------

class PipelineConfigError(Exception):
    """Raised when there is an error setting up a pipeline."""

    pass


class PipelineRuntimeError(Exception):
    """Raised when there is a pipeline related error at runtime."""

    pass


class PipelineStopIteration(Exception):
    """This stops the iteration of `next()` in pipeline tasks.

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

class Manager(config.Reader):
    """Pipeline manager for setting up and running pipeline tasks.

    The manager is in charge of initializing all pipeline tasks, setting them
    up by providing the appropriate parameters, then executing the methods of
    the each task in the appropriate order. It also handles intermediate data
    products and ensuring that the correct products are passed between tasks.

    """

    logging = config.Property(default='warning', proptype=str)
    multiprocessing = config.Property(default=1, proptype=int)
    cluster = config.Property(default={}, proptype=dict)
    tasks = config.Property(default=[], proptype=list)

    @classmethod
    def from_yaml_file(cls, file_name):
        """Initialize the pipeline from a YAML configuration file.

        Parameters
        ----------
        file_name: string
            Path to YAML pipeline configuration file.

        Returns
        -------
        self: Pipeline object
        """

        with open(file_name) as f:
            yaml_doc = f.read()
        return cls.from_yaml_str(yaml_doc)

    @classmethod
    def from_yaml_str(cls, yaml_doc):
        """Initialize the pipeline from a YAML configuration string.

        Parameters
        ----------
        yaml_doc: string
            Yaml configuration document.

        Returns
        -------
        self: Pipeline object
        """

        yaml_params = yaml.load(yaml_doc)
        self = cls.from_config(yaml_params['pipeline'])
        self.all_params = yaml_params
        return self


    def run(self):
        """Main driver method for the pipeline.

        This function initializes all pipeline tasks and runs the pipeline
        through to completion.

        """

        # Set logging level.
        numeric_level = getattr(logging, self.logging.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid logging level: %s' % self.logging)
        logging.basicConfig(level=numeric_level)
        # Initialize all tasks.
        pipeline_tasks = []
        for ii, task_spec in enumerate(self.tasks):
            try:
                task = self._setup_task(task_spec)
            except PipelineConfigError as e:
                msg = "Setting up task %d caused an error - " % ii
                msg += str(e)
                new_e = PipelineConfigError(msg)
                # This preserves the traceback.
                raise new_e.__class__, new_e, sys.exc_info()[2]
            pipeline_tasks.append(task)
            logger.debug("Added %s to task list." % task.__class__.__name__)
        # Run the pipeline.
        while pipeline_tasks:
            for task in list(pipeline_tasks):  # Copy list so we can alter it.
                # These lines control the flow of the pipeline.
                try:
                    out = task._pipeline_next()
                except _PipelineMissingData:
                    if pipeline_tasks.index(task) == 0:
                        msg = ("%s missing input data and is at beginning of"
                               " task list. Advancing state."
                               % task.__class__.__name__)
                        logger.debug(msg)
                        task._pipeline_advance_state()
                    break
                except _PipelineFinished:
                    pipeline_tasks.remove(task)
                    continue
                # Now pass the output data products to any task that needs
                # them.
                out_keys = task._out_keys
                if out is None:     # This iteration supplied no output.
                    continue
                elif len(out_keys) == 0:    # Output not handled by pipeline.
                    continue
                elif len(out_keys) == 1:
                    if type(out_keys) is tuple:
                        # In config file, written as `out: out_key`. No
                        # unpacking if `out` is a length 1 sequence.
                        out = (out,)
                    else:   # `out_keys` is a list.
                        # In config file, written as `out: [out_key,]`.
                        # `out` must be a length 1 sequence.
                        pass
                elif len(out_keys) != len(out):
                    msg = ('Found unexpected number of outputs in %s (got %i expected %i)' %
                           (task.__class__.__name__, len(out), len(out_keys)))
                    raise PipelineRuntimeError(msg)
                keys = str(out_keys)
                msg = "%s produced output data product with keys %s."
                msg = msg % (task.__class__.__name__, keys)
                logger.debug(msg)
                for receiving_task in pipeline_tasks:
                    receiving_task._pipeline_inspect_queue_product(out_keys, out)


    def _setup_task(self, task_spec):
        """Set up a pipeline task from the spec given in the tasks list."""

        # Check that only the expected keys are in the task spec.
        for key in task_spec.keys():
            if not key in ['type', 'params', 'requires', 'in', 'out']:
                msg = "Task got an unexpected key '%s' in 'tasks' list." % key
                raise PipelineConfigError(msg)
        # 'type' is a required key.
        try:
            task_path = task_spec['type']
        except KeyError:
            msg = "'type' not specified for task."
            raise PipelineConfigError(msg)
        if task_path in local_tasks.keys():
            task_cls = local_tasks[task_path]
        else:
            try:
                task_cls = _import_class(task_path)
            except Exception as e:
                e_str = e.__class__.__name__
                e_str += ': ' + str(e)
                msg = "Loading task '%s' caused error - " % task_path
                msg += e_str
                raise PipelineConfigError(msg)

        # Get the parameters and initialize the class.
        params = {}

        if 'params' in task_spec:

            # If params is a dict, assume params are inline
            if isinstance(task_spec['params'], dict):
                params.update(task_spec['params'])
            else:  # Otherwise assume it's a list of keys

                param_keys = task_spec['params']

                # Must be a list of keys, convert if only one key was specified.
                if not isinstance(param_keys, list):
                    param_keys = [param_keys]

                # Locate param sections, and add to dict
                try:
                    for param_key in param_keys:
                        params.update(self.all_params[param_key])
                except KeyError:
                    msg = "Parameter group %s not found in config." % param_key
                    raise PipelineConfigError(msg)

        # Setup task
        task = task_cls._pipeline_from_config(params, task_spec)
        # Set up data product keys.
        return task


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
    cacheable
    embarrassingly_parallelizable

    Methods
    -------
    __init__
    setup
    next
    finish

    """

    # Overridable Attributes
    # -----------------------

    def __init__(self):
        """Initialize pipeline task.

        May be overridden with no arguments.  Will be called after any
        `config.Property` attributes are set and after 'input' and 'requires'
        keys are set up.

        """

        pass

    def setup(self, requires=None):
        """First analysis stage of pipeline task.

        May be overridden with any number of positional only arguments
        (defaults are allowed).  Pipeline data-products will be passed as
        specified by `requires` keys in the pipeline setup.

        Any return values will be treated as pipeline data-products as
        specified by the `out` keys in the pipeline setup.

        """

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

    @classmethod
    def _pipeline_from_config(cls, config, task_spec):
        self = cls.__new__(cls)
        self.read_config(config)
        self._pipeline_setup(task_spec)
        self.__init__()
        return self

    def _pipeline_setup(self, task_spec):
        """Setup the 'requires', 'in' and 'out' keys for this task."""

        # Put pipeline in state such that `setup` is the next stage called.
        self._pipeline_advance_state()
        # Parse the task spec.
        requires = _format_product_keys(task_spec, 'requires')
        in_ = _format_product_keys(task_spec, 'in')
        out = _format_product_keys(task_spec, 'out')
        # Inspect the `setup` method to see how many arguments it takes.
        setup_argspec = inspect.getargspec(self.setup)
        # Make sure it matches `requires` keys list specified in config.
        n_requires = len(requires)
        try:
            len_defaults = len(setup_argspec.defaults)
        except TypeError:    # defaults is None
            len_defaults = 0
        min_req = len(setup_argspec.args) - len_defaults - 1
        if n_requires < min_req:
            msg = ("Didn't get enough 'requires' keys. Expected at least"
                   " %d and only got %d." % (min_req, n_requires))
            raise PipelineConfigError(msg)
        if (n_requires > len(setup_argspec.args) - 1
            and setup_argspec.varargs is None):
            msg = ("Got too many 'requires' keys. Expected at most %d and"
                   " got %d." % (len(setup_argspec.args) - 1, n_requires))
            raise PipelineConfigError(msg)
        # Inspect the `next` method to see how many arguments it takes.
        next_argspec = inspect.getargspec(self.next)
        # Make sure it matches `in` keys list specified in config.
        n_in = len(in_)
        try:
            len_defaults = len(next_argspec.defaults)
        except TypeError:    # defaults is None
            len_defaults = 0
        min_in = len(next_argspec.args) - len_defaults - 1
        if n_in < min_in:
            msg = ("Didn't get enough 'in' keys. Expected at least"
                   " %d and only got %d." % (min_in, n_in))
            raise PipelineConfigError(msg)
        if (n_in > len(next_argspec.args) - 1
            and next_argspec.varargs is None):
            msg = ("Got too many 'in' keys. Expected at most %d and"
                   " got %d." % (len(next_argspec.args) - 1, n_in))
            raise PipelineConfigError(msg)
        # Now that all data product keys have been verified to be valid, store
        # them on the instance.
        self._requires_keys = requires
        self._requires = [None] * n_requires
        self._in_keys = in_
        self._in = [Queue.Queue() for i in range(n_in)]
        self._out_keys = out

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
            # Delete inputs to free memory.
            self._requires = None
            self._pipeline_state = "next"
        elif self._pipeline_state == "next":
            # Make sure input queues are empty then delete them so no more data
            # can be queued.
            for in_, in_key in zip(self._in, self._in_keys):
                if not in_.empty():
                    # XXX Clean up.
                    print "Something left: %i" % in_.qsize()

                    msg = "Task finished %s iterating `next()` but input queue \'%s\' isn't empty." % (self.__class__.__name__, in_key)
                    warnings.warn(msg)

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
            else:
                msg = "Task %s calling 'setup()'." % self.__class__.__name__
                logger.debug(msg)
                out = self.setup(*tuple(self._requires))
                self._pipeline_advance_state()
                return out
        elif self._pipeline_state == "next":
            # Check if we have all the required input data.
            for in_ in self._in:
                if in_.empty():
                    raise _PipelineMissingData()
            else:
                # Get the next set of data to be run.
                args = ()
                for in_ in self._in:
                    args += (in_.get(),)
                try:
                    msg = "Task %s calling 'next()'." % self.__class__.__name__
                    logger.debug(msg)
                    out = self.next(*args)
                    return out
                except PipelineStopIteration:
                    # Finished iterating `next()`.
                    self._pipeline_advance_state()
        elif self._pipeline_state == "finish":
            msg = "Task %s calling 'finish()'." % self.__class__.__name__
            logger.debug(msg)
            out = self.finish()
            self._pipeline_advance_state()
            return out
        elif self._pipeline_state == "raise":
            raise _PipelineFinished()
        else:
            raise PipelineRuntimeError()

    def _pipeline_inspect_queue_product(self, keys, products):
        """Inspect data products and queue them as inputs if applicable.

        Compare a list of data products keys to the keys expected by this task
        as inputs to `setup()` ('requires') and `next()` ('in').  If there is a
        match, store the corresponding data product to be used in the next
        invocation of these methods.

        """

        n_keys = len(keys)
        for ii in range(n_keys):
            key = keys[ii]
            product = products[ii]
            for jj, requires_key in enumerate(self._requires_keys):
                if requires_key == key:
                    # Make sure that `setup()` hasn't already been run or this
                    # data product already set.
                    msg = "%s stowing data product with key %s for 'requires'."
                    msg = msg % (self.__class__.__name__, key)
                    logger.debug(msg)
                    if self._requires is None:
                        msg = ("Tried to set 'requires' data product, but"
                               "`setup()` already run.")
                        raise PipelineRuntimeError(msg)
                    if not self._requires[jj] is None:
                        msg = "'requires' data product set more than once."
                        raise PipelineRuntimeError(msg)
                    else:
                        # Accept the data product and store for later use.
                        self._requires[jj] = product
            for jj, in_key in enumerate(self._in_keys):
                if in_key == key:
                    msg = "%s queue data product with key %s for 'in'."
                    msg = msg % (self.__class__.__name__, key)
                    logger.debug(msg)
                    # Check that task is still accepting inputs.
                    if self._in is None:
                        msg = ("Tried to queue 'requires' data product, but"
                               "`next()` iteration already completed.")
                        raise PipelineRuntimeError(msg)
                    else:
                        # Accept the data product and store for later use.
                        self._in[jj].put(product)


class _OneAndOne(TaskBase):
    """Base class for tasks that have (at most) one input and one output

    This is not a user base class and simply holds code that is common to
    `SingleBase` and `IterBase`.

    """

    input_root = config.Property(default='None', proptype=str)
    output_root = config.Property(default='None', proptype=str)

    def process(self, input):
        """Override this method with your data processing task.
        """

        output = input
        return output

    def __init__(self):
        """Checks inputs and outputs and stuff."""

        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = inspect.getargspec(self.process)
        n_args = len(pro_argspec.args) - 1
        if n_args  > 1:
            msg = ("`process` method takes more than 1 argument, which is not"
                   " allowed.")
            raise PipelineConfigError(msg)
        if pro_argspec.varargs or pro_argspec.keywords or pro_argspec.defaults:
            msg = ("`process` method may not have variable length or optional"
                   " arguments.")
            raise PipelineConfigError(msg)
        if n_args == 0:
            self._no_input = True
        else:
            self._no_input = False

        # Make sure we know where to get the data from.
        if self.input_root == "None":
            if len(self._in) != n_args:
                msg = ("No data to iterate over. 'input_root' is 'None' and"
                       " there are no 'in' keys.")
                raise PipelineConfigError(msg)
        else:
            if len(self._in) != 0:
                msg = ("For data input, supplied both a file path and an 'in'"
                       " key.  If not reading to disk, set 'input_root' to"
                       " 'None'.")
                raise PipelineConfigError(msg)
            if n_args != 1:
                msg = ("Reading input from disk but `process` method takes no"
                       " arguments.")

    def read_process_write(self, input, input_filename, output_filename):
        """Reads input, executes any processing and writes output."""

        # Read input if needed.
        if input is None and not self._no_input:
            if input_filename is None:
                raise RuntimeError('No file to read from.')
            input_filename = self.input_root + input_filename
            input_filename = path.expanduser(input_filename)
            logger.info("%s reading data from file %s." %
                        (self.__class__.__name__, input_filename))
            input = self.read_input(input_filename)
        # Analyse.
        if self._no_input:
            if not input is None:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set.")
            output = self.process()
        else:
            output = self.process(input)
        # Write output if needed.
        if self.output_root != 'None' and not output is None:
            if output_filename is None:
                raise RuntimeError('No file to write to.')
            output_filename = self.output_root + output_filename
            output_filename = path.expanduser(output_filename)
            logger.info("%s writing data to file %s." %
                        (self.__class__.__name__, output_filename))
            output_dirname = os.path.dirname(output_filename)
            if not os.path.isdir(output_dirname):
                os.makedirs(output_dirname)
            self.write_output(output_filename, output)
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

    def write_output(self, filename, output):
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

    Methods
    -------
    next
    setup
    process
    finish
    read_input
    cast_input
    write_output

    """

    input_filename = config.Property(default='', proptype=str)
    output_filename = config.Property(default='', proptype=str)


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
        return self.read_process_write(input, self.input_filename,
                                       self.output_filename)


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


    Methods
    -------
    next
    setup
    process
    finish
    read_input
    cast_input
    write_output

    """

    file_middles = config.Property(default=[], proptype=list)
    input_ext = config.Property(default='', proptype=str)
    output_ext = config.Property(default='', proptype=str)

    def __init__(self):
        _OneAndOne.__init__(self)
        self.iteration = 0

    def next(self, input=None):
        """Should not need to override."""

        # Sort out filenames.
        if self.iteration >= len(self.file_middles):
            if not self.input_root == 'None':
                # We are iterating over input files and have run out.
                raise PipelineStopIteration()
            else:
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
        output = self.read_process_write(input, input_filename,
                                       output_filename)
        self.iteration += 1
        return output


class H5IOMixin(object):
    """Provides hdf5 IO for pipeline tasks.

    As a mixin, this must be combined (using multiple inheritance) with a
    subclass of `TaskBase`, providing the full task API.

    Provides the methods `read_input`, `read_output` and `write_output` for
    hdf5 data.

    """

    # TODO, implement reading on disk (i.e. no copy to memory).
    #ondisk = config.Property(default=False, proptype=bool)

    def read_input(self, filename):
        """Method for reading hdf5 input."""

        from caput import memh5
        return memh5.MemGroup.from_hdf5(filename, mode='r')

    def read_output(self, filename):
        """Method for reading hdf5 output (from caches)."""

        # Replicate code from read_input in case read_input is overridden.
        from caput import memh5
        return memh5.MemGroup.from_hdf5(filename, mode='r')

    def write_output(self, filename, output):
        """Method for writing hdf5 output.

        `output` to be written must be either a `memh5.MemGroup` or an
        `h5py.Group` (which include `hdf5.File` objects). In the latter case
        the buffer is flushed if `filename` points to the same file and a copy
        is made otherwise.

        """

        from caput import memh5
        import h5py

        # Ensure parent directory is present.
        dirname = path.dirname(filename)
        if not path.isdir(dirname):
            os.makedirs(dirname)
        # Cases for `output` object type.
        if isinstance(output, memh5.MemGroup):
            # Already in memory.
            output.to_hdf5(filename, mode='w')
        elif isinstance(output, h5py.Group):
            if path.isfile(filename) and path.samefile(output.file.filename,
                                                       filename):
                # `output` already lives in this file.
                output.flush()
            else:
                # Copy to memory then to disk
                # XXX This can be made much more efficient using a direct copy.
                out_copy = memh5.MemGroup.from_hdf5(output)
                out_copy.to_hdf5(filename, mode='w')


class BasicContMixin(object):
    """Provides IO for BasicCont objects in pipeline tasks.

    As a mixin, this must be combined (using multiple inheritance) with a
    subclass of `TaskBase`, providing the full task API.

    Provides the methods `read_input`, `read_output` and `write_output` for
    BasicCont data which gets written to HDF5 files.

    """

    # TODO, implement reading on disk (i.e. no copy to memory).
    #ondisk = config.Property(default=False, proptype=bool)

    # Private setting for reading of inputs, should be overriden in sub class.
    _distributed = False
    _comm = None

    def read_input(self, filename):
        """Method for reading hdf5 input."""

        from caput import memh5
        return memh5.BasicCont.from_file(filename, distributed=self._distributed, comm=self._comm)

    def read_output(self, filename):
        """Method for reading hdf5 output (from caches)."""

        # Replicate code from read_input in case read_input is overridden.
        from caput import memh5
        return memh5.BasicCont.from_file(filename, distributed=self._distributed, comm=self._comm)

    def write_output(self, filename, output):
        """Method for writing hdf5 output.

        `output` to be written must be either a :class:`memh5.BasicCont` object.
        """

        from caput import memh5
        import h5py

        # Ensure parent directory is present.
        dirname = path.dirname(filename)
        if dirname != '' and not path.isdir(dirname):
            os.makedirs(dirname)
        # Cases for `output` object type.
        if not isinstance(output, memh5.BasicCont):
            raise RuntimeError('Object to write out is not an instance of memh5.BasicCont')

        # Already in memory.
        output.save(filename)


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


# Internal Functions
# ------------------

def _format_product_keys(spec, name):
    """Formats the pipeline task product keys.

    In the pipeline config task list, the values of 'requires', 'in' and 'out'
    are keys representing data products.  This function gets that key from the
    task's entry of the task list, defaults to zero, and ensures it's formated
    as a sequence of strings.

    """

    try:
        out_prod = spec[name]
    except KeyError:
        # Default to empty if not present.
        return []
    # Turn into a sequence if only one key was provided.
    if not type(out_prod) is list:
        # Making this a tuple instead of a list is significant.  It only
        # impacts the 'out' product key and affects how return values are
        # unpacked.
        out_prod = (out_prod,)
    # Check that all the keys provided are strings.
    for prod in out_prod:
        if not isinstance(prod, basestring):
            msg = "Data product keys must be strings."
            raise PipelineConfigError(msg)
    return out_prod


def _import_class(class_path):
    """Import class dynamically from a string."""
    path_split = class_path.split('.')
    module_path = '.'.join(path_split[:-1])
    class_name = path_split[-1]
    if module_path:
        m = __import__(module_path)
        for comp in path_split[1:]:
            m = getattr(m, comp)
        task_cls = m
    else:
        task_cls = globals()[class_name]
    return task_cls


if __name__ == "__main__":
    import doctest
    doctest.testmod()
