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
- :py:class:`Task`
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

A pipeline task is a subclass of :class:`Task` intended to perform some small,
modular piece analysis. The developer of the task must specify what input
parameters the task expects as well as code to perform the actual processing
for the task.

Input parameters are specified by adding class attributes whose values are
instances of :class:`config.Property`. For instance a task definition might begin
with

>>> from caput import config
>>> class SpamTask(Task):
...     eggs = config.Property(proptype=str)

This defines a new task named :class:`SpamTask` with a parameter named *eggs*, whose
type is a string.  The class attribute :attr:`SpamTask.eggs` will replaced with an
instance attribute when an instance of the task is initialized, with it's value
read from the pipeline configuration YAML file (see next section).

The actual work for the task is specified by over-ridding any of the
:meth:`~Task.setup`, :meth:`~Task.next` or
:meth:`~Task.finish` methods (:meth:`~Task.__init__` may also be
implemented`).  These are executed in order, with :meth:`~TaskBask.next`
possibly being executed many times.  Iteration of :meth:`next` is halted by
raising a :exc:`PipelineStopIteration`.  Here is a example of a somewhat
trivial but fully implemented task:

>>> class PrintEggs(Task):
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
...             raise exceptions.PipelineStopIteration
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

>>> class GetEggs(Task):
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
...             raise exceptions.PipelineStopIteration
...         egg = self.eggs[self.i]
...         self.i += 1
...         return egg
...
...     def finish(self):
...         print("Finished GetEggs.")

>>> class CookEggs(Task):
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

When the above pipeline is executed, it produces the following output.

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

The order in which tasks are executed is determined by a priority system
using the following criteria, in decreasing importance:

1. Task must be available to execute some step.
2. Task priority. This is set by two factors:

   * Dynamic priority: tasks which have a higher net consumption
     (inputs consumed minus outputs created).
   * Base priority: user-configurable base priority is added to
     the dynamic priority.

3. Pipeline configuration order.

If no tasks are available to run, the following execution rules are applied:

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

The example output order is because `PrintEggs` will always have higher priority
than `GetEggs`, so it will run to completion _before_ `GetEggs` starts generating
anything. Only once `PrintEggs` is done will the other tasks run. Even though
`CookEggs` has the highest priority, it cannot do anything without `GetEggs` running
first.

If the above rules seem somewhat opaque, consider the following example which
illustrates these rules in a pipeline with a slightly more non-trivial flow.

>>> class DoNothing(Task):
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
configuration attributes directly (or call :meth:`~Task.read_config` on an
appropriate dictionary); and then added to the pipeline using the
:meth:`~Manager.add_task` to add the instance and specify the queues it connects to.

To inject products into the pipeline, use the :class:`~Input` and supply it an
iterator as an argument. Each item will be fed into the pipeline one by one. To take
outputs from the pipeline, simply use the :class:`~Output` task. By default this
simply saves everything it receives into a list (which can be accessed via the task's
`outputs` attribute, e.g. with `save_output.outputs` after running the example below),
but it can be given a callback function to apply processing to each argument in turn.

>>> m = Manager()
>>> m.add_task(extensions.Input(["platypus", "dinosaur"]), out="key1")
>>> cook = CookEggs()
>>> cook.style = "coddled"
>>> m.add_task(cook, in_="key1")
>>> save_output = extensions.Output()
>>> m.add_task(save_output, in_="key1")
>>> print_output = extensions.Output(lambda x: print("I love %s eggs!" % x))
>>> m.add_task(print_output, in_="key1")
>>> m.run()
Setting up CookEggs.
Cooking coddled platypus eggs.
I love platypus eggs!
Cooking coddled dinosaur eggs.
I love dinosaur eggs!
Finished CookEggs.

Advanced Tasks
--------------

Several subclasses of :class:`Task` provide advanced functionality for tasks that
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

from ._pipeline import (
    Manager as Manager,
    Task as Task,
    local_tasks as local_tasks,
)

from . import (
    exceptions as exceptions,
    extensions as extensions,
    tasklib as tasklib,
)
