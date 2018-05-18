# A short script designed as a test playground for the new python-based
#  caput-pipeline implementation
from caput import pipeline
from caput import config
import logging
import sys
import inspect
import Queue
import os
from os import path
import warnings



logger = logging.getLogger(__name__)
local_tasks = {}


class PythonManager(config.Reader):
    """Pipeline manager for setting up and running pipeline tasks.

    This WIP version will be all python, no YAML.

    The manager is in charge of initializing all pipeline tasks, setting them
    up by providing the appropriate parameters, then executing the methods of
    the each task in the appropriate order. It also handles intermediate data
    products and ensuring that the correct products are passed between tasks.

    """

    logging = config.Property(default='warning', proptype=str)
    multiprocessing = config.Property(default=1, proptype=int)
    cluster = config.Property(default={}, proptype=dict)
    # tasks = config.Property(default=[], proptype=list)
    # task_specs = config.Property(default=[], proptype=list)
    tasks = []
    task_specs = []

    def add_task_purelist(self, task, requires, input_, output):
        self.task_specs.append([requires, input_, output])
        single_task_compilation = [task, requires, input_, output]
        self.tasks.append(single_task_compilation)
        return self.tasks

    def add_task(self, task, requires, input_, output):
        single_task_spec = {"type": task, "requires": requires, "in": input_, "out": output}
        self.tasks.append(single_task_spec)
        return self.tasks

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
        #print "task enumerate is ", enumerate(self.tasks)
        for ii, task_spec in enumerate(self.tasks):
            #print "task_spec is ", task_spec
            try:
                task = self._setup_task_python(task_spec)
            except pipeline.PipelineConfigError as e:
                msg = "Setting up task %d caused an error - " % ii
                msg += str(e)
                new_e = pipeline.PipelineConfigError(msg)
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
                except pipeline._PipelineMissingData:
                    if pipeline_tasks.index(task) == 0:
                        msg = ("%s missing input data and is at beginning of"
                               " task list. Advancing state."
                               % task.__class__.__name__)
                        logging.logger.debug(msg)
                        task._pipeline_advance_state()
                    break
                except pipeline._PipelineFinished:
                    pipeline_tasks.remove(task)
                    continue
                # Now pass the output data products to any task that needs
                # them.
                out_keys = task._out_keys
                print out_keys
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
                    print "Out_keys are ", out_keys[:]
                    print "Out is ", out[:]
                    msg = ('Found unexpected number of outputs in %s (got %i expected %i)' %
                           (task.__class__.__name__, len(out), len(out_keys)))
                    raise pipeline.PipelineRuntimeError(msg)
                keys = str(out_keys)
                msg = "%s produced output data product with keys %s."
                msg = msg % (task.__class__.__name__, keys)
                logging.logger.debug(msg)
                for receiving_task in pipeline_tasks:
                    receiving_task._pipeline_inspect_queue_product(out_keys, out)

    def _setup_task_python(self, task_spec):
        """Set up a pipeline task from the spec given in the tasks list."""

        # Setup task
        for key in task_spec.keys():
            if not key in ['type', 'params', 'requires', 'in', 'out']:
                msg = "Task got an unexpected key '%s' in 'tasks' list." % key
                raise pipeline.PipelineConfigError(msg)
        # 'type' is a required key.
        try:
            task_path = task_spec['type']
            task_cls = task_path
        except KeyError:
            msg = "'type' not specified for task."
            raise pipeline.PipelineConfigError(msg)
        #if task_path in local_tasks.keys():
        #    task_cls = local_tasks[task_path]
        #    print "Task_cls is local and is ", task_cls
        #else:
        #    try:
        #        task_cls = _import_class(task_path)
        #        print "Task_cls was added and is ", task_cls
        #    except Exception as e:
        #        e_str = e.__class__.__name__
        #        e_str += ': ' + str(e)
        #        msg = "Loading task '%s' caused error - " % task_path
        #        msg += e_str
        #        raise pipeline.PipelineConfigError(msg)
        # Set up data product keys.
        #task = task_cls._pipeline_from_config(task_spec)
        task = task_cls._pipeline_setup(task_spec)
        return task


class ModifiedTaskBase(pipeline.TaskBase):

    def _pipeline_setup(self, task_spec):
        """Setup the 'requires', 'in' and 'out' keys for this task."""
        # Put pipeline in state such that `setup` is the next stage called.
        self._pipeline_advance_state()
        # Parse the task spec.
        requires = task_spec['requires']
        in_ = task_spec['in']
        out = task_spec['out']
        # Inspect the `setup` method to see how many arguments it takes.
        setup_argspec = inspect.getargspec(self.setup)
        # Make sure it matches `requires` keys list specified in config.
        try:
            n_requires = len(requires)
        except TypeError:
            n_requires = 0
        try:
            len_defaults = len(setup_argspec.defaults)
        except TypeError:    # defaults is None
            len_defaults = 0
        min_req = len(setup_argspec.args) - len_defaults - 1
        if n_requires < min_req:
            msg = ("Didn't get enough 'requires' keys. Expected at least"
                   " %d and only got %d." % (min_req, n_requires))
            raise pipeline.PipelineConfigError(msg)
        if (n_requires > len(setup_argspec.args) - 1
            and setup_argspec.varargs is None):
            msg = ("Got too many 'requires' keys. Expected at most %d and"
                   " got %d." % (len(setup_argspec.args) - 1, n_requires))
            raise pipeline.PipelineConfigError(msg)
        # Inspect the `next` method to see how many arguments it takes.
        next_argspec = inspect.getargspec(self.next)
        # Make sure it matches `in` keys list specified in config.
        try:
            n_in = len(in_)
        except TypeError:
            n_in = 0
        try:
            len_defaults = len(next_argspec.defaults)
        except TypeError:    # defaults is None
            len_defaults = 0
        min_in = len(next_argspec.args) - len_defaults - 1
        if n_in < min_in:
            msg = ("Didn't get enough 'in' keys. Expected at least"
                   " %d and only got %d." % (min_in, n_in))
            raise pipeline.PipelineConfigError(msg)
        if (n_in > len(next_argspec.args) - 1
            and next_argspec.varargs is None):
            msg = ("Got too many 'in' keys. Expected at most %d and"
                   " got %d." % (len(next_argspec.args) - 1, n_in))
            raise pipeline.PipelineConfigError(msg)
        # Now that all data product keys have been verified to be valid, store
        # them on the instance.
        self._requires_keys = requires
        self._requires = [None] * n_requires
        self._in_keys = in_
        self._in = [Queue.Queue() for i in range(n_in)]
        self._out_keys = out
        print "Self._out_keys is ", self._out_keys
        return self


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
                    raise pipeline._PipelineMissingData()
            else:
                msg = "Task %s calling 'setup()'." % self.__class__.__name__
                logger.debug(msg)
                out = self.setup(*tuple(self._requires))
                self._pipeline_advance_state()
                print "Out in setup is ", out
                return out
        elif self._pipeline_state == "next":
            # Check if we have all the required input data.
            for in_ in self._in:
                if in_.empty():
                    raise pipeline._PipelineMissingData()
            else:
                # Get the next set of data to be run.
                args = ()
                for in_ in self._in:
                    args += (in_.get(),)
                try:
                    msg = "Task %s calling 'next()'." % self.__class__.__name__
                    logger.debug(msg)
                    out = self.next(*args)
                    print "Out in next is ", self.next(*args)
                    return out
                except pipeline.PipelineStopIteration:
                    # Finished iterating `next()`.
                    self._pipeline_advance_state()
        elif self._pipeline_state == "finish":
            msg = "Task %s calling 'finish()'." % self.__class__.__name__
            logger.debug(msg)
            out = self.finish()
            self._pipeline_advance_state()
            return out
        elif self._pipeline_state == "raise":
            raise pipeline._PipelineFinished()
        else:
            raise pipeline.PipelineRuntimeError()


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
            #for in_, in_key in zip(self._in, self._in_keys):
            if not self._in==[]:
                # XXX Clean up.
                print "Something left: %i" % self._in.qsize()

                msg = "Task finished %s iterating `next()` but input queue \'%s\' isn't empty." % (self.__class__.__name__, in_key)
                warnings.warn(msg)

            self._in = None
            self._pipeline_state = "finish"
        elif self._pipeline_state == "finish":
            self._pipeline_state = "raise"
        elif self._pipeline_state == "raise":
            pass
        else:
            raise pipeline.PipelineRuntimeError()
