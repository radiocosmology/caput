"""Base pipeline tasks.

Base pipeline tasks provide standard functionality, like logging,
MPI support, and container handling. These are intended to be subclassed
when building more complex and/or distributed tasks.
"""

from __future__ import annotations

import logging
import os
from inspect import getfullargspec

import numpy as np

from ... import config
from ...memdata import MemDataset, MemDiskGroup, MemGroup, fileformats
from .. import Task, exceptions, extensions

__all__ = [
    "ContainerTask",
    "LoggedTask",
    "MPILogFilter",
    "MPILoggedTask",
    "MPITask",
    "SetMPILogging",
    "group_tasks",
]


class MPILogFilter(logging.Filter):
    """Filter log entries by MPI rank.

    Also this will optionally add MPI rank information, and add an elapsed time
    entry.

    Parameters
    ----------
    add_mpi_info : bool, optional
        Add MPI rank/size info to log records that don't already have it.
    level_rank0 : int, optional
        Log level for messages from rank=0. Default is `logging.INFO`.
    level_all : int, optional
        Log level for messages from all other ranks. Default is `logging.WARN`.
    """

    def __init__(
        self,
        add_mpi_info: bool = True,
        level_rank0: int = logging.INFO,
        level_all: int = logging.WARN,
    ):
        from mpi4py import MPI

        self.add_mpi_info = add_mpi_info

        self.level_rank0 = level_rank0
        self.level_all = level_all

        self.comm = MPI.COMM_WORLD

    def filter(self, record):
        """Add MPI info if desired."""
        try:
            record.mpi_rank
        except AttributeError:
            if self.add_mpi_info:
                record.mpi_rank = self.comm.rank
                record.mpi_size = self.comm.size

        # Add a new field with the elapsed time in seconds (as a float)
        record.elapsedTime = record.relativeCreated * 1e-3

        # Return whether we should filter the record or not.
        return (record.mpi_rank == 0 and record.levelno >= self.level_rank0) or (
            record.mpi_rank > 0 and record.levelno >= self.level_all
        )


def _log_level(x):
    """Interpret the input as a logging level.

    Parameters
    ----------
    x : int | str
        Explicit integer logging level or one of 'DEBUG', 'INFO', 'WARN',
        'ERROR' or 'CRITICAL'.

    Returns
    -------
    log_level_integer : int
        The logging level as an integer.

    Raises
    ------
    ValueError
        If the input is not a valid logging level.
    """
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARN,
        "WARNING": logging.WARN,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    if isinstance(x, int):
        return x

    if isinstance(x, str) and x in level_dict:
        return level_dict[x.upper()]

    raise ValueError(f"Logging level {x!r} not understood")


class SetMPILogging(Task):
    """A task used to configure MPI aware logging.

    Attributes
    ----------
    level_rank0 : int, optional
        Log level for rank=0. Default is "INFO".
    level_all : int, optional
        Log level for all ranks except rank=0. Default is "WARN".
    """

    level_rank0 = config.Property(proptype=_log_level, default=logging.INFO)
    level_all = config.Property(proptype=_log_level, default=logging.WARN)

    def __init__(self):
        import math

        from mpi4py import MPI

        logging.captureWarnings(True)

        rank_length = int(math.log10(MPI.COMM_WORLD.size)) + 1

        mpi_fmt = f"[MPI %(mpi_rank){rank_length:d}d/%(mpi_size){rank_length:d}d]"
        filt = MPILogFilter(level_all=self.level_all, level_rank0=self.level_rank0)

        # This uses the fact that caput.pipeline.Manager has already
        # attempted to set up the logging. We just insert our custom filter
        root_logger = logging.getLogger()
        ch = root_logger.handlers[0]
        ch.addFilter(filt)

        formatter = logging.Formatter(
            "%(elapsedTime)8.1fs "
            + mpi_fmt
            + " - %(levelname)-8s %(name)s: %(message)s"
        )

        ch.setFormatter(formatter)


class LoggedTask(Task):
    """A task with logger support.

    Attributes
    ----------
    log_level : int, optional
        Logging level for this task. Default is ``None``.
    """

    log_level = config.Property(proptype=_log_level, default=None)

    def __init__(self):
        # Get the logger for this task
        self._log = logging.getLogger(f"{self.__module__}.{self.__class__.__name__}")

        # Set the log level for this task if specified
        if self.log_level is not None:
            self.log.setLevel(self.log_level)

    @property
    def log(self):
        """The logger object for this task."""
        return self._log


class MPITask(Task):
    """Base class for MPI using tasks.

    Just ensures that the task gets a `comm` attribute.

    Attributes
    ----------
    comm : MPI.Comm | None, optional
        MPI communicator.
    """

    comm = None

    def __init__(self):
        from mpi4py import MPI

        # Set default communicator
        self.comm = MPI.COMM_WORLD


class _AddRankLogAdapter(logging.LoggerAdapter):
    """Add the rank of the logging process to a log message.

    Attributes
    ----------
    calling_obj : Any | None
        An object with a `comm` property that will be queried for the rank.
    """

    calling_obj = None

    def process(self, msg, kwargs):
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        kwargs["extra"]["mpi_rank"] = self.calling_obj.comm.rank
        kwargs["extra"]["mpi_size"] = self.calling_obj.comm.size

        return msg, kwargs


class MPILoggedTask(MPITask, LoggedTask):
    """A task base that has MPI aware logging."""

    def __init__(self):
        # Initialise the base classes
        MPITask.__init__(self)
        LoggedTask.__init__(self)

        # Replace the logger with a LogAdapter instance that adds MPI process
        # information
        logadapter = _AddRankLogAdapter(self._log, None)
        logadapter.calling_obj = self
        self._log = logadapter


class ContainerTask(MPILoggedTask, extensions.ContainerIOMixin):
    """Implements a task whose inputs and outputs are :py:class:`~caput.containers.Container` objects.

    This task implements writing of the output when requested, and handles various
    types of metadata associated with the container objects.

    Tasks inheriting from this class should override :py:meth:`~.ContainerTask.process`
    and optionally :py:meth:`~.ContainerTask.setup` or :py:meth:`~.ContainerTask.process_finish`.
    They should not override :py:meth:`~.ContainerTask.next` or :py:meth:`~.ContainerBase.finish`.

    Output will be written (using :py:meth:`write_output`) to the file ``self.output_name``.

    Attributes
    ----------
    save : bool | list[bool], optional
        Whether to save the output to disk or not. Can be provided as a list
        if multiple outputs are being handled. Default is False.
    attrs : dict | None, optional
        A mapping of attribute names and values to set in the `.attrs` at the root of
        the output container. String values will be formatted according to the standard
        Python `.format(...)` rules, and can interpolate several other values into the
        string. These are:

        - `count`: an integer giving which iteration of the task is this.
        - `tag`: a string identifier for the output derived from the
                 containers `tag` attribute. If that attribute is not present
                 `count` is used instead.
        - `key`: the name of the output key.
        - `task`: the (unqualified) name of the task.
        - `input_tags`: a list of the tags for each input argument for the task.
        - Any existing attribute in the container can be interpolated by the name of
          its key. The specific values above will override any attribute with the same
          name.

        Incorrectly formatted values will cause an error to be thrown. Default is
        ``None``
    tag : str, optional
        Set a format for the tag attached to the output. This is a Python format string
        which can interpolate the variables listed under `attrs` above. For example a
        tag of "cat{count}" will generate catalogs with the tags "cat1", "cat2", etc.
        Default is `{tag}`.
    output_name : str | list[str], optional
        A python format string used to construct the filename. All variables given under
        `attrs` above can be interpolated into the filename. Can be provided as a list
        if multiple output are being handled.
        Valid identifiers are:

          - `count`: an integer giving which iteration of the task is this.
          - `tag`: a string identifier for the output derived from the
                   containers `tag` attribute. If that attribute is not present
                   `count` is used instead.
          - `key`: the name of the output key.
          - `task`: the (unqualified) name of the task.
          - `output_root`: the value of the output root argument. This is deprecated
                           and is just used for legacy support. The default value of
                           `output_name` means the previous behaviour works.

        Default is `{output_root}{tag}.h5`.

    compression : bool | dict, optional
        Set compression options for each dataset. Provided as a dict with the dataset
        names as keys and values for `chunks`, `compression`, and `compression_opts`.
        Any datasets not included in the dict (including if the dict is empty), will use
        the default parameters set in the dataset spec. If set to `False` (or anything
        that evaluates to `False`, other than an empty dict) chunks and compression will
        be disabled for all datasets. If no argument in provided, the default parameters
        set in the dataset spec are used. Note that this will modify these parameters on
        the container itself, such that if it is written out again downstream in the
        pipeline these will be used. Default is ``True``.
    output_root : str, optional
        Pipeline settable parameter giving the first part of the output path.
        Deprecated in favour of specifying the output path directly in `output_name`.
    nan_check : bool, optional
        Check the output for NaNs (and infs) logging if they are present. Default
        is ``True``.
    nan_dump : bool, optional
        If NaN's are found, dump the container to disk. Default is ``True``.
    nan_skip : bool, optional
        If NaN's are found, don't pass on the output. Default is ``True``.
    versions : dict[str, str], optional
        Keys are module names (str) and values are their version strings. This is
        attached to output metadata. Default is {}.
    pipeline_config : dict, optional
        Global pipeline configuration. This is attached to output metadata. Default
        is {}.

    Raises
    ------
    :py:exc:`~caput.pipeline.exceptions.PipelineRuntimeError`
        If this is used as a baseclass to a task overriding `self.process` with variable
        length or optional arguments.
    """

    save = config.Property(
        default=False, proptype=lambda x: x if isinstance(x, list) else bool(x)
    )

    output_root = config.Property(default="", proptype=str)
    output_name = config.Property(
        default="{output_root}{tag}.h5",
        proptype=lambda x: x if isinstance(x, list) else str(x),
    )
    output_format = extensions.file_format()
    compression = config.Property(
        default=True, proptype=lambda x: x if isinstance(x, dict) else bool(x)
    )

    nan_check = config.Property(default=True, proptype=bool)
    nan_skip = config.Property(default=True, proptype=bool)
    nan_dump = config.Property(default=True, proptype=bool)

    # Metadata to get attached to the output
    versions = config.Property(default={}, proptype=dict)
    pipeline_config = config.Property(default={}, proptype=dict)

    tag = config.Property(proptype=str, default="{tag}")
    attrs = config.Property(proptype=dict, default=None)

    _count = 0

    done = False
    _no_input = False

    def __init__(self):
        super().__init__()

        # Inspect the `process` method to see how many arguments it takes.
        pro_argspec = getfullargspec(self.process)
        n_args = len(pro_argspec.args) - 1

        if pro_argspec.varargs or pro_argspec.varkw or pro_argspec.defaults:
            msg = "`process` method may not have variable length or optional arguments."
            raise exceptions.PipelineRuntimeError(msg)

        if n_args == 0:
            self._no_input = True
        else:
            self._no_input = False

    def next(self, *input):
        """Iterates through inputs.

        You should not need to override this. Implement
        :py:meth:`~.ContainerTask.process` instead.
        """
        self.log.info(f"Starting next for task {self.__class__.__name__}")

        self.comm.Barrier()

        # This should only be called once.
        try:
            if self.done:
                raise exceptions.PipelineStopIteration
        except AttributeError:
            self.done = True

        # Extract a list of the tags for all input arguments
        input_tags = [
            (str(icont.attrs.get("tag")) if isinstance(icont, MemDiskGroup) else "")
            for icont in input
        ]

        # Process input and fetch output
        if self._no_input:
            if len(input) > 0:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set.")
            output = self.process()
        else:
            output = self.process(*input)

        # Return immediately if output is None to skip writing phase.
        if output is None:
            return None

        # Ensure output is a tuple
        if not isinstance(output, tuple):
            output = (output,)

        # Insert the input tags into the output containers
        for opt in output:
            opt.attrs["input_tags"] = input_tags

        # Process each output individually
        output = tuple(self._process_output(opt, ii) for ii, opt in enumerate(output))

        # Increment internal counter
        self._count = self._count + 1

        self.log.info(f"Leaving next for task {self.__class__.__name__}")

        # Return the output for the next task. If there is
        # only a single output, don't wrap as a tuple
        return output if len(output) > 1 else output[0]

    def finish(self):
        """Should not need to override. Implement :py:meth:`~.ContainerTask.process_finish` instead."""
        class_name = self.__class__.__name__

        self.log.info(f"Starting finish for task {class_name}")

        if not hasattr(self, "process_finish"):
            self.log.info(f"No finish for task {class_name}")
            return None

        output = self.process_finish()

        # Return immediately if output is None to skip writing phase.
        if output is None:
            self.log.info(f"Leaving finish for task {class_name}")
            return None

        # Ensure output is a tuple
        if not isinstance(output, tuple):
            output = (output,)

        # Process each output individually
        output = tuple(self._process_output(opt, ii) for ii, opt in enumerate(output))

        self.log.info(f"Leaving finish for task {class_name}")

        # If there is only a single output, don't wrap as a tuple
        return output if len(output) > 1 else output[0]

    def _process_output(self, output, ii=0):
        """Check a single output container and write it if needed."""
        if not isinstance(output, MemDiskGroup):
            raise exceptions.PipelineRuntimeError(
                f"Task must output a valid memdata container; given {type(output)}"
            )

        # Set the tag according to the format
        idict = self._interpolation_dict(output, ii)

        # Set the attributes in the output container (including from the `tag` config
        # option)
        attrs_to_set = {} if self.attrs is None else self.attrs.copy()
        attrs_to_set["tag"] = self.tag
        for attrname, attrval in attrs_to_set.items():
            if isinstance(attrval, str):
                attrval = attrval.format(**idict)
            output.attrs[attrname] = attrval

        # Check for NaN's etc
        output = self._nan_process_output(output)

        # Write the output if needed
        self._save_output(output, ii)

        return output

    def _save_output(self, output, ii=0):
        """Save a single output container and return the file path if it was saved."""
        if output is None:
            return None

        # Parse compression/chunks options
        def walk_dset_tree(grp, root=""):
            # won't find forbidden datasets like index_map but we are not compressing those
            datasets = []
            for key in grp:
                if isinstance(grp[key], MemGroup):
                    datasets += walk_dset_tree(grp[key], f"{root}{key}/")
                else:
                    datasets.append(root + key)
            return datasets

        if isinstance(self.compression, dict):
            # We want to overwrite some compression settings
            datasets = walk_dset_tree(output)
            for ds in self.compression:
                if ds in datasets:
                    for key, val in self.compression[ds].items():
                        self.log.debug(
                            f"Overriding default compression setting on dataset {ds}: {key}={val}."
                        )
                        setattr(output._data._storage_root[ds], key, val)
                    # shorthand for bitshuffle
                    if output[ds].compression in (
                        "bitshuffle",
                        fileformats.H5FILTER,
                    ):
                        output[ds].compression = fileformats.H5FILTER
                        if output[ds].compression_opts is None:
                            output._data._storage_root[ds].compression_opts = (
                                0,
                                fileformats.H5_COMPRESS_LZ4,
                            )
                else:
                    self.log.warning(
                        f"Ignoring config entry in `compression` for non-existing dataset `{ds}`"
                    )
        elif not self.compression:
            # Disable compression
            for ds in walk_dset_tree(output):
                output._data._storage_root[ds].chunks = None
                output._data._storage_root[ds].compression = None
                output._data._storage_root[ds].compression_opts = None

        # Routine to write output if needed.
        if isinstance(self.save, list):
            save = self.save[ii]
        else:
            save = self.save

        if save:
            # add metadata to output
            metadata = {"versions": self.versions, "config": self.pipeline_config}
            for key, value in metadata.items():
                output.add_history(key, value)

            # Construct the filename
            name_parts = self._interpolation_dict(output, ii)
            if self.output_root != "":
                self.log.warning("Use of `output_root` is deprecated.")
                name_parts["output_root"] = self.output_root

            if isinstance(self.output_name, list):
                outfile = self.output_name[ii].format(**name_parts)
            else:
                outfile = self.output_name.format(**name_parts)

            # Expand any variables in the path
            outfile = os.path.expanduser(outfile)
            outfile = os.path.expandvars(outfile)

            self.log.debug(f"Writing output {outfile} to disk.")
            self.write_output(
                outfile,
                output,
                file_format=self.output_format,
            )
            return outfile

        return None

    def _nan_process_output(self, output):
        """Check a single output container for NaN's and handle them."""
        if not isinstance(output, MemDiskGroup):
            raise exceptions.PipelineRuntimeError(
                f"Task must output a valid memdata container; given {type(output)}"
            )

        if self.nan_check:
            nan_found = self._nan_check_walk(output)

            if nan_found and self.nan_dump:
                # Construct the filename
                tag = output.attrs["tag"] if "tag" in output.attrs else self._count
                outfile = "nandump_" + self.__class__.__name__ + "_" + str(tag) + ".h5"

                self.log.debug(f"NaN found. Dumping {outfile}")

                self.write_output(outfile, output)

            if nan_found and self.nan_skip:
                self.log.debug("NaN found. Skipping output.")
                return None

        return output

    def _interpolation_dict(self, output, ii=0):
        """Get the set of variables that can be used to create the output filename."""
        idict = dict(output.attrs)
        if "tag" in output.attrs:
            tag = output.attrs["tag"]
        elif "input_tags" in output.attrs and len(output.attrs["input_tags"]):
            tag = output.attrs["input_tags"][0]
        else:
            tag = self._count

        idict.update(
            tag=tag,
            count=self._count,
            task=self.__class__.__name__,
            key=(
                self._out_keys[ii]
                if hasattr(self, "_out_keys") and self._out_keys
                else ""
            ),
            output_root=self.output_root,
        )
        return idict

    def _nan_check_walk(self, cont):
        """Walk through a memdata container and check for NaN's and Inf's.

        Logs any issues found and returns True if there were any found.
        """
        from mpi4py import MPI

        if isinstance(cont, MemDiskGroup):
            cont = cont._data

        stack = [cont]
        found = False

        # Walk over the container tree...
        while stack:
            n = stack.pop()

            # Check the dataset for non-finite numbers
            if isinstance(n, MemDataset):
                # Try to test for NaN's and infs. This will fail for compound datatypes...
                # casting to ndarray, bc MPI ranks may fall out of sync, if a nan or inf are found
                arr = n[:].view(np.ndarray)
                try:
                    all_finite = np.isfinite(arr).all()
                except TypeError:
                    continue

                if not all_finite:
                    nans = np.isnan(arr).sum()
                    if nans > 0:
                        self.log.info(
                            f"NaN's found in dataset {n.name} [{nans} of {arr.size} elements]"
                        )
                        found = True
                        break

                    infs = np.isinf(arr).sum()
                    if infs > 0:
                        self.log.info(
                            f"Inf's found in dataset {n.name} [{infs} of {arr.size} elements]"
                        )
                        found = True
                        break

            elif isinstance(n, MemGroup | MemDiskGroup):
                for item in n.values():
                    stack.append(item)

        # All ranks need to know if any rank found a NaN/Inf
        return self.comm.allreduce(found, op=MPI.MAX)


# Alias for backwards compatibility
class SingleTask(ContainerTask):
    """Alias for :py:meth:`~caput.pipeline.tasklib.ContainerTask` for backwards compatibility."""

    def __init__(self):
        import warnings

        warnings.warn(
            "`SingleTask` is deprecated, and is an alias of `ContainerTask`. "
            "Please use `ContainerTask` instead.",
            DeprecationWarning,
        )
        super().__init__()


def group_tasks(*tasks):
    r"""Create a Task that groups a bunch of tasks together.

    This method creates a class that inherits from all the subtasks, and
    calls each `process` method in sequence, passing the output of one to the
    input of the next.

    This should be used like:

    >>> class SuperTask(group_tasks(SubTask1, SubTask2)):  # doctest: +SKIP
    ...     pass

    At the moment, if the ensemble has more than one setup method, the
    SuperTask will need to implement an override that correctly calls each.
    """

    class TaskGroup(*tasks):
        # TODO: figure out how to make the setup work at the moment it just picks the first in MRO
        # def setup(self, x): pass

        def process(self, x):
            for t in tasks:
                self.log.debug(f"Calling process for subtask {t.__name__!s}")
                x = t.process(self, x)

            return x

    return TaskGroup


# =========================================================
# Stuff that is likely deprecated but I still have to check
# =========================================================


class _ReturnLastInputOnFinish(SingleTask):
    """Workaround for `caput.pipeline` issues.

    This caches its input on every call to `process` and then returns
    the last one for a finish call.
    """

    x = None

    def process(self, x):
        """Take a reference to the input.

        Parameters
        ----------
        x : Any
            Object to cache
        """
        self.x = x

    def process_finish(self):
        """Return the last input to process.

        Returns
        -------
        last_input : Any
            Last input to process.
        """
        return self.x


class _ReturnFirstInputOnFinish(SingleTask):
    """Workaround for `caput.pipeline` issues.

    This caches its input on the first call to `process` and
    then returns it for a finish call.
    """

    x = None

    def process(self, x):
        """Take a reference to the input.

        Parameters
        ----------
        x : Any
            Object to cache
        """
        if self.x is None:
            self.x = x

    def process_finish(self):
        """Return the last input to process.

        Returns
        -------
        last_input : Any
            Last input to process.
        """
        return self.x
