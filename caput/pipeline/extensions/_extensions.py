"""Assorted extensions for :py:class:`~caput.pipeline.Task`.

Anything in here will potentially be deprecated.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import TYPE_CHECKING

from ... import config
from ...memdata import fileformats, lock_file
from .. import exceptions
from .._pipeline import Task
from ._configtypes import file_format

if TYPE_CHECKING:
    from ...memdata import MemGroup
    from ...memdata._memh5 import GroupLike

__all__ = ["ContainerMixin", "H5IOMixin"]


# Set the module logger.
logger = logging.getLogger(__name__)


class _OneAndOne(Task):
    """Base class for tasks that have (at most) one input and one output.

    This is not a user base class and simply holds code that is common to
    :py:class:`SingleBase` and :py:class:`IterBase`.
    """

    input_root = config.Property(default="None", proptype=str)
    output_root = config.Property(default="None", proptype=str)
    output_format = file_format()

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
        This is called by the :py:class:`Manager` after it's added to the pipeline
        and has special attributes like ``_requires_keys``, ``_requires``, ``_in_keys``,
        ``_in``, ``-_out_keys`` set. Call :py:meth:`super().validate` if you overwrite this.

        Raises
        ------
        :py:exc:`~caput.config.CaputConfigError`
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
            msg = "`process` method may not have variable length or optional arguments."
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
                msg = "Reading input from disk but `process` method takes no arguments."
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

    Inherits from :py:class:`~caput.pipeline.Task`.

    Tasks inheriting from this class should override :py:meth:`~caput.pipeline.Task.process`
    and optionally :py:meth:`~caput.pipeline.Task.setup`, :py:meth:`~caput.pipeline.Task.finish`,
    :py:meth:`~caput.pipeline._OneAndOne.read_input`, :py:meth:`~caput.pipeline._OneAndOne.write_output`
    and :py:meth:`~caput.pipeline._OneAndOne.cast_input`.
    They should not override :py:meth:`~caput.pipeline.Task.next`.

    If the value of :py:attr:`~.SingleBase.input_root` is anything other than the string "None"
    then the input will be read (using :py:meth:`~.SingleBase.read_input`) from the file
    ``self.input_root + self.input_filename``.  If the input is specified both as
    a filename and as a product key in the pipeline configuration, an error
    will be raised upon initialization.


    If the value of :py:attr:`~.SingleBase.output_root` is anything other than the string
    "None" then the output will be written (using :py:meth:`~.SingleBase.write_output`) to the
    file ``self.output_root + self.output_filename``.

    Attributes
    ----------
    input_root
        Pipeline settable parameter giving the first part of the input path.
        If set to 'None' no input is read. Either it is assumed that no input
        is required or that input is recieved from the pipeline.
    input_filename
        Pipeline settable parameter giving the last part of input path. The
        full input path is ``self.input_root + self.input_filename``.
    output_root
        Pipeline settable parameter giving the first part of the output path.
        If set to 'None' no output is written.
    output_filename
        Pipeline settable parameter giving the last part of output path. The
        full output path is ``self.output_root + self.output_filename``.
    """

    input_filename: str = config.Property(default="", proptype=str)
    output_filename: str = config.Property(default="", proptype=str)
    output_format = file_format()
    output_compression: str = config.Property(default=None, proptype=str)
    output_compression_opts: dict | str | None = config.Property(default=None)

    def next(self, input=None):
        """Should not need to override."""
        # This should only be called once.
        try:
            if self.done:
                raise exceptions.PipelineStopIteration
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
                raise exceptions.PipelineStopIteration
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
    """Provides HDF5 IO for pipeline tasks.

    As a mixin, this must be combined (using multiple inheritance) with a
    subclass of `TaskBase`, providing the full task API.

    Provides the methods `read_input`, `read_output` and `write_output` for
    HDF5 data.
    """

    # TODO, implement reading on disk (i.e. no copy to memory).
    # ondisk = config.Property(default=False, proptype=bool)

    @staticmethod
    def read_input(filename: str) -> MemGroup:
        """Method for reading HDF5 input."""
        from ...memdata import MemGroup

        return MemGroup.from_hdf5(filename, mode="r")

    @staticmethod
    def read_output(filename: str) -> MemGroup:
        """Method for reading HDF5 output (from caches)."""
        from ...memdata import MemGroup

        # Replicate code from read_input in case read_input is overridden.
        return MemGroup.from_hdf5(filename, mode="r")

    @staticmethod
    def write_output(  # noqa: D417
        filename: str,
        output: GroupLike,
        file_format: fileformats.FileFormat | None = None,
        **kwargs: dict,
    ) -> None:
        r"""Method for writing HDF5 output.

        Parameters
        ----------
        filename
            File name
        output
            `output` to be written. If this is a `h5py.Group` (which include `hdf5.File` objects)
            the buffer is flushed if `filename` points to the same file and a copy is made otherwise.
        file_format
            File format to use. If this is not specified, the file format is guessed based on the type of
            `output` or the `filename`. If guessing is not successful, HDF5 is used.
        \**kwargs
            Arbitrary keyword arguments.
        """
        import h5py

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
        if isinstance(output, MemGroup):
            # Already in memory.

            # Lock file
            with lock_file(filename, comm=output.comm) as fn:
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
                out_copy = MemGroup.from_hdf5(output)

                # Lock file as we write
                with lock_file(filename, comm=out_copy.comm) as fn:
                    out_copy.to_hdf5(fn, mode="w")
        elif isinstance(output, zarr.Group):
            if os.path.isdir(filename) and os.path.samefile(
                output.store.path, filename
            ):
                pass
            else:
                logger.debug(f"Copying {output.store}:{output.path} to {filename}.")
                from ...util import mpitools

                if mpitools.rank == 0:
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        output.store,
                        zarr.DirectoryStore(filename),
                        source_path=output.path,
                    )
                logger.debug(
                    f"Copied {n_copied} items ({n_bytes_copied} bytes), skipped {n_skipped} items."
                )


class ContainerMixin:
    """Provides IO for Container objects in pipeline tasks.

    As a mixin, this must be combined (using multiple inheritance) with a
    subclass of `TaskBase`, providing the full task API.

    Provides the methods :py:meth:`~.ContainerMixin.read_input`,
    :py:meth:`~.ContainerMixin.read_output` and :py:meth:`~.ContainerMixin.write_output`
    for :py:class:`~caput.containers.Container` data which gets written to disk.
    """

    # TODO, implement reading on disk (i.e. no copy to memory).
    # ondisk = config.Property(default=False, proptype=bool)

    # Private setting for reading of inputs, should be overriden in sub class.
    _distributed = False
    _comm = None

    def read_input(self, filename):
        """Method for reading input from disk.

        Any file format supported by :py:class:`~caput.memdata.BasicCont`
        is supported.

        Parameters
        ----------
        filename : PathLike
            Path to a file to load.

        Returns
        -------
        container : Container
            Container instance from a file.
        """
        from ...containers import Container

        return Container.from_file(
            filename, distributed=self._distributed, comm=self._comm
        )

    def read_output(self, filename):
        """Method for reading output (from caches).

        Parameters
        ----------
        filename : PathLike
            Path to a file to read.

        Returns
        -------
        container : Container
            Container instance from a file.
        """
        # Replicate code from read_input in case read_input is overridden.
        from ...containers import Container

        return Container.from_file(
            filename, distributed=self._distributed, comm=self._comm
        )

    @staticmethod
    def write_output(  # noqa: D417
        filename,
        output,
        file_format=None,
        **kwargs,
    ):
        r"""Method for writing output to disk.

        Parameters
        ----------
        filename : PathLike
            File name.
        output : Container
            Data to be written.
        file_format : FileFormat
            File format to use. Default is ``None``, in which
            case the file format is guessed.
        \**kwargs : Any
            Arbitrary keyword arguments.
        """
        from ...containers import Container

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
        if not isinstance(output, Container):
            raise RuntimeError(
                "Object to write out is not an instance of containers.Container"
            )

        # Already in memory.
        output.save(filename, file_format=file_format, **kwargs)


class SingleH5Base(H5IOMixin, SingleBase):
    """Base class for tasks with hdf5 input and output.

    Inherits from :class:`H5IOMixin` and :class:`SingleBase`.
    """


class IterH5Base(H5IOMixin, IterBase):
    """Base class for iterating over hdf5 input and output.

    Inherits from :class:`H5IOMixin` and :class:`IterBase`.
    """
