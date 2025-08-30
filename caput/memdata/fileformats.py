"""Interface for file formats supported by :py:mod:`~caput.memdata`.

:py:mod:`fileformats` provides a unified interface for reading and writing
data in file formats supported by :py:mod:`~caput.memdata`. The intention of
this module is to abstract away the details of each file format, providing a
consistent API for users and other parts of :py:mod:`~caput.memdata`.

Fileformats are implemented as subclasses of :py:class:`FileFormat`, using the
name of the file format as the subclass name.

Supported Backends
------------------
- :py:mod:`h5py`
- :py:mod:`zarr`

Examples
--------
Opening and writing to a file:

>>> from caput.memdata import fileformats
>>>
>>> with fileformats.HDF5.open("example.h5", "w") as fh:
...     fh["dataset"] = range(10)

Compression
-----------
File compression and chunking is specified using :py:meth:`~.FileFormat.compression_kwargs`.

The backends implemented in :py:mod:`~caput.memdata.fileformats` use the `bitshuffle`_
compression filter by default, if installed. Otherwise, no compression is used.
Additional compression filters can be implemented by subclassing :py:class:`.FileFormat`
and overriding the :py:meth:`~.FileFormat.compression_kwargs` method.

.. _`bitshuffle`: https://github.com/kiyo-masui/bitshuffle
"""

from __future__ import annotations

import abc

# Check for availability of each supported filetype
# At least one filetype must have a backend installed
BACKENDS = {}

try:
    import h5py
except ImportError:
    ...
else:
    BACKENDS[h5py.__name__] = h5py

try:
    import zarr
except ImportError:
    ...
else:
    BACKENDS[zarr.__name__] = zarr

if not BACKENDS:
    raise RuntimeError(
        "No supported backends were found. "
        "At least one of [`h5py`, `zarr`] must be installed."
    )

# Check the availability of custom compression filters
# TODO: We shouldn't globally disable compression in this case,
# since standard filter like `gzip` should still be available.
try:
    import numcodecs
    from bitshuffle.h5 import H5_COMPRESS_LZ4, H5FILTER
except ModuleNotFoundError:
    H5FILTER = None
    H5_COMPRESS_LZ4 = None
    _compression_enabled = False
else:
    _compression_enabled = True

# There's a bug in `hdf5` prior to version 1.13.1 which could cause
# parallel writes to fail with certain chunk and/or compression
# parameters. `fileformats` will still work with serial I/O, so
# just warn the user.
if (
    _compression_enabled
    and ("h5py" in BACKENDS.keys())
    and (h5py.version.hdf5_version_tuple < (1, 13, 1))
):
    import warnings

    warnings.warn(
        "HDF5 parallel compression has flaws prior to version 1.13.1, and can fail "
        f"unexpectedly. The current linked version is {h5py.version.hdf5_version_tuple}.",
        RuntimeWarning,
    )


class FileFormat(metaclass=abc.ABCMeta):
    """Abstract base class for file formats supported by this module.

    Attributes
    ----------
    module : ModuleType
        Python module used for file I/O.
    compression : bool
        Whether compression is enabled for this file format.
    """

    module = None
    compression = False

    @classmethod
    def open(cls, *args, **kwargs):
        """Open a file.

        Not implemented in base class

        Returns
        -------
        file_handler : Any
            A file handler implemented by the subclass
        """
        raise NotImplementedError

    @classmethod
    def compression_kwargs(
        cls, compression=None, compression_opts=None, compressor=None
    ):
        """Arrange compression arguments into a format expected by the file format module.

        The use case of the `compression` and `compressor` arguments may
        vary in each subclass.

        Not implemented in base class.

        Parameters
        ----------
        compression : str | int | None
            Name or identifier of HDF5 compression filter.
        compression_opts : tuple | None
            Options for the selected filter. See HDF5 documentation
            for compression filters.
        compressor : :py:mod:`numcodecs` compressor
            Instance of a :py:mod:`numcodecs` compression Codec.

        Returns
        -------
        kwargs : dict
            Compression arguments as required by the file format module.
        """
        raise NotImplementedError


class HDF5(FileFormat):
    """Interface for using the HDF5 file format."""

    module = BACKENDS.get("h5py")
    compression = _compression_enabled

    @classmethod
    def open(cls, *args, **kwargs) -> h5py.File:  # noqa: D417
        r"""Open an HDF5 file using h5py.

        Parameters
        ----------
        \*args : Any
            Positional arguments passed to :py:class:`h5py.File`.
        \**kwargs : Any
            Keyword arguments passed to :py:class:`h5py.File`.

        Returns
        -------
        hdf5_handler : h5py.File
            Opened instance of :py:class:`h5py.File`.
        """
        if cls.module is None:
            raise RuntimeError(":py:mod:`h5py` is not installed.")

        return h5py.File(*args, **kwargs)

    @classmethod
    def compression_kwargs(
        cls,
        compression=None,
        compression_opts=None,
        compressor=None,
    ):
        """Format compression arguments for h5py API.

        Parameters
        ----------
        compression : str | int | None, optional
            Name or identifier of HDF5 compression filter.
        compression_opts : tuple[int, int] | None, optional
            See HDF5 documentation for compression filters.
        compressor : None
            Not supported.

        Returns
        -------
        kwargs : dict
            Compression arguments as required by the file format module.
        """
        if compressor:
            raise NotImplementedError(
                ":py:class:`numcodecs` compressor not supported for HDF5."
            )

        if compression == "bitshuffle" and not cls.compression:
            raise RuntimeError(
                "Install with 'compression' extra_require to use bitshuffle/numcodecs compression filters."
            )

        if cls.compression and compression in (
            "bitshuffle",
            H5FILTER,
            str(H5FILTER),
        ):
            if compression_opts is None:
                raise ValueError("Compression enabled but no options were provided.")

            compression = H5FILTER

            try:
                blocksize, c = compression_opts
            except ValueError as e:
                raise ValueError(
                    f"Failed to interpret compression_opts: {e}\ncompression_opts: {compression_opts}."
                ) from e

            if blocksize is None:
                blocksize = 0

            if c in (str(H5_COMPRESS_LZ4), "lz4"):
                c = H5_COMPRESS_LZ4

            compression_opts = (blocksize, c)

        if compression is not None:
            return {"compression": compression, "compression_opts": compression_opts}

        return {}


class Zarr(FileFormat):
    """Interface for using zarr file format from caput."""

    module = BACKENDS.get("zarr")
    compression = _compression_enabled

    @classmethod
    def open(cls, *args, **kwargs):  # noqa: D417
        r"""Open a zarr file.

        Parameters
        ----------
        \*args : Any
            Positional arguments passed to :py:meth:`zarr.open_group`.
        \**kwargs : Any
            Keyword arguments passed to :py:meth:`zarr.open_group`.

        Returns
        -------
        file_handler : zarr.Group
            Opened instance of :py:class:`zarr.Group`.
        """
        if cls.module is None:
            raise RuntimeError(":py:mod:`zarr` is not installed.")

        return zarr.open_group(*args, **kwargs)

    @classmethod
    def compression_kwargs(
        cls,
        compression=None,
        compression_opts=None,
        compressor=None,
    ):
        """Format compression arguments for zarr API.

        Only `compressor` *or* `compression`/`compression_opts`
        should be provided.

        Parameters
        ----------
        compression : str | int | None, optional
            Name or identifier of compression filter.
        compression_opts : tuple[int, int] | None, optional
            See HDF5 and/or Zarr documentation for compression filters.
        compressor : :py:class:`~numcodecs.abc.Codec` | None
            :py:mod:`numcodecs` compression codec for :py:mod:`zarr`.
            See the numcodecs docs for more information.

        Returns
        -------
        kwargs : dict
            Compression arguments as required by the file format module.
        """
        if compressor and (compression or compression_opts):
            raise ValueError(
                "Got both compressor and compression options. Only `compressor` or "
                "`compression`/`compression_opts` should be provided."
            )

        if compression:
            if not cls.compression:
                raise ValueError(
                    "Install with 'compression' extra_require to use bitshuffle/numcodecs compression filters."
                )

            if compression == "gzip":
                return {"compressor": numcodecs.gzip.GZip(level=compression_opts)}

            if compression in (H5FILTER, str(H5FILTER), "bitshuffle"):
                try:
                    blocksize, c = compression_opts
                except ValueError as e:
                    raise ValueError(
                        f"Failed to interpret compression_opts: {e}\ncompression_opts: {compression_opts}"
                    ) from e

                if c in (H5_COMPRESS_LZ4, str(H5_COMPRESS_LZ4)):
                    c = "lz4"

                if blocksize is None:
                    blocksize = 0

                return {
                    "compressor": numcodecs.Blosc(
                        c,
                        shuffle=numcodecs.blosc.BITSHUFFLE,
                        blocksize=int(blocksize) if blocksize is not None else None,
                    )
                }

            raise ValueError(f"Compression filter not supported in zarr: {compression}")

        return {"compressor": compressor}


class ZarrProcessSynchronizer:
    """A context manager for Zarr's ProcessSynchronizer that removes the lock files when done.

    If an MPI communicator is supplied, only rank 0 will attempt to remove files.

    Parameters
    ----------
    name : str
        Name of the lockfile directory.
    comm : MPI.Comm | None, optional
        MPI communicator (optional).
    """

    def __init__(self, name, comm=None):
        self.name = name
        self._comm = comm

    def __enter__(self):
        return zarr.ProcessSynchronizer(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._comm is None or self._comm.rank == 0:
            remove_file_or_dir(self.name)


def remove_file_or_dir(name):
    """Remove the file or directory with the given name.

    Parameters
    ----------
    name : str
        File or directory name to remove.
    """
    import os
    import shutil

    if os.path.isdir(name):
        try:
            shutil.rmtree(name)
        except FileNotFoundError:
            pass
    else:
        try:
            os.remove(name)
        except FileNotFoundError:
            pass


def guess_file_format(name, default=HDF5):
    """Guess the file format from the file name.

    Parameters
    ----------
    name : os.PathLike
        File name.
    default : HDF5
        Fallback value if format can't be guessed. Default :py:class:`.HDF5`.

    Returns
    -------
    file_format : FileFormat
        Guessed :py:class:`.FileFormat` instance.
    """
    import pathlib

    if isinstance(name, pathlib.Path):
        name = str(name)

    if name.endswith(".zarr.zip"):
        return Zarr

    if name.endswith(".zarr") or pathlib.Path(name).is_dir():
        return Zarr

    if name.endswith(".h5") or name.endswith(".hdf5"):
        return HDF5

    return default


def check_file_format(filename, file_format, data):
    """Attempt to detect the format of a file.

    Parameters
    ----------
    filename : str
        File name.
    file_format : FileFormat | None
        Expected file format. If not ``None``, an exception is raised if the
        detected file type is not this value. If ``None``, no check is done
        against the detected format.
    data : Any
        If this is an :py:class:`h5py.Group` or :py:class:`zarr.Group`, it will be used to
        guess or confirm the file format.

    Returns
    -------
    file_format : FileFormat
        The detected :py:class:`.FileFormat`.
    """
    # check <file_format> value
    if file_format not in (None, HDF5, Zarr):
        raise ValueError(
            f"Unexpected value for <file_format>: {file_format} "
            f"(expected caput.fileformats.HDF5, caput.fileformats.Zarr or None)."
        )

    # guess file format from <output>
    if ("h5py" in BACKENDS) and isinstance(data, h5py.Group):
        file_format_guess_output = HDF5
    elif ("zarr" in BACKENDS) and isinstance(data, zarr.Group):
        file_format_guess_output = Zarr
    else:
        file_format_guess_output = None

    # guess file format from <filename>
    file_format_guess_name = guess_file_format(filename, None)

    # make sure guesses don't mismatch and decide on the format
    if (
        file_format_guess_output
        and file_format_guess_name
        and file_format_guess_name != file_format_guess_output
    ):
        raise ValueError(
            f"<file_format> ({file_format}) and <filename> ({filename}) don't seem to match."
        )
    file_format_guess = (
        file_format_guess_output if file_format_guess_output else file_format_guess_name
    )

    if file_format is None:
        file_format = file_format_guess
    elif file_format != file_format_guess:
        raise ValueError(
            f"Value of <file_format> ({file_format}) doesn't match <filename> ({filename}) "
            f"and type of data ({type(data).__name__})."
        )

    return file_format
