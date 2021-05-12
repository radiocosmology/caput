"""Interface for file formats supported by caput: HDF5 and Zarr."""
import logging
import h5py
import numcodecs
import zarr

# Copied from https://github.com/kiyo-masui/bitshuffle/blob/master/src/bshuf_h5filter.h
BSHUF_H5FILTER = 32008
BSHUF_H5_COMPRESS_LZ4 = 2

logger = logging.getLogger(__name__)


class FileFormat:
    """Abstract base class for file formats supported by this module."""

    module = None

    @staticmethod
    def open(*args, **vargs):
        """
        Open a file.

        Not implemented in base class
        """
        raise NotImplementedError

    @staticmethod
    def compression_kwargs(compression=None, compression_opts=None, compressor=None):
        """
        Sort compression arguments in a format expected by file format module.

        Parameters
        ----------
        compression : str or int
            Name or identifier of HDF5 compression filter.
        compression_opts
            See HDF5 documentation for compression filters.
        compressor : `numcodecs` compressor
            As required by `zarr`.

        Returns
        -------
        dict
            Compression arguments as required by the file format module.
        """
        if compressor and (compression or compression_opts):
            raise ValueError(
                f"Found more than one kind of compression args: compression ({compression}, {compression_opts}) "
                f"and compressor {compressor}."
            )


class HDF5(FileFormat):
    """Interface for using HDF5 file format from caput."""

    module = h5py

    @staticmethod
    def open(*args, **kwargs):
        """Open an HDF5 file using h5py."""
        return h5py.File(*args, **kwargs)

    @staticmethod
    def compression_kwargs(compression=None, compression_opts=None, compressor=None):
        """Format compression arguments for h5py API."""
        super(HDF5, HDF5).compression_kwargs(compression, compression_opts, compressor)
        if compressor:
            raise NotImplementedError
        if compression in ("bitshuffle", BSHUF_H5FILTER, str(BSHUF_H5FILTER)):
            compression = BSHUF_H5FILTER
            try:
                blocksize, c = compression_opts
            except ValueError as e:
                logger.error(
                    f"Failed to interpret compression_opts: {e}\n{compression_opts}\nUsing blocksize=0, c=2."
                )
                blocksize = 0
                c = BSHUF_H5_COMPRESS_LZ4
            if blocksize is None:
                blocksize = 0
            if c in (str(BSHUF_H5_COMPRESS_LZ4), "lz4"):
                c = BSHUF_H5_COMPRESS_LZ4
            compression_opts = (blocksize, c)

        if compression is not None:
            return {"compression": compression, "compression_opts": compression_opts}
        return {}


class Zarr(FileFormat):
    """Interface for using zarr file format from caput."""

    module = zarr

    @staticmethod
    def open(*args, **kwargs):
        """Open a zarr file."""
        return zarr.open_group(*args, **kwargs)

    @staticmethod
    def compression_kwargs(compression=None, compression_opts=None, compressor=None):
        """Format compression arguments for zarr API."""
        super(Zarr, Zarr).compression_kwargs(compression, compression_opts, compressor)
        if compression:
            if compression == "gzip":
                return {"compressor": numcodecs.gzip.GZip(level=compression_opts)}
            if compression in (BSHUF_H5FILTER, str(BSHUF_H5FILTER), "bitshuffle"):
                try:
                    blocksize, c = compression_opts
                except ValueError as e:
                    logger.error(
                        f"Failed to interpret compression_opts: {e}\n{compression_opts}\nUsing blocksize=0, c=lz4."
                    )
                    blocksize = 0
                    c = "lz4"
                if c in (BSHUF_H5_COMPRESS_LZ4, str(BSHUF_H5_COMPRESS_LZ4)):
                    c = "lz4"
                else:
                    raise ValueError(
                        f"Unknown value for cname in HDF5 compression opts: {c}"
                    )
                if blocksize is None:
                    blocksize = 0
                return {
                    "compressor": numcodecs.Blosc(
                        c,
                        shuffle=numcodecs.blosc.BITSHUFFLE,
                        blocksize=int(blocksize) if blocksize is not None else None,
                    )
                }
            else:
                raise ValueError(
                    f"Compression filter not supported in zarr: {compression}"
                )
        else:
            return {"compressor": compressor}


def guess_file_format(name, default=HDF5):
    """
    Guess the file format from the file name.

    Parameters
    ----------
    name : str
        File name.
    default : FileFormat or None
        Fallback value if format can't be guessed. Default `fileformats.HDF5`.

    Returns
    -------
    format : `FileFormat`
        File format guessed.
    """
    import pathlib

    if name.endswith(".zarr") or pathlib.Path(name).is_dir():
        return Zarr
    if name.endswith(".h5"):
        return HDF5
    return default


def check_file_format(filename, file_format, data):
    """
    Compare file format with guess from filename and data. Return concluded format.

    Parameters
    ----------
    filename : str
        File name.
    file_format : FileFormat or None
        File format. None if it should be guessed.
    data : any
        If this is an h5py.Group or zarr.Group, it will be used to guess or confirm the file format.

    Returns
    -------
    file_format : HDF5 or Zarr
        File format.
    """

    # check <file_format> value
    if file_format not in (None, HDF5, Zarr):
        raise ValueError(
            f"Unexpected value for <file_format>: {file_format} "
            f"(expected caput.fileformats.HDF5, caput.fileformats.Zarr or None)."
        )

    # guess file format from <output>
    if isinstance(data, h5py.Group):
        file_format_guess_output = HDF5
    elif isinstance(data, zarr.Group):
        file_format_guess_output = Zarr
    else:
        file_format_guess_output = None

    # guess file format from <filename>
    file_format_guess_name = guess_file_format(filename, None)

    # make sure guesses don't mismatch and decide on the format
    if (
        file_format_guess_output
        and file_format_guess_output
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
