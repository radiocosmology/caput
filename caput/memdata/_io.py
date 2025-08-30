""":py:mod:`~.caput.memdata` I/O utilities.

General MPI-I/O utilities used by `:py:mod:`~.caput.memdata`. Don't import there
directly from the private module. They may be moved elsewhere at any time.
"""

from __future__ import annotations

import os


def open_h5py_mpi(f, mode, use_mpi=True, comm=None):
    """Ensure that we have an h5py File object.

    Opens with MPI-IO if possible.

    The returned file handle is annotated with two attributes: `.is_mpi`
    which says whether the file was opened as an MPI file and `.opened` which
    says whether it was opened in this call.

    Parameters
    ----------
    f : os.PathLike | h5py.File | h5py.Group
        Filename to open, or already open file object. If already open this
        is just returned as is.
    mode : str
        Mode to open file in.
    use_mpi : bool, optional
        Whether to use MPI-IO or not (default True)
    comm : MPI.Comm | None, optional
        MPI communicator to use. Uses `COMM_WORLD` if not set.

    Returns
    -------
    file_handle : h5py.File
        File handle for :py:class:`h5py.File`, with two extra attributes
        ``.is_mpi`` and ``.opened``.
    """
    import h5py

    has_mpi = h5py.get_config().mpi

    if isinstance(f, str):
        # Open using MPI-IO if we can
        if has_mpi and use_mpi:
            from mpi4py import MPI

            comm = comm if comm is not None else MPI.COMM_WORLD
            fh = h5py.File(f, mode, driver="mpio", comm=comm)
        else:
            fh = h5py.File(f, mode)
        fh.opened = True
    elif isinstance(f, h5py.File | h5py.Group):
        fh = f
        fh.opened = False
    else:
        raise ValueError(
            f"Can't write to {f} (Expected a h5py.File, h5py.Group or str filename)."
        )

    fh.is_mpi = fh.file.driver == "mpio"

    return fh


class lock_file:
    """Manage a lock file around a file creation operation.

    Parameters
    ----------
    filename : str
        Final name for the file.
    preserve : bool, optional
        Keep the temporary file in the event of failure.
    comm : MPI.Comm | None, optional
        If not ``None``, only rank=0 will create/remove the lock file and move the
        file.

    Examples
    --------
    >>> from caput import memdata
    >>> container = memdata.BasicCont()
    >>> with lock_file("file_to_create.h5") as fname:
    ...     container.save(fname)
    """

    def __init__(self, filename, preserve=False, comm=None):
        if comm is not None and not hasattr(comm, "rank"):
            raise ValueError("comm argument does not seem to be an MPI communicator.")

        self.filename = filename
        # If comm not specified, set internal rank0 marker to True,
        # so that rank>0 tasks can open their own files
        self.rank0 = True if comm is None else comm.rank == 0
        self.preserve = preserve

    def __enter__(self):
        if self.rank0:
            with open(self.lockfile, "w+") as fh:
                fh.write("")

        return self.tmpfile

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.rank0:
            # Check if exception was raised and delete the temp file if needed
            if exc_type is not None:
                if not self.preserve:
                    os.remove(self.tmpfile)
            # Otherwise things were successful and we should move the file over
            else:
                os.rename(self.tmpfile, self.filename)

            # Finally remove the lock file
            os.remove(self.lockfile)

        return False

    @property
    def tmpfile(self):
        """Full path to the lockfile (without file extension)."""
        base, fname = os.path.split(self.filename)
        return os.path.join(base, "." + fname)

    @property
    def lockfile(self):
        """Full path to the lockfile (with file extension)."""
        return self.tmpfile + ".lock"
