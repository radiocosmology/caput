"""A set of miscellaneous routines that don't really fit anywhere more specific."""

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union, overload

import h5py
import numpy as np

if TYPE_CHECKING:
    from mpi4py import MPI


def vectorize(**base_kwargs):
    """Improved vectorization decorator.

    Unlike the :class:`np.vectorize` decorator this version works on methods in
    addition to functions. It also gives an actual scalar value back for any
    scalar input, instead of returning a 0-dimension array.

    Parameters
    ----------
    **base_kwargs
        Any keyword arguments accepted by :class:`np.vectorize`

    Returns
    -------
    vectorized_function : func
    """

    class _vectorize_desc:
        # See
        # http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
        # for a description of this pattern

        def __init__(self, func):
            # Save a reference to the function and set various properties so the
            # docstrings etc. get passed through
            self.func = func
            self.__doc__ = func.__doc__
            self.__name__ = func.__name__
            self.__module__ = func.__module__

        def __call__(self, *args, **kwargs):
            # This gets called whenever the wrapped function is invoked
            arr = np.vectorize(self.func, **base_kwargs)(*args, **kwargs)

            if not arr.shape:
                arr = arr.item()

            return arr

        def __get__(self, obj, objtype=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, objtype)
            return self.__class__(new_func)

    return _vectorize_desc


def scalarize(dtype=np.float64):
    """Handle scalars and other iterables being passed to numpy requiring code.

    Parameters
    ----------
    dtype : np.dtype, optional
        The output datatype. Used only to set the return type of zero-length arrays.

    Returns
    -------
    vectorized_function : func
    """

    class _scalarize_desc:
        # See
        # http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
        # for a description of this pattern

        def __init__(self, func):
            # Save a reference to the function and set various properties so the
            # docstrings etc. get passed through
            self.func = func
            self.__doc__ = func.__doc__
            self.__name__ = func.__name__
            self.__module__ = func.__module__

        def __call__(self, *args, **kwargs):
            # This gets called whenever the wrapped function is invoked

            args, scalar, empty = zip(*[self._make_array(a) for a in args])

            if all(empty):
                return np.array([], dtype=dtype)

            ret = self.func(*args, **kwargs)

            if all(scalar):
                ret = ret[0]

            return ret

        @staticmethod
        def _make_array(x):
            # Change iterables to arrays and scalars into length-1 arrays

            from skyfield import timelib

            # Special handling for the slightly awkward skyfield types
            if isinstance(x, timelib.Time):
                if isinstance(x.tt, np.ndarray):
                    scalar = False
                else:
                    scalar = True
                    x = x.ts.tt_jd(np.array([x.tt]))

            elif isinstance(x, np.ndarray):
                scalar = False

            elif isinstance(x, (list, tuple)):
                x = np.array(x)
                scalar = False

            else:
                x = np.array([x])
                scalar = True

            return (x, scalar, len(x) == 0)

        def __get__(self, obj, objtype=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, objtype)
            return self.__class__(new_func)

    return _scalarize_desc


def listize(**_):
    """Make functions that already work with `np.ndarray` or scalars accept lists.

    Also works with tuples.

    Returns
    -------
    listized_function : func
    """

    class _listize_desc:
        def __init__(self, func):
            # Save a reference to the function and set various properties so the
            # docstrings etc. get passed through
            self.func = func
            self.__doc__ = func.__doc__
            self.__name__ = func.__name__
            self.__module__ = func.__module__

        def __call__(self, *args, **kwargs):
            # This gets called whenever the wrapped function is invoked

            new_args = []
            for arg in args:
                if isinstance(arg, (list, tuple)):
                    arg = np.array(arg)
                new_args.append(arg)

            return self.func(*new_args, **kwargs)

        def __get__(self, obj, objtype=None):
            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, objtype)
            return self.__class__(new_func)

    return _listize_desc


@overload
def open_h5py_mpi(
    f: Union[str, Path, h5py.File],
    mode: str,
    use_mpi: bool = True,
    comm: Optional["MPI.Comm"] = None,
) -> h5py.File: ...


@overload
def open_h5py_mpi(
    f: h5py.Group, mode: str, use_mpi: bool = True, comm: Optional["MPI.Comm"] = None
) -> h5py.Group: ...


def open_h5py_mpi(f, mode, use_mpi=True, comm=None):
    """Ensure that we have an h5py File object.

    Opens with MPI-IO if possible.

    The returned file handle is annotated with two attributes: `.is_mpi`
    which says whether the file was opened as an MPI file and `.opened` which
    says whether it was opened in this call.

    Parameters
    ----------
    f : string, h5py.File or h5py.Group
        Filename to open, or already open file object. If already open this
        is just returned as is.
    mode : string
        Mode to open file in.
    use_mpi : bool, optional
        Whether to use MPI-IO or not (default True)
    comm : mpi4py.Comm, optional
        MPI communicator to use. Uses `COMM_WORLD` if not set.

    Returns
    -------
    fh : h5py.File
        File handle for h5py.File, with two extra attributes `.is_mpi` and
        `.opened`.
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
    elif isinstance(f, (h5py.File, h5py.Group)):
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
    comm : MPI.COMM, optional
        If present only rank=0 will create/remove the lock file and move the
        file.

    Returns
    -------
    tmp_name : str
        File name to use in the locked block.

    Examples
    --------
    >>> from . import memh5
    >>> container = memh5.BasicCont()
    >>> with lock_file('file_to_create.h5') as fname:
    ...     container.save(fname)
    ...
    """

    def __init__(self, name, preserve=False, comm=None):
        if comm is not None and not hasattr(comm, "rank"):
            raise ValueError("comm argument does not seem to be an MPI communicator.")

        self.name = name
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
                os.rename(self.tmpfile, self.name)

            # Finally remove the lock file
            os.remove(self.lockfile)

        return False

    @property
    def tmpfile(self):
        """Full path to the lockfile (without file extension)."""
        base, fname = os.path.split(self.name)
        return os.path.join(base, "." + fname)

    @property
    def lockfile(self):
        """Full path to the lockfile (with file extension)."""
        return self.tmpfile + ".lock"


# TODO: remove this. This was to support a patching of this routine to support Python 2
# that used to exist in here. This will be removed when all other repos are changed to
# use the version from `inspect`
def getfullargspec(*args, **kwargs):
    """See `inspect.getfullargspec`.

    This is a Python 2 patch that will be removed.
    """
    import inspect
    import warnings

    warnings.warn(
        "This patch to support Python 2 is no longer needed and will be removed.",
        DeprecationWarning,
    )

    return inspect.getfullargspec(*args, **kwargs)


def import_class(class_path):
    """Import class dynamically from a string.

    Parameters
    ----------
    class_path : str
        Fully qualified path to the class. If only a single component, look up in the
        globals.

    Returns
    -------
    class : class object
        The class we want to load.
    """
    path_split = class_path.split(".")
    module_path = ".".join(path_split[:-1])
    class_name = path_split[-1]
    if module_path:
        m = importlib.import_module(module_path)
        task_cls = getattr(m, class_name)
    else:
        task_cls = globals()[class_name]
    return task_cls
