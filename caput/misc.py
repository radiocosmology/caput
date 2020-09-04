"""
A set of miscellaneous routines that don't really fit anywhere more specific.

**Functions**

.. autosummary::
    :toctree: generated/

    vectorize
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614
from future.utils import PY3
from past.builtins import basestring

# === End Python 2/3 compatibility

import collections
import importlib
import inspect
import os

import numpy as np


def vectorize(**base_kwargs):
    """An improved vectorization decorator.

    Unlike the :class:`np.vectorize` decorator this version works on methods in
    addition to functions. It also gives an actual scalar value back for any
    scalar input, instead of returning a 0-dimension array.

    Parameters
    ----------
    **kwargs
        Any keyword arguments accepted by :class:`np.vectorize`

    Returns
    -------
    vectorized_function : func
    """

    class _vectorize_desc(object):
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

            if not len(arr.shape):
                arr = arr.item()

            return arr

        def __get__(self, obj, type=None):

            # As a descriptor, this gets called whenever this is used to wrap a
            # function, and simply binds it to the instance

            if obj is None:
                return self

            new_func = self.func.__get__(obj, type)
            return self.__class__(new_func)

    return _vectorize_desc


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

    if isinstance(f, basestring):
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
        raise ValueError("Did not receive a h5py.File or filename")

    fh.is_mpi = fh.file.driver == "mpio"

    return fh


class lock_file(object):
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

        from . import mpiutil

        if comm is not None and not hasattr(comm, "rank"):
            raise ValueError("comm argument does not seem to be an MPI communicator.")

        self.name = name
        self.rank0 = mpiutil.rank0 if comm is None else comm.rank == 0
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
        base, fname = os.path.split(self.name)
        return os.path.join(base, "." + fname)

    @property
    def lockfile(self):
        return self.tmpfile + ".lock"


# Wrapper for getfullargspec
def getfullargspec(f):
    """Python 2 compatible implementation of `inspect.getfullargspec`.

    Parameters
    ----------
    f : function
        Callable to inspect.

    Returns
    -------
    fullargspec : namedtuple
        Named tuple with various fields. See `inspect.getfullargspec`.
    """

    if PY3:
        return inspect.getfullargspec(f)
    else:
        argspec = inspect.getargspec(f)

        fullargspec_type = collections.namedtuple(
            "FullArgSpec",
            [
                "args",
                "varargs",
                "varkw",
                "defaults",
                "kwonlyargs",
                "kwonlydefaults",
                "annotations",
            ],
        )

        return fullargspec_type(
            args=argspec.args,
            varargs=argspec.varargs,
            defaults=argspec.defaults,
            varkw=argspec.keywords,  # This is the equivalent field
            kwonlyargs=[],  # KW only args do not exist in Python 2
            kwonlydefaults=[],
            annotations=[],  # Annotations do not exist in Python 2
        )


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
