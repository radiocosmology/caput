"""Helper routines for profiling the CPU and IO usage of code."""

import math
import logging
import os
from pathlib import Path
from typing import Optional

from . import mpiutil


class Profiler:
    """A context manager to profile a block of code using various profilers.

    Parameters
    ----------
    profile
        Whether to run the profiler or not.
    profiler
        Which profiler to run. Currently `cProfile` and `pyinstrument` are supported.
    comm
        An optional MPI communicator. This is only used for labelling the output files.
    path
        The optional path under which to write the profiles.  If not set use the
        current directory.
    """

    profilers = ["cProfile", "pyinstrument"]

    def __init__(
        self,
        profile: bool = True,
        profiler: str = "cProfile",
        comm: Optional["mpiutil.MPI.IntraComm"] = None,
        path: Optional[os.PathLike] = None,
    ):
        self.profile = profile

        if profiler not in self.profilers:
            raise ValueError(f"Unsupported profiler: {profiler}")

        self.profiler = profiler
        self.comm = comm
        self._pr = None

        if path is None:
            self.path = Path.cwd()
        else:
            self.path = Path(path)

    def __enter__(self):

        if not self.profile:
            return

        if self.profiler == "cProfile":
            import cProfile

            self._pr = cProfile.Profile()
            self._pr.enable()

        elif self.profiler == "pyinstrument":
            import pyinstrument

            self._pr = pyinstrument.Profiler()
            self._pr.start()

    def __exit__(self, *args, **kwargs):

        if not self.profile:
            return

        if self.comm is None:
            rank = mpiutil.rank
            size = mpiutil.size
        else:
            rank = self.comm.rank
            size = self.comm.size

        rank_length = int(math.log10(size)) + 1

        if self.profiler == "cProfile":
            self._pr.disable()
            self._pr.dump_stats(f"profile_{rank:0{rank_length}}.prof")

        elif self.profiler == "pyinstrument":
            self._pr.stop()
            with open(f"profile_{rank:0{rank_length}}.txt", "w") as fh:
                fh.write(self._pr.output_text(unicode=True))


class IOUsage:
    """A context manager that gives the amount of IO done.

    To access the IO usage the context manager object must be created and bound to a
    variable *before* the *with* statement.

    >>> u = IOUsage()
    >>> with u:
    ...     print("do some IO in here")
    do some IO in here
    >>> print(u.usage)  #doctest: +ELLIPSIS
    {...}

    Parameters
    ----------
    logger
        If a logging object is passed the values of the IO done counters are logged
        at INFO level.
    """

    _start = None

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger
        self._usage = {}

    @staticmethod
    def _get_io():  # pylint: disable=no-self-use
        # Get the cumulative IO performed

        import psutil

        if psutil.MACOS:
            d = psutil.disk_io_counters()
        else:
            p = psutil.Process()
            d = p.io_counters()

        # There doesn't seem to be a public API for this
        return d._asdict()

    @staticmethod
    def _units(key):  # pylint: disable=no-self-use
        # Try and infer the units for this particular counter

        suffix = key.split("_")[-1]

        if suffix == "count":
            return ""
        elif suffix == "time":
            return "ms"
        elif suffix == "bytes":
            return "bytes"
        else:
            return ""

    def __enter__(self):
        self._start = self._get_io()

    def __exit__(self, *args, **kwargs):

        f = self._get_io()

        for name in f:
            self._usage[name] = f[name] - self._start[name]

        if self._logger:
            for key, value in self.usage.items():
                self._logger.info(f"IO usage({key}): {value} {self._units(key)}")

    @property
    def usage(self):
        """The IO usage within the block."""
        return self._usage.copy()
