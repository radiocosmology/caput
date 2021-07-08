"""Helper routines for profiling the CPU and IO usage of code."""

import math
import collections
import time
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import psutil

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

    profilers = ["cprofile", "pyinstrument", "psutil"]

    def __init__(
        self,
        profile: bool = True,
        profiler: str = "cprofile",
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

        if self.profiler == "cprofile":
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

        if self.profiler == "cprofile":
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


class PSUtilProfiler(psutil.Process):
    def __init__(
        self,
        use_profiler: bool = True,
        name: str = "",
        logger: Optional[logging.Logger] = None,
    ):
        """
        use_profiler
            Whether to run the profiler or not.
        name : str
            Default description of what is profiled. Used by the context manager.
        """

        self.use_profiler = use_profiler
        self._name = name
        self._start_cpu_times = {}
        self._start_memory = {}
        self._start_io = {}
        self._start_disk_io = {}
        self._start_time = {}

        self._logger = logger

        super().__init__()

        if self._use_profiler and self._logger:
            self._logger.info(f"Profiling pipeline: {self.cpu_count} cores available.")

    def __eq__(self, other):
        if not isinstance(other, psutil.Process):
            return False
        return (
            psutil.Process.__eq__(self, other)
            and self._start_cpu_times == other._start_cpu_times
            and self._start_memory == other._start_memory
            and self._start_io == other._start_io
            and self._start_disk_io == other._start_disk_io
            and self._start_time == other._start_time
        )

    def __enter__(self):
        if not self.use_profiler:
            return
        self.start(self._name)

    def __exit__(self, *args, **kwargs):
        if not self.use_profiler:
            return
        self.stop(self._name)

    def start(self, name):
        """
        Start profiling.
        Results generated when `stop` is called are based on this start time.
        Attributes
        ----------
        name : str
            Description of what is profiled. You have to pass the same str to `stop`.
        """
        self._start_time[name] = time.time()

        # Get all stats at the same time
        with self.oneshot():
            self._start_cpu_times[name] = self.cpu_times()
            self.cpu_percent()
            self._start_memory[name] = self.memory_full_info().uss
            if psutil.MACOS:
                self._start_memory[name] = psutil.disk_io_counters()
            else:
                self._start_disk_io[name] = self.io_counters()

    def stop(self, name):
        """
        Stop profiler and return results.
        `start` must be called first.
        Attributes
        ----------
        name : str
            Description of what is profiled. Has to have been passed to `start` before.
        Returns
        -------
        cpu_times : `collections.namedtuple`
            Same as `psutil.cpu_times`. Process CPU times since `start` was called in seconds.
        cpu_percent :  float
            Process CPU utilization since `start` was called as percentage. Can be >100 if multiple threads run on
            different cores. See `PSUtil.cpu_count` for available cores.
        disk_io : `collections.namedtuple`
            Same as `psutil.io_counters` (on Linux) or `psutil.disk_io_counters` (on MacOS).
            Difference since `start` was called.
        memory : str
            Difference of memory in use by this process since `start` was called. If negative,
            less memory is in use now.
        Raises
        ------
        RuntimeError
            If stop was called before start.
        """
        stop_time = time.time()

        # Get all stats at the same time
        with self.oneshot():
            cpu_times = self.cpu_times()
            cpu_percent = self.cpu_percent()
            memory = self.memory_full_info().uss
            if psutil.MACOS:
                disk_io = psutil.disk_io_counters()
            else:
                disk_io = self.io_counters()

        if name not in self._start_cpu_times:
            raise RuntimeError(
                f"PSUtilProfiler.stop was called before start for '{name}'."
            )

        # Construct results
        CPU_Times = collections.namedtuple("CPUtimes", list(cpu_times._fields))
        cpu_times = CPU_Times(*np.subtract(cpu_times, self._start_cpu_times.pop(name)))
        DiskIO = collections.namedtuple("DiskIO", list(disk_io._fields))
        disk_io = DiskIO(*np.subtract(disk_io, self._start_disk_io.pop(name)))
        memory = memory - self._start_memory.pop(name)

        def bytes2human(num):
            for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
                if abs(num) < 1024.0:
                    return f"{num:3.1f}{unit}B"
                num /= 1024.0
            return f"{num:.1f}YiB"

        memory = bytes2human(memory)

        time_s = stop_time - self._start_time.pop(name)

        if time_s < 0.1 and self._logger:
            self._logger.info(
                f"{name} ran for {time_s:.4f} < 0.1s, results might be inaccurate."
            )

        if self._logger:
            self._logger.info(
                f"{name} ran for {time_s:.4f}s\n"
                f"---------------------------------------------------------------------------------------------"
                f"\n{cpu_times}\n"
                f"Average CPU load: {cpu_percent}\n"
                f"{disk_io}\n"
                f"Change in (uss) memory: {memory}\n"
                f"=============================================================================================\n"
            )

    @property
    def cpu_count(self):
        """Number of cores available to this process."""
        return len(self.cpu_affinity())
