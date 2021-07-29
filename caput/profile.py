"""Helper routines for profiling the CPU and IO usage of code."""

import math
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

    profilers = ["cprofile", "pyinstrument"]

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
    """A context manager that profiles using psutil.

    To access the profiling data the context manager object
    must be created and bound to a variable *before* the *with* statement.

    Dumps results into csv file, one for each rank.

    >>> p = PSUtilProfiler(label='task-label')
    >>> with p:
    ...     print("do some task in here")
    do some task in here
    >>> print(p.usage)  #doctest: +ELLIPSIS
    {...}

    `start` and `stop` can be used to use the PSUtilProfiler outside of cotnext management.

    >>> p = PSUtilProfiler(label='task-label')
    >>> p.start()
    >>> print('do some task in here')
    do some task in here
    >>> p.stop()
    >>> print(p.usage) #doctest: +ELLIPSIS
    {...}

    Parameters
    ----------
    use_profiler : bool
        Whether to run the profiler or not.
    label : str
        Default description of what is being profiled.
    logger
        If a logging object is passed the values of the IO done counters are logged
        at INFO level.
    comm
        An optional MPI communicator. This is only used for labelling the output files.
    path
        The optional directory path under which to write the profile csvs.  If not set use the
        current directory.
    """

    def __init__(
        self,
        use_profiler: bool = True,
        label: str = "",
        logger: Optional[logging.Logger] = None,
        comm: Optional["mpiutil.MPI.IntraComm"] = None,
        path: Optional[os.PathLike] = None,
    ):
        self._use_profiler = use_profiler
        self._label = label
        self._usage = {}
        self._logger = logger
        self.comm = comm

        if self.comm is None:
            rank = mpiutil.rank
        else:
            rank = self.comm.rank

        if path is None:
            self.path = Path.cwd()
        else:
            if not self.path.is_dir() or not self.path.exists():
                raise ValueError(
                    f"Make sure {self.path} passed to PSUtillProfiler is a directory that exists."
                )
            self.path = Path(path)

        self.path = self.path / f"perf_{rank}.csv"

        self._start_cpu_times = None
        self._start_memory = None
        self._start_disk_io = None
        self._start_time = None

        super().__init__()

        if self._use_profiler and not self.path.exists():

            import csv

            with open(self.path, mode="w") as fp:

                colnames = [
                    "task_name",
                    "time_s",
                    "cpu_times_user",
                    "cpu_times_system",
                    "cpu_times_children_user",
                    "cpu_times_children_system",
                    "cpu_times_iowait",
                    "average_cpu_load_percent",
                    "disk_io_read_count",
                    "disk_io_write_count",
                    "disk_io_read_bytes",
                    "disk_io_write_bytes",
                    "disk_io_read_chars",
                    "disk_io_write_chars",
                    "memory_change_uss",
                    "current_available_memory_bytes",
                    "current_total_used_memory_bytes",
                ]
                cw = csv.writer(fp)
                cw.writerow(colnames)

        if self._use_profiler and self._logger:
            self._logger.info(f"Profiling pipeline: {self.cpu_count} cores available.")

    def __eq__(self, other):
        if not isinstance(other, psutil.Process):
            return False
        return (
            psutil.Process.__eq__(self, other)
            and self._start_cpu_times == other._start_cpu_times
            and self._start_memory == other._start_memory
            and self._start_disk_io == other._start_disk_io
            and self._start_time == other._start_time
        )

    def __enter__(self):
        if not self._use_profiler:
            return
        self.start()

    def __exit__(self, *args, **kwargs):
        if not self._use_profiler:
            return
        self.stop()

    def start(self):
        """
        Start profiling.

        Results generated when `stop` is called are based on this start time.

        """
        self._start_time = time.time()

        # Get all stats at the same time
        with self.oneshot():
            self._start_cpu_times = self.cpu_times()
            self.cpu_percent()
            self._start_memory = self.memory_full_info().uss
            if psutil.MACOS:
                self._start_memory = psutil.disk_io_counters()
            else:
                self._start_disk_io = self.io_counters()

    def stop(self):
        """
        Stop profiler. Dump results to csv file and/or log and/or set results on self.usage.

        `start` must be called first.

        Returns
        -------
        Sets usage dictionary on `self` with the following attributes, under 'label' key.

        cpu_times : `dict`
            dict version of `psutil.cpu_times`. Process CPU times since `start` was called in seconds.
        cpu_percent :  float
            Process CPU utilization since `start` was called as percentage. Can be >100 if multiple threads run on
            different cores. See `PSUtil.cpu_count` for available cores.
        disk_io : `dict`
            dict version of `psutil.io_counters` (on Linux) or `psutil.disk_io_counters` (on MacOS).
            Difference since `start` was called.
        memory : str
            Difference of memory in use by this process since `start` was called. If negative,
            less memory is in use now.
        used_memory : str
            Current used memory at the time of the task's end.
        available_memory : str
            Current memory available to the system at the time of the task's end.

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
            used_memory = psutil.virtual_memory().used
            available_memory = psutil.virtual_memory().available
            if psutil.MACOS:
                disk_io = psutil.disk_io_counters()
            else:
                disk_io = self.io_counters()

        if self._start_cpu_times is None:
            raise RuntimeError(f"PSUtilProfiler.stop was called before start'.")

        # Construct results
        self._usage = {"task_name": self._label}

        cpu_times_arr = np.subtract(cpu_times, self._start_cpu_times)
        disk_io_arr = np.subtract(disk_io, self._start_disk_io)

        cpu_times = {
            k: v for (k, v) in zip(cpu_times._fields, cpu_times_arr)
        }  # contain results in dictionary
        disk_io = {
            k: v for (k, v) in zip(disk_io._fields, disk_io_arr)
        }  # contain results in dictionary

        memory = memory - self._start_memory

        self._usage["cpu_times"] = cpu_times
        self._usage["cpu_percent"] = cpu_percent
        self._usage["disk_io"] = disk_io

        def bytes2human(num):
            for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
                if abs(num) < 1024.0:
                    return f"{num:3.1f}{unit}B"
                num /= 1024.0
            return f"{num:.1f}YiB"

        self._usage["memory"] = memory
        self._usage["used_memory"] = used_memory
        self._usage["available_memory"] = available_memory

        time_s = stop_time - self._start_time

        self._usage["time_s"] = time_s

        if time_s < 0.1 and self._logger:
            self._logger.info(
                f"{self._label} ran for {time_s:.4f} < 0.1s, results might be inaccurate.\n"
            )

        if self._logger:
            self._logger.info(f"{self._label} ran for {time_s:.4f}s")
            self._logger.info(f"{cpu_times}")
            self._logger.info(f"average CPU load: {cpu_percent}")
            self._logger.info(f"{disk_io}")
            self._logger.info(f"change in (uss) memory: {bytes2human(memory)}")
            self._logger.info(
                f"current available memory: {bytes2human(available_memory)}"
            )
            self._logger.info(f"current total used memory: {bytes2human(used_memory)}")

        with open(self.path, mode="a", newline="") as fp:
            import csv

            cw = csv.writer(fp)
            cw.writerow(
                [
                    self._label,
                    time_s,
                    cpu_times["user"],
                    cpu_times["system"],
                    cpu_times["children_user"],
                    cpu_times["children_system"],
                    cpu_times["iowait"],
                    cpu_percent,
                    disk_io["read_count"],
                    disk_io["write_count"],
                    disk_io["read_bytes"],
                    disk_io["write_bytes"],
                    disk_io["read_chars"],
                    disk_io["write_chars"],
                    memory,
                    available_memory,
                    used_memory,
                ]
            )

    @property
    def cpu_count(self):
        """Number of cores available to this process."""
        return len(self.cpu_affinity())

    @property
    def usage(self):
        """The memory and cpu usage within the block."""
        return self._usage.copy()
