"""Profiling for the caput pipeline"""
import collections
import logging
import time

import numpy as np
import psutil

logger = logging.getLogger(__name__)


class PSUtilProfiler(psutil.Process):
    """CPU, I/O and memory profiler using psutil."""

    def __init__(self):
        self._start_cpu_times = {}
        self._start_memory = {}
        self._start_io = {}
        self._start_disk_io = {}
        self._start_time = {}

        super().__init__()

        logger.info(f"Profiling pipeline: {self.cpu_count} cores available.")

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

        if time_s < 0.1:
            logger.info(
                f"{name} ran for {time_s:.4f} < 0.1s, results might be inaccurate."
            )

        logger.info(
            f"{name} ran for {time_s:.4f}s\n"
            f"---------------------------------------------------------------------------------------------"
            f"\n{cpu_times}\n"
            f"Average CPU load: {cpu_percent}\n"
            f"{disk_io}\n"
            f"Change in (uss) memory: {memory}\n"
            f"============================================================================================="
        )

    @property
    def cpu_count(self):
        """Number of cores available to this process."""
        return len(self.cpu_affinity())
