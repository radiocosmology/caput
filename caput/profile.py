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
