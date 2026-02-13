"""Base Task for distributed random number generation."""

import zlib

import numpy as np

from ... import config
from ...algorithms import random
from .base import MPILoggedTask


class RandomTask(MPILoggedTask):
    """A base class for MPI tasks that need to generate random numbers.

    Attributes
    ----------
    seed : int
        Set the seed for use in the task. If not set, a random seed is generated and
        broadcast to all ranks. The seed being used is logged, to repeat a previous
        run, simply set this as the seed parameter.
    threads : int
        Set the number of threads to use for the random number generator. If not
        explicitly set this will use the value of the `OMP_NUM_THREADS` environment
        variable, or fall back to four.
    """

    seed = config.Property(proptype=int, default=None)
    threads = config.Property(proptype=int, default=None)

    def __init__(self) -> None:
        """Generate and set a new random seed for this task.

        This will generate a new random seed on rank 0, broadcast it to all ranks,
        and set the `_local_seed` attribute accordingly.

        .. warning::
            Generating the seed is a collective operation so all ranks must
            participate in the call to `__init__`.
        """
        # Initialize the base class
        super().__init__()

        seed = self.seed

        if seed is None:
            # Generate a random seed on rank 0 and broadcast it to all ranks
            seed = np.random.SeedSequence().entropy
            seed = self.comm.bcast(seed, root=0)

        self.log.info(f"Using random seed: {seed}")
        # Construct the new MPI-process and task specific seed. This mixes an integer
        # checksum of the class name with the MPI-rank to generate a new hash.
        # NOTE: the slightly odd (rank + 1) is to ensure that even rank=0 mixes in
        # the class seed
        cls_name = f"{self.__module__}.{self.__class__.__name__}"
        cls_seed = zlib.adler32(cls_name.encode())

        self._local_seed = seed + (self.comm.rank + 1) * cls_seed
        # Don't set the random number generator until the first
        # time it gets called
        self._rng = None

    @property
    def rng(self):
        """A random number generator for this task.

        .. warning::
            Initialising the RNG is a collective operation if the seed is not set,
            and so all ranks must participate in the first access of this property.

        Returns
        -------
        rng : MultiThreadedRNG
            A deterministically seeded random number generator suitable for use in
            MPI jobs.
        """
        if self._rng is None:
            self._rng = random.MultithreadedRNG(self.local_seed, threads=self.threads)

        return self._rng

    @property
    def local_seed(self):
        """Get the seed to be used on this rank.

        Returns
        -------
        rank_local_seed : int
            Seed local to this rank.
        """
        return self._local_seed
