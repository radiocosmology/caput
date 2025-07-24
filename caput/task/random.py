"""Base class for distributed random number generation."""

import zlib

import numpy as np

from .. import config
from ..lib import random
from ._core import MPILoggedTask


class RandomTask(MPILoggedTask):
    """A base class for MPI tasks that need to generate random numbers.

    Attributes
    ----------
    seed : int, optional
        Set the seed for use in the task. If not set, a random seed is generated and
        broadcast to all ranks. The seed being used is logged, to repeat a previous
        run, simply set this as the seed parameter.
    threads : int, optional
        Set the number of threads to use for the random number generator. If not
        explicitly set this will use the value of the `OMP_NUM_THREADS` environment
        variable, or fall back to four.
    """

    seed = config.Property(proptype=int, default=None)
    threads = config.Property(proptype=int, default=None)

    _rng = None

    @property
    def rng(self) -> np.random.Generator:
        """A random number generator for this task.

        .. warning::
            Initialising the RNG is a collective operation if the seed is not set,
            and so all ranks must participate in the first access of this property.

        Returns
        -------
        rng : np.random.Generator
            A deterministically seeded random number generator suitable for use in
            MPI jobs.
        """
        if self._rng is None:
            self._rng = random.MultithreadedRNG(self.local_seed, threads=self.threads)

        return self._rng

    _local_seed = None

    @property
    def local_seed(self) -> int:
        """Get the seed to be used on this rank.

        .. warning::
            Generating the seed is a collective operation if the seed is not set,
            and so all ranks must participate in the first access of this property.
        """
        if self._local_seed is None:
            if self.seed is None:
                # Use seed sequence to generate a random seed
                seed = np.random.SeedSequence().entropy
                seed = self.comm.bcast(seed, root=0)
            else:
                seed = self.seed

            self.log.info(f"Using random seed: {seed}")

            # Construct the new MPI-process and task specific seed. This mixes an
            # integer checksum of the class name with the MPI-rank to generate a new
            # hash.
            # NOTE: the slightly odd (rank + 1) is to ensure that even rank=0 mixes in
            # the class seed
            cls_name = f"{self.__module__}.{self.__class__.__name__}"
            cls_seed = zlib.adler32(cls_name.encode())
            self._local_seed = seed + (self.comm.rank + 1) * cls_seed

        return self._local_seed
