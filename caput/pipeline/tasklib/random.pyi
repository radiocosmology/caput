from ...algorithms.random import MultithreadedRNG
from .base import MPILoggedTask

class RandomTask(MPILoggedTask):
    seed: int
    threads: int
    _rng: MultithreadedRNG | None
    @property
    def rng(self) -> MultithreadedRNG: ...
    _local_seed: int | None
    @property
    def local_seed(self) -> int: ...
