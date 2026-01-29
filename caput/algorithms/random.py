"""Multithreaded random number generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ClassVar

    import numpy.typing as npt


import concurrent.futures
import os

import numpy as np

_rng: np.random.Generator | None = None
_default_bitgen = np.random.SFC64


__all__ = ["MultithreadedRNG", "default_rng"]


def default_rng() -> np.random.Generator:
    """Returns an instance of the default random number generator to use.

    This creates a randomly seeded generator using the fast SFC64 bit generator
    underneath. This is only initialised on the first call, subsequent calls will
    return the same Generator.

    Returns
    -------
    rng : :py:class:`~np.random.Generator`
        The default random number generator instance.
    """
    global _rng

    if _rng is None:
        _rng = np.random.Generator(_default_bitgen())

    return _rng


class MultithreadedRNG(np.random.Generator):
    """A multithreaded random number generator.

    This wraps specific methods to allow generation across multiple threads. The
    following methods are supported:

    - random
    - integers
    - uniform
    - normal
    - standard_normal
    - poisson
    - power

    Parameters
    ----------
    seed : int | None, optional
        The seed to use.
    threads : int | None, optional
        The number of threads to use. If not set, this tries to get the number from the
        `OMP_NUM_THREADS` environment variable, or just uses 4 if that is also not set.
    bitgen : :py:class:`~np.random.BitGenerator` | None, optional
        The BitGenerator to use, if not set this uses `_default_bitgen`.
    """

    _parallel_threshold: int = 1000

    # The methods to generate parallel versions for. This table is:
    # method name, number of initial parameter arguments, default data type, if there is
    # a dtype argument, and if there is an out argument. See `_build_method` for
    # details.
    PARALLEL_METHODS: ClassVar[dict[str, tuple]] = {
        "random": (0, np.float64, True, True),
        "integers": (2, np.int64, True, False),
        "uniform": (2, np.float64, False, False),
        "normal": (2, np.float64, False, False),
        "standard_normal": (0, np.float64, True, True),
        "poisson": (1, np.float64, False, False),
        "power": (1, np.float64, False, False),
    }

    def __init__(
        self, seed: int | None = None, threads: int | None = None, bitgen=None
    ) -> None:
        if bitgen is None:
            bitgen = _default_bitgen

        # Initialise this object with the given seed. This allows methods that don't
        # have multithreaded support to work
        super().__init__(bitgen(seed))

        if threads is None:
            threads = int(os.environ.get("OMP_NUM_THREADS", 4))

        # Initialise the parallel thread pool
        self._threads: int = threads
        self._random_generators: list = [
            np.random.Generator(bitgen(seed=s))
            for s in np.random.SeedSequence(seed).spawn(threads)
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(threads)

        # Create the methods and attach them to this instance.
        for method, spec in self.PARALLEL_METHODS.items():
            setattr(self, method, self._build_method(method, *spec))

    def _build_method(
        self,
        name: str,
        nparam: int,
        defdtype: npt.DTypeLike,
        has_dtype: bool,
        has_out: bool,
    ) -> Callable:
        """Build a method for generating random numbers from a given distribution.

        As the underlying methods are in Cython they can't be adequately introspected
        and so we need to provide information about the signature.

        Parameters
        ----------
        name : str
            The name of the generation method in `np.random.Generator`.
        nparam : int
            The number of distribution parameters that come before the `size` argument.
        defdtype : dtype
            The default datatype used if none is explicitly supplied.
        has_dtype : bool
            Does the underlying method have a dtype argument?
        has_out : bool
            Does the underlying method have an `out` parameter for directly filling an
            array.

        Returns
        -------
        parallel_method : callable
            A method for generating in parallel.
        """
        method = getattr(np.random.Generator, name)

        def _call(*args, **kwargs):
            orig_args = list(args)
            orig_kwargs = dict(kwargs)

            # Try and get the size
            if len(args) > nparam:
                size = args[nparam]
            elif "size" in kwargs:
                size = kwargs.pop("size")
            else:
                size = None

            # Try and get an out argument
            if has_out and "out" in kwargs:
                out = kwargs.pop("out")
                size = out.shape
            else:
                out = None

            # Try to figure out the dtype so we can pre-allocate the array for filling
            if has_dtype and len(args) > nparam + 1:
                dtype = args[nparam + 1]
            elif has_dtype and "dtype" in kwargs:
                dtype = kwargs.pop("dtype")
            else:
                dtype = defdtype

            # Trim any excess positional arguments
            args = args[:nparam]

            # Check that all the parameters are scalar
            all_scalar = all(np.isscalar(arg) for arg in args)

            # Check that any remaining kwargs (assumed to be parameters are also scalar)
            all_scalar &= all(np.isscalar(arg) for arg in kwargs.values())

            # If neither size nor out is set we can't parallelise this so just call
            # directly.
            # Additionally if the distribution arguments are not scalars there may be
            # some complex broadcasting required, so we also drop out if that is true.
            if (size is None and out is None) or not all_scalar:
                return method(self, *orig_args, **orig_kwargs)

            flatsize = np.prod(size)

            # If the total size is too small, then just call directly
            if flatsize < self._parallel_threshold:
                return method(self, *orig_args, **orig_kwargs)

            # Create the output array if required
            if out is None:
                out = np.empty(size, dtype)

            # Figure out how to split up the array
            step = int(np.ceil(flatsize / self._threads))

            # A worker method for each thread to fill its part of the array with the
            # random numbers
            def _fill(gen: np.random.Generator, local_array: np.ndarray) -> None:
                if has_dtype:
                    kwargs["dtype"] = dtype
                if has_out:
                    if out.dtype != dtype:
                        raise TypeError(
                            f"Output array of type f{local_array.dtype} does not "
                            f"match dtype argument {dtype}."
                        )
                    method(gen, *args, **kwargs, out=local_array)
                else:
                    local_array[:] = method(
                        gen,
                        *args,
                        **kwargs,
                        size=len(local_array),
                    )

            # Generate the numbers with each worker thread
            futures = [
                self._executor.submit(
                    _fill,
                    self._random_generators[i],
                    out.ravel()[(i * step) : ((i + 1) * step)],
                )
                for i in range(self._threads)
            ]
            concurrent.futures.wait(futures)

            for ii, future in enumerate(futures):
                if (e := future.exception()) is not None:
                    raise RuntimeError(
                        f"An exception occurred in thread {ii} (and maybe others)."
                    ) from e

            return out

        # Copy over the docstring for the method
        _call.__doc__ = "Multithreaded version.\n" + method.__doc__

        return _call

    def __del__(self) -> None:
        self._executor.shutdown(False)
