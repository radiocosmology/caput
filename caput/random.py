"""Utilities for drawing random numbers."""

import concurrent.futures
import contextlib
import os
from collections.abc import Callable
from typing import ClassVar

import numpy as np

_rng = None
_default_bitgen = np.random.SFC64


def default_rng():
    """Returns an instance of the default random number generator to use.

    This creates a randomly seeded generator using the fast SFC64 bit generator
    underneath. This is only initialise on the first call, subsequent calls will
    return the same Generator.

    Returns
    -------
    rng : np.random.Generator
    """
    global _rng

    if _rng is None:
        _rng = np.random.Generator(_default_bitgen())

    return _rng


def complex_normal(loc=0.0, scale=1.0, size=None, dtype=None, rng=None, out=None):
    """Get a set of complex normal variables.

    By default generate standard complex normal variables.

    Parameters
    ----------
    size : tuple
        Shape of the array of variables.
    loc : np.ndarray or complex float, optional
        The mean of the complex output. Can be any array which broadcasts against
        an array of `size`.
    scale : np.ndarray or float, optional
        The standard deviation of the complex output. Can be any array which
        broadcasts against an array of `size`.
    dtype : {np.complex64, np.complex128}, optional
        Output datatype.
    rng : np.random.Generator, optional
        Generator object to use.
    out : np.ndarray[shape], optional
        Array to place output directly into.

    Returns
    -------
    out : np.ndarray[shape]
        Complex gaussian variates.
    """
    # Validate/set size argument
    if size is None and out is None:
        size = (1,)
    elif out is not None and size is None:
        size = out.shape
    elif out is not None and size is not None and out.shape != size:
        raise ValueError(
            f"Shape of output array ({out.shape}) != size argument ({size}"
        )

    # Validate/set dtype argument
    if dtype is None and out is None:
        dtype = np.complex128
    elif dtype is None and out is not None:
        dtype = out.dtype.type
    elif out is not None and dtype is not None and out.dtype.type != dtype:
        raise ValueError(
            f"Dtype of output array ({out.dtype.type}) != dtype argument ({dtype}"
        )

    if rng is None:
        rng = default_rng()

    _type_map = {
        np.complex64: np.float32,
        np.complex128: np.float64,
    }

    if dtype not in _type_map:
        raise ValueError(
            f"Only dtype must be complex64 or complex128. Got dtype={dtype}."
        )

    if out is None:
        out = np.ndarray(size, dtype=dtype)

    # Fill the complex array by creating a real type view of it
    rtype = _type_map[dtype]
    rsize = size[:-1] + (size[-1] * 2,)
    rng.standard_normal(rsize, dtype=rtype, out=out.view(rtype))

    # Use inplace ops for scaling and adding to avoid intermediate arrays
    rscale = scale / 2**0.5
    out *= rscale

    # Don't bother with the additions if not needed
    if np.any(loc != 0.0):
        out += loc

    return out


def standard_complex_normal(shape, dtype=None, rng=None):
    """Get a set of standard complex normal variables.

    Parameters
    ----------
    shape : tuple
        Shape of the array of variables.
    dtype : {np.complex64, np.complex128}, optional
        Output datatype.
    rng : np.random.Generator, optional
        Generator object to use.

    Returns
    -------
    out : np.ndarray[shape]
        Complex gaussian variates.
    """
    return complex_normal(size=shape, dtype=dtype, rng=rng)


def standard_complex_wishart(m, n, rng=None):
    """Draw a standard Wishart matrix.

    Parameters
    ----------
    m : integer
        Number of variables (i.e. size of matrix).
    n : integer
        Number of measurements the covariance matrix is estimated from.
    rng : np.random.Generator, optional
        Random number generator to use.

    Returns
    -------
    B : np.ndarray[m, m]
    """
    if rng is None:
        rng = default_rng()

    # Fill in normal variables in the lower triangle
    T = np.zeros((m, m), dtype=np.complex128)
    T[np.tril_indices(m, k=-1)] = (
        rng.standard_normal(m * (m - 1) // 2)
        + 1.0j * rng.standard_normal(m * (m - 1) // 2)
    ) / 2**0.5

    # Gamma variables on the diagonal
    for i in range(m):
        T[i, i] = rng.gamma(n - i) ** 0.5

    # Return the square to get the Wishart matrix
    return np.dot(T, T.T.conj())


def complex_wishart(C, n, rng=None):
    """Draw a complex Wishart matrix.

    Parameters
    ----------
    C : np.ndarray[:, :]
        Expected covaraince matrix.
    n : integer
        Number of measurements the covariance matrix is estimated from.
    rng : np.random.Generator, optional
        Random number generator to use.

    Returns
    -------
    C_samp : np.ndarray
        Sample covariance matrix.
    """
    import scipy.linalg as la

    # Find Cholesky of C
    L = la.cholesky(C, lower=True)

    # Generate a standard Wishart
    A = standard_complex_wishart(C.shape[0], n, rng=rng)

    # Transform to get the Wishart variable
    return np.dot(L, np.dot(A, L.T.conj()))


@contextlib.contextmanager
def mpi_random_seed(seed, extra=0, gen=None):
    """Use a specific random seed and return to the original state on exit.

    This is designed to work for MPI computations, incrementing the actual seed of
    each process by the MPI rank. Overall each process gets the numpy seed:
    `numpy_seed = seed + mpi_rank + 4096 * extra`. This can work for either the
    global numpy.random context or for new np.random.Generator.


    Parameters
    ----------
    seed : int
        Base seed to set. If seed is :obj:`None`, re-seed randomly.
    extra : int, optional
        An extra part of the seed, which should be changed for calculations
        using the same seed, but that want different random sequences.
    gen: :class: `Generator`
        A RandomGen bit_generator whose internal seed state we are going to
        influence.

    Yields
    ------
    If we are setting the numpy.random context, nothing is yielded.

    :class: `Generator`
        If we are setting the RandomGen bit_generator, it will be returned.
    """
    import warnings

    from . import mpiutil

    warnings.warn(
        "This routine has fatal flaws. Try using `RandomTask` instead",
        category=DeprecationWarning,
    )

    # Just choose a random number per process as the seed if nothing was set.
    if seed is None:
        seed = np.random.randint(2**30)

    # Construct the new process specific seed
    new_seed = seed + mpiutil.rank + 4096 * extra
    np.random.seed(new_seed)

    # we will be setting the numpy.random context
    if gen is None:
        # Copy the old state for restoration later.
        old_state = np.random.get_state()

        # Enter the context block, and reset the state on exit.
        try:
            yield
        finally:
            np.random.set_state(old_state)

    # we will be setting the randomgen context
    else:
        # Copy the old state for restoration later.
        old_state = gen.state

        # Enter the context block, and reset the state on exit.
        try:
            yield gen
        finally:
            gen.state = old_state


class MultithreadedRNG(np.random.Generator):
    """A multithreaded random number generator.

    This wraps specific methods to allow generation across multiple threads. See
    `PARALLEL_METHODS` for the specific methods wrapped.

    Parameters
    ----------
    seed
        The seed to use.
    nthreads
        The number of threads to use. If not set, this tries to get the number from the
        `OMP_NUM_THREADS` environment variable, or just uses 4 if that is also not set.
    bitgen
        The BitGenerator to use, if not set this uses `_default_bitgen`.
    """

    _parallel_threshold = 1000

    # The methods to generate parallel versions for. This table is:
    # method name, number of initial parameter arguments, default data type, if there is
    # a dtype argument, and if there is an out argument. See `_build_method` for
    # details.
    PARALLEL_METHODS: ClassVar = {
        "random": (0, np.float64, True, True),
        "integers": (2, np.int64, True, False),
        "uniform": (2, np.float64, False, False),
        "normal": (2, np.float64, False, False),
        "standard_normal": (0, np.float64, True, True),
        "poisson": (1, np.float64, False, False),
        "power": (1, np.float64, False, False),
    }

    def __init__(
        self,
        seed: int | None = None,
        threads: int | None = None,
        bitgen: np.random.BitGenerator | None = None,
    ):
        if bitgen is None:
            bitgen = _default_bitgen

        # Initialise this object with the given seed. This allows methods that don't
        # have multithreaded support to work
        super().__init__(bitgen(seed))

        if threads is None:
            threads = int(os.environ.get("OMP_NUM_THREADS", 4))

        # Initialise the parallel thread pool
        self._threads = threads
        self._random_generators = [
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
        defdtype: np.dtype,
        has_dtype: bool,
        has_out: bool,
    ) -> Callable:
        """Build a method for generating random numbers from a given distribution.

        As the underlying methods are in Cython they can't be adequately introspected
        and so we need to provide information about the signature.

        Parameters
        ----------
        name
            The name of the generation method in `np.random.Generator`.
        nparam
            The number of distribution parameters that come before the `size` argument.
        defdtype
            The default datatype used if non is explicitly supplied.
        has_dtype
            Does the underlying method have a dtype argument?
        has_out
            Does the underlying method have an `out` parameter for directly filling an
            array.

        Returns
        -------
        parallel_method
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

    def __del__(self):
        self._executor.shutdown(False)
