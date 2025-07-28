"""Fast FFT implementation using FFTW.

This module adds some minor abstraction to use pyfftw in a way
which seems to be faster than using the `pyfftw.builders` interface,
and uses the same api as `scipy.fft` and `numpy.fft`.

Only forward and reverse complex->complex transforms
are currently supported.

Examples
--------
The core of this module is the :class:`FFT`, which essentially just
abstracts the :class:`pyfftw:FFTW` in the simplest way.

>>> import numpy as np
>>> from caput.fft import fftw
>>>
>>> shape = (24, 50)
>>> x = np.random.rand(*shape) + 1j * np.random.rand(*shape)
>>>
>>> fftobj = fftw.FFT(x.shape, x.dtype, axes=-1)
>>>
>>> X = fftobj.fft(x)
>>> xi = fftobj.ifft(X)
>>>
>>> np.allclose(x, xi)
True

The direct API can also be used, although it is slower when doing repeated
transforms for arrays of the same shape and type because a new :class:`FFT`
has to be created each time.

References
----------
.. https://pyfftw.readthedocs.io
.. http://www.fftw.org

Classes
=======
- :py:class:`FFT`

Functions
=========
- :py:meth:`fft`
- :py:meth:`ifft`
- :py:meth:`fftconvolve`
- :py:meth:`fftwindow`
"""

from __future__ import annotations

# NOTE: Due to a bug in pyfftw, it needs to be imported before
# numpy in order to avoid some sort of namespace collision.
# If you run into a RuntimeError when trying to use the `FFT`
# class, make sure that your environment imports `pyfftw`
# before `numpy`. Hopefully this will be fixed soon.
try:
    import pyfftw
except ImportError as exc:
    raise ImportError(
        "`pyfftw` is not installed. Install `pyfftw` via `caput[fftw]`."
    ) from exc

import numpy as np

from ..util import mpitools


class FFT:
    """Faster FFTs with FFTW."""

    def __init__(
        self,
        shape: tuple,
        dtype: type,
        axes: None | int | tuple = None,
        forward: bool = True,
        backward: bool = True,
    ):
        """Create FFTW objects for repeat use.

        This implementation is most efficient when used to repeatedly
        apply ffts to arrays with the same shape and dtype, because a
        single, highly optimised pathway can be used with a single
        initialisation.

        Even for a single use this will typically
        be faster than the `scipy.fft` or `numpy.fft` implementations,
        especially when multiple cores can be used.

        Parameters
        ----------
        shape
            The shape of the arrays to initialise for
        dtype
            Datatype to create a pathway for. At the moment, only
            complex -> complex or real -> real are supported. The
            `pyfftw` implementation of the real -> real backward
            transform will destroy the input array
        axes
            Axes over which to apply the fft. Default is all axes.
        forward
            If true, initialise the forward fft. Default is True.
        backward
            If true, initialise the backward fft. Default is True.
        """
        if not np.issubdtype(dtype, np.complexfloating):
            raise TypeError("Only complex->complex transforms are currently supported.")

        self._nsimd = pyfftw.simd_alignment
        ncpu = mpitools.cpu_count()
        flags = ("FFTW_MEASURE",)

        if axes is None:
            axes = tuple(range(len(shape)))
        elif isinstance(axes, int):
            axes = (axes,)

        # Store fft params
        self._params = {
            "ncpu": ncpu,
            "simd_alignment": self._nsimd,
            "shape": shape,
            "dtype": dtype,
            "axes": axes,
            "flags": flags,
        }

        fftargs = {
            "input_array": pyfftw.empty_aligned(shape, dtype, n=self._nsimd),
            "output_array": pyfftw.empty_aligned(shape, dtype, n=self._nsimd),
            "axes": axes,
            "flags": flags,
            "threads": ncpu,
        }

        if forward:
            self._fft = pyfftw.FFTW(direction="FFTW_FORWARD", **fftargs)

        if backward:
            self._ifft = pyfftw.FFTW(direction="FFTW_BACKWARD", **fftargs)

    @property
    def params(self):
        """Display the parameters of this FFT.

        Returns
        -------
        params: dict
            ncpu, simd alignment, shape, dtype, axes, and flags
            used by this FFT object.
        """
        return self._params

    def fft(self, x):
        """Perform a forward FFT.

        Parameters
        ----------
        x : np.ndarray
            Input array, must match the dtype specified
            at creation

        Returns
        -------
        fft : np.ndarray
            DFT of the input array over specified axes
        """
        try:
            return self._fft(
                input_array=x,
                output_array=pyfftw.empty_aligned(x.shape, x.dtype, n=self._nsimd),
            )
        except AttributeError:
            raise RuntimeError("Forward fft not initialised.")

    def ifft(self, x):
        """Perform a backward FFT.

        When performing the backward real -> real IFFT,
        the input array is destroyed.

        Parameters
        ----------
        x : np.ndarray
            Input array, must match the dtype specified
            at creation

        Returns
        -------
        fft : np.ndarray
            IDFT of the input array over specified axes
        """
        try:
            return self._ifft(
                input_array=x,
                output_array=pyfftw.empty_aligned(x.shape, x.dtype, n=self._nsimd),
            )
        except AttributeError:
            raise RuntimeError("Backward fft not initialised.")

    def fftconvolve(self, in1, in2):
        """Convolve two arrays by multiplying in the Fourier domain.

        `in1` and `in2` must have the same dtype, and both the forward
        and backward FFTs must be initialised.

        Parameters
        ----------
        in1 : np.ndarray
            First input array
        in2 : np.ndarray
            Second input array to by convolved with `x`. Must have
            the same dtype as `x`.

        Returns
        -------
        out : np.ndarray
            Discrete convolution of `in1` and `in2`
        """
        X1 = self.fft(in1)
        X2 = self.fft(in2)

        X1 *= X2

        return self.ifft(X1)

    def fftwindow(self, x, window):
        """Apply a window function in Fourier space.

        The only difference between this and `fftconvolve` is that
        this assumes that `window` is _already_ in the Fourier domain,
        and `window` can be real or complex when `x` is complex.

        Both the forward and backward FFTs must be initialised.

        Parameters
        ----------
        x : np.ndarray
            Input array
        window : np.ndarray
            Window to be applied in the Fourier domain.

        Returns
        -------
        out : np.ndarray
            Input array `x` with `window` applied in the Fourier domain.
        """
        X = self.fft(x)
        X *= window

        return self.ifft(X)


def fft(x, axes=None):
    """Perform a forward discrete Fourer Transform.

    If the fourier transform is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the `FFT` class directly to avoid creating new `FFT` objects.

    Parameters
    ----------
    x : np.ndarray
        Input array, real or complex
    axes : None | int | tuple
        Axes over which to take the fft. Default is all axes.

    Returns
    -------
    fft : np.ndarray
        DFT of the input array over specified axes
    """
    fftobj = FFT(x.shape, x.dtype, axes, forward=True, backward=False)

    return fftobj.fft(x)


def ifft(x, axes=None):
    """Perform an inverse discrete Fourier Transform.

    If the fourier transform is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the `FFT` class directly to avoid creating new `FFT` objects.

    Parameters
    ----------
    x : np.ndarray
        Input array, real or complex
    axes : None | int | tuple
        Axes over which to take the ifft. Default is all axes.

    Returns
    -------
    fft : np.ndarray
        IDFT of the input array over specified axes
    """
    fftobj = FFT(x.shape, x.dtype, axes, forward=False, backward=True)

    return fftobj.ifft(x)


def fftconvolve(in1, in2, axes=None):
    """Convolve two arrays by multiplying in the Fourier domain.

    `in1` and `in2` must have the same dtype.

    If the convolution is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the `FFT` class directly to avoid creating new `FFT` objects.

    Parameters
    ----------
    in1 : np.ndarray
        First input array
    in2 : np.ndarray
        Second input array to by convolved with `x`. Must have
        the same dtype as `x`.
    axes : None | int | tuple
        Axes over which to do the convolution. Default is all axes.

    Returns
    -------
    out : np.ndarray
        Discrete convolution of `in1` and `in2`
    """
    fftobj = FFT(in1.shape, in1.dtype, axes, forward=True, backward=True)

    return fftobj.fftconvolve(in1, in2)


def fftwindow(x, window, axes):
    """Apply a window function in Fourier space.

    The only difference between this and `fftconvolve` is that
    this assumes that `window` is _already_ in the Fourier domain,
    and `window` can be real or complex when `x` is complex.

    If the window is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the `FFT` class directly to avoid creating new `FFT` objects.

    Parameters
    ----------
    x : np.ndarray
        Input array
    window : np.ndarray
        Window to be applied in the Fourier domain.
    axes : None | int | tuple
        Axes over which to apply the window. Default is all axes.

    Returns
    -------
    out : np.ndarray
        Input array `x` with `window` applied in the Fourier domain.
    """
    fftobj = FFT(x.shape, x.dtype, axes, forward=True, backward=True)

    return fftobj.fftwindow(x, window)
