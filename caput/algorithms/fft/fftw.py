"""Fast FFT implementation using FFTW.

This module adds some minor abstraction to use `pyfftw`_ in a way
which seems to be faster than using the `pyfftw.builders`_ interface,
and uses the same api as `scipy.fft`_ and `numpy.fft`_.

Only forward and reverse complex->complex transforms
are currently supported.

Examples
--------
The core of this module is the :py:class:`.FFTW`, which essentially just
abstracts the :py:class:`pyfftw.FFTW` class in a simple way.

>>> import numpy as np
>>> from caput.algorithms.fft import fftw
>>>
>>> shape = (24, 50)
>>> x = np.random.rand(*shape) + 1j * np.random.rand(*shape)
>>>
>>> fftobj = fftw.FFTW(x.shape, x.dtype, axes=-1)
>>>
>>> X = fftobj.fft(x)
>>> xi = fftobj.ifft(X)
>>>
>>> np.allclose(x, xi)
True

Alternatively, for one-off transforms, the module-level functions
can be used, which create :py:class:`.FFTW` objects internally.

.. _`pyfftw`: https://pyfftw.readthedocs.io/en/latest/
.. _`pyfftw.builders`: https://pyfftw.readthedocs.io/en/latest/source/pyfftw/builders/builders.html
.. _`scipy.fft`: https://docs.scipy.org/doc/scipy/reference/fft.html
.. _`numpy.fft`: https://numpy.org/doc/stable/reference/routines.fft.html
"""

from __future__ import annotations

import numpy as np
import pyfftw

from ...util import mpitools

__all__ = ["FFTW", "fft", "fftconvolve", "fftwindow", "ifft"]


class FFTW:
    r"""Create FFTW objects for repeat use.

    This implementation is most efficient when used to repeatedly
    apply FFTs to arrays with the same shape and dtype, because a
    single, highly optimised pathway can be used with a single
    initialisation.

    Even for a single use, this will typically
    be faster than the :py:meth:`scipy.fft` or :py:meth:`numpy.fft` implementations,
    especially when multiple cores are available.

    Parameters
    ----------
    shape : tuple[int, ...]
        The shape of the arrays to initialise.
    dtype : dtype
        Datatype to create a pathway for. At the moment, only
        ``complex -> complex`` or ``real -> real`` are supported. The
        :py:mod:`pyfftw` implementation of the ``real -> real`` backward
        transform will destroy the input array.
    axes : int | iterable[int], optional
        Axes over which to apply the FFT. Default is all axes.
    forward : bool, optional
        If true, initialise the forward FFT. Default is True.
    backward : bool, optional
        If true, initialise the backward FFT. Default is True.
    """

    def __init__(self, shape, dtype, axes=None, forward=True, backward=True):
        """Create FFTW objects for repeat use."""
        # Ensure dtype is a numpy dtype
        dtype = np.dtype(dtype)

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
        parameters : dict
            ncpu, simd alignment, shape, dtype, axes, and flags
            used by this FFTW object.
        """
        return self._params

    def fft(self, x):
        """Perform a forward FFT.

        Parameters
        ----------
        x : complex array_like
            Input array, must match the dtype specified at creation.

        Returns
        -------
        fft : complex ndarray
            DFT of the input array over specified axes.

        Raises
        ------
        AttributeError
            If the forward FFT is not initialized.
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

        When performing the backward ``real -> real`` IFFT,
        the input array is destroyed.

        Parameters
        ----------
        x : complex array_like
            Input array, must match the dtype specified at creation.

        Returns
        -------
        ifft : complex ndarray
            IDFT of the input array over specified axes.

        Raises
        ------
        AttributeError
            If the backward FFT is not initialized.
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
        in1 : complex array_like
            First input array
        in2 : complex_array_like
            Second input array to by convolved with `x`. Must have
            the same dtype as `x`.

        Returns
        -------
        convolution : complex ndarray
            Discrete convolution of `in1` and `in2`

        Raises
        ------
        AttributeError
            If either the forward or the backward FFT is not initialized.
        """
        X1 = self.fft(in1)
        X2 = self.fft(in2)

        X1 *= X2

        return self.ifft(X1)

    def fftwindow(self, x, window):
        """Apply a window function in Fourier space.

        The only difference between this and `fftconvolve` is that
        this assumes that `window` is *already* in the Fourier domain,
        and `window` can be real or complex when `x` is complex.

        Both the forward and backward FFTs must be initialised.

        Parameters
        ----------
        x : (..., N, ...) complex array_like
            Input array
        window : (..., N, ...) array_like
            Window to be applied in the Fourier domain.

        Returns
        -------
        windowed : (..., N, ...) complex ndarray
            Input array `x` with `window` applied in the Fourier domain.

        Raises
        ------
        AttributeError
            If either the forward or the backward FFT is not initialized.
        """
        X = self.fft(x)
        X *= window

        return self.ifft(X)


def fft(x, axes=None):
    """Perform a forward discrete Fourier Transform.

    If the fourier transform is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the :py:class:`.FFTW` class directly to avoid creating new `FFTW` objects.

    Parameters
    ----------
    x : complex array_like
        Complex input array
    axes : int | iterable[int], optional
        Axes over which to take the fft. Default is all axes.

    Returns
    -------
    fft : complex ndarray
        DFT of the input array over specified axes
    """
    fftobj = FFTW(x.shape, x.dtype, axes, forward=True, backward=False)

    return fftobj.fft(x)


def ifft(x, axes=None):
    """Perform an inverse discrete Fourier Transform.

    If the fourier transform is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the :py:class:`.FFTW` class directly to avoid creating new `FFTW` objects.

    Parameters
    ----------
    x : complex array_like
        Complex input array
    axes : int | iterable[int], optional
        Axes over which to take the ifft. Default is all axes.

    Returns
    -------
    ifft : complex ndarray
        IDFT of the input array over specified axes
    """
    fftobj = FFTW(x.shape, x.dtype, axes, forward=False, backward=True)

    return fftobj.ifft(x)


def fftconvolve(in1, in2, axes=None):
    """Convolve two arrays by multiplying in the Fourier domain.

    `in1` and `in2` must have the same dtype.

    If the convolution is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the :py:class:`.FFTW` class directly to avoid creating new `FFTW` objects.

    Parameters
    ----------
    in1 : complex array_like
        First input array
    in2 : complex array_like
        Second input array to by convolved with `x`. Must have
        the same dtype as `x`.
    axes : int | iterable[int], optional
        Axes over which to do the convolution. Default is all axes.

    Returns
    -------
    convolution : complex ndarray
        Discrete convolution of `in1` and `in2`
    """
    fftobj = FFTW(in1.shape, in1.dtype, axes, forward=True, backward=True)

    return fftobj.fftconvolve(in1, in2)


def fftwindow(x, window, axes=None):
    """Apply a window function in Fourier space.

    The only difference between this and :py:func:`.fftconvolve` is that
    this assumes that ``window`` is *already* in the Fourier domain,
    and ``window`` can be real or complex when ``x`` is complex.

    If the window is to be applied repeatedly to
    arrays with the same size and dtype, it is faster to use
    the :py:class:`.FFTW` class directly to avoid creating new :py:class`.FFTW`
    objects.

    Parameters
    ----------
    x : (..., N, ...) complex array_like
        Input array
    window : (..., N, ...) array_like
        Window to be applied in the Fourier domain.
    axes : int | iterable[int], optional
        Axes over which to apply the window. Default is all axes.

    Returns
    -------
    out : (..., N, ...) complex ndarray
        Input array `x` with `window` applied in the Fourier domain.
    """
    fftobj = FFTW(x.shape, x.dtype, axes, forward=True, backward=True)

    return fftobj.fftwindow(x, window)
