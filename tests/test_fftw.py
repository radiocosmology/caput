"""Unit tests for the fftw module."""

import numpy as np
import pytest

from caput import fftw
from scipy import fft as sfft


ARRAY_SIZE = (100, 111)
SEED = 12345
ATOL = 1e-10
rng = np.random.Generator(np.random.SFC64(SEED))

# NOTE: only complex->complex transforms are currently supported,
# but we still want to test that a proper error is raised
random_double_array = rng.standard_normal(size=ARRAY_SIZE, dtype=np.float64)
random_complex_array = rng.standard_normal(
    size=ARRAY_SIZE
) + 1.0j * rng.standard_normal(size=ARRAY_SIZE)


@pytest.mark.parametrize("x", [random_double_array])
def test_invalid_type(x):
    """Test that an error is raised with a non-complex type."""
    with pytest.raises(TypeError):
        fftw.FFT(x.shape, x.dtype)


@pytest.mark.parametrize("x", [random_complex_array])
@pytest.mark.parametrize("ax", [(0,), (1,), None])
def test_forward_backward(x, ax):
    """Test that ifft(fft(x)) returns the original array."""
    # Test the direct class implementation
    fftobj = fftw.FFT(x.shape, x.dtype, ax)

    if np.isrealobj(x):
        # pyfftw will destroy the input array for
        # real->real inverse transform, but we want
        # to test that it _won't_ destroy the array in
        # the complex case
        xi = fftobj.ifft(fftobj.fft(x.copy()))
    else:
        xi = fftobj.ifft(fftobj.fft(x))

    assert np.allclose(x, xi, atol=ATOL)

    # Test the api
    if np.isrealobj(x):
        xi = fftw.ifft(fftw.fft(x.copy()))
    else:
        xi = fftw.ifft(fftw.fft(x))

    assert np.allclose(x, xi, atol=ATOL)


@pytest.mark.parametrize("x", [random_complex_array])
@pytest.mark.parametrize("ax", [(0,), (1,), None])
def test_scipy(x, ax):
    """Test that this produces the same results as `scipy.fft`."""
    Xc = fftw.fft(x, ax)
    ixc = fftw.ifft(Xc, ax)

    # Scipy requires different calls for 1D, 2D, real, and complex cases
    if ax is not None and len(ax) == 1:
        Xs = sfft.fft(x, axis=ax[0])
        ixs = sfft.ifft(Xs, axis=ax[0])
    else:
        Xs = sfft.fft2(x)
        ixs = sfft.ifft2(Xs)

    assert np.allclose(Xc, Xs, atol=ATOL)
    assert np.allclose(ixc, ixs, atol=ATOL)
