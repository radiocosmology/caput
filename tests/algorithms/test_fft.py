"""Unit tests for the fftw module."""

import numpy as np
import pytest

from caput.algorithms import fft


ARRAY_SIZE = (100, 111)
SEED = 12345
ATOL = 1e-12
rng = np.random.Generator(np.random.SFC64(SEED))

# NOTE: only complex->complex transforms are currently supported,
# but we still want to test that a proper error is raised
random_double_array = rng.standard_normal(size=ARRAY_SIZE, dtype=np.float64)
random_complex_array = rng.standard_normal(
    size=ARRAY_SIZE
) + 1.0j * rng.standard_normal(size=ARRAY_SIZE)

# -----------
# FFTW tests
# -----------


@pytest.mark.parametrize("x", [random_double_array])
def test_invalid_type_fftw(x):
    """Test that an error is raised with a non-complex type."""
    with pytest.raises(TypeError):
        fft.fftw.FFTW(x.shape, x.dtype)


@pytest.mark.parametrize("x", [random_complex_array])
@pytest.mark.parametrize("ax", [(0,), (1,), None])
def test_forward_backward_fftw(x, ax):
    """Test that ifft(fft(x)) returns the original array."""
    # Test the direct class implementation
    fftobj = fft.fftw.FFTW(x.shape, x.dtype, ax)

    if np.isrealobj(x):
        # pyfftw will destroy the input array for
        # real->real inverse transform, but we want
        # to test that it _won't_ destroy the array in
        # the complex case
        xi = fftobj.ifft(fftobj.fft(x.copy()))
    else:
        xi = fftobj.ifft(fftobj.fft(x))

    assert np.allclose(x, xi, atol=ATOL)

    if np.isrealobj(x):
        xi = fft.fftw.ifft(fft.fftw.fft(x.copy()))
    else:
        xi = fft.fftw.ifft(fft.fftw.fft(x))

    assert np.allclose(x, xi, atol=ATOL)


# --------------
# `scipy` tests
# --------------


@pytest.mark.parametrize("x", [random_complex_array, random_double_array])
@pytest.mark.parametrize("ax", [(0,), (1,), (-2, -1), None])
def test_forward_backward_scipy(x, ax):
    """Test that ifft(fft(x)) returns the original array."""
    xi = fft.ifftn(fft.fftn(x, axes=ax), axes=ax)

    assert np.allclose(x, xi, atol=ATOL)


# --------
# Compare
# --------


@pytest.mark.parametrize("x", [random_complex_array])
@pytest.mark.parametrize("ax", [(0,), (1,), None])
def test_equal_scipy_fftw(x, ax):
    """Test that `fftw` and `scipy_fft` give the same results.'"""
    Xc = fft.fftw.fft(x, ax)
    ixc = fft.fftw.ifft(Xc, ax)

    Xs = fft.fftn(x, axes=ax)
    ixs = fft.ifftn(Xs, axes=ax)

    assert np.allclose(Xc, Xs, atol=ATOL)
    assert np.allclose(ixc, ixs, atol=ATOL)
