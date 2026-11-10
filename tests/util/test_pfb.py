"""Tests for polyphase filter bank windowing and transforms."""

import numpy as np

from caput.util import pfb


def test_sinc_windows_have_expected_length_and_windowing():
    """Build sinc, Hann-sinc, and Hamming-sinc windows of the requested length."""
    ntap, lblock = 3, 8
    expected_sinc = np.sinc(
        np.linspace(-ntap / 2, ntap / 2, ntap * lblock, endpoint=False)
    )

    np.testing.assert_array_equal(pfb.sinc_window(ntap, lblock), expected_sinc)
    np.testing.assert_allclose(
        pfb.sinc_hann(ntap, lblock), expected_sinc * np.hanning(ntap * lblock)
    )
    np.testing.assert_allclose(
        pfb.sinc_hamming(ntap, lblock), expected_sinc * np.hamming(ntap * lblock)
    )


def test_pfb_apply_returns_expected_number_of_spectra_and_channels():
    """Transform a timestream and check the shape of the resulting spectra."""
    ntap, lblock = 2, 8
    timestream = np.arange(4 * lblock, dtype=np.float64)
    transform = pfb.PFB(ntap, lblock)

    result = transform.apply(timestream)

    assert result.shape == (len(timestream) // lblock - (ntap - 1), lblock // 2)
    assert result.dtype == np.dtype(np.complex128)
