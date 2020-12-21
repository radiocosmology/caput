"""Tools for calculating the effects of the CASPER tools PFB.

This module can:
- Evaluate the typical window functions used
- Evaluate a python model of the PFB
- Calculate the decorrelation effect for signals offset by a known time delay.
"""
# === Start Python 2/3 compatibility
from __future__ import absolute_import, division, print_function, unicode_literals
from future.builtins import *  # noqa  pylint: disable=W0401, W0614
from future.builtins.disabled import *  # noqa  pylint: disable=W0401, W0614

# === End Python 2/3 compatibility

import numpy as np
from scipy.interpolate import interp1d


def sinc_window(ntap, lblock):
    """Sinc window function.

    Parameters
    ----------
    ntap : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntap * lblock]
    """
    # Sampling locations of sinc function
    X = np.linspace(-ntap / 2, ntap / 2, ntap * lblock, endpoint=False)

    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so we can just X
    return np.sinc(X)


def sinc_hann(ntap, lblock):
    """Hann-sinc window function.

    Parameters
    ----------
    ntap : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntap * lblock]
    """

    return sinc_window(ntap, lblock) * np.hanning(ntap * lblock)


def sinc_hamming(ntap, lblock):
    """Hamming-sinc window function.

    Parameters
    ----------
    ntap : integer
        Number of taps.
    lblock: integer
        Length of block.

    Returns
    -------
    window : np.ndarray[ntap * lblock]
    """

    return sinc_window(ntap, lblock) * np.hamming(ntap * lblock)


class PFB(object):
    """Model for the CASPER PFB.

    This is the PFB used in CHIME and other experiments.

    Parameters
    ----------
    ntap : int
        Number of taps (i.e. blocks) used in one step of the PFB.
    lblock : int
        The length of a block that gets transformed. This is twice the number
        of output frequencies.
    window : function, optional
        The window function being used. If not set, use a Sinc-Hamming window.
    oversample : int, optional
        The amount to oversample when calculating the decorrelation ratio.
        This will improve accuracy.
    """

    def __init__(self, ntap, lblock, window=None, oversample=4):

        self.ntap = ntap
        self.lblock = lblock

        self.window = sinc_hamming if window is None else window
        self.oversample = oversample

    def apply(self, timestream):
        """Apply the PFB to a timestream.

        Parameters
        ----------
        timestream : np.ndarray
            Timestream to process.

        Returns
        -------
        pfb : np.ndarray[:, lblock // 2]
            Array of PFB frequencies.
        """

        # Number of blocks
        nblock = timestream.size // self.lblock - (self.ntap - 1)

        # Initialise array for spectrum
        spec = np.zeros((self.nblock, self.lblock // 2), dtype=np.complex128)

        # Window function
        w = self.window(self.ntap, self.lblock)

        # Iterate over blocks and perform the PFB
        for bi in range(nblock):
            # Cut out the correct timestream section
            ts_sec = timestream[(bi * self.lblock) : ((bi + self.ntap) * self.lblock)]

            # Perform a real FFT (with applied window function)
            ft = np.fft.rfft(ts_sec * w)

            # Choose every n-th frequency
            spec[bi] = ft[:: self.ntap]

        return spec

    _decorr_interp = None

    def decorrelation_ratio(self, delay):
        """Calculates the decorrelation caused by a relative relay of two timestreams.

        This is caused by the fact that the PFB is generated from a finite
        time window of data.

        Parameters
        ----------
        delay : array_like
            The relative delay between the correlated streams in units of
            samples (not required to be an integer).

        Returns
        -------
        decorrelation : array_like
            The decorrelation ratio.
        """

        if self._decorr_interp is None:

            N = self.ntap * self.lblock

            # Calculate the window and zero pad the array by a factor of oversample
            window_extended = np.zeros(N * self.oversample)
            window_extended[:N] = self.window(self.ntap, self.lblock)

            # Calculate the FFT and copy into an array over padded by another factor of
            # oversample. As we are doing real/inverse-real FFTs the actual length of
            # this array has the usual 1/2 N + 1 sizing.
            wf = np.fft.rfft(window_extended)
            wfpad = np.zeros(N * self.oversample ** 2 // 2 + 1, dtype=np.complex128)
            wfpad[: wf.size] = np.abs(wf) ** 2

            # Calculate the ratio and the effective delays it is available at
            decorrelation_ratio = np.fft.irfft(wfpad)
            tau = np.fft.fftfreq(
                N * self.oversample ** 2, d=(1.0 / (N * self.oversample))
            )

            # Extract only the relevant range of time
            tau_r = tau[np.abs(tau) <= N]
            dc_r = decorrelation_ratio[np.abs(tau) <= N] / decorrelation_ratio[0]

            self._decorr_interp = interp1d(
                tau_r,
                dc_r,
                kind="linear",
                fill_value=0,
                assume_sorted=False,
                bounds_error=False,
            )

        return self._decorr_interp(delay)
