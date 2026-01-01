"""Tools for calculating the effects of the CASPER tools PFB.

This module can:
- Evaluate the typical window functions used
- Evaluate a python model of the PFB
- Calculate the decorrelation effect for signals offset by a known time delay.

Window functions
================
- :py:meth:`sinc_window`
- :py:meth:`sinc_hanning`
- :py:meth:`sinc_hamming`

PFB
===
- :py:meth:`pfb`
- :py:meth:`decorrelation_ratio`
"""

import numpy as np
from scipy.interpolate import CubicSpline, interp1d


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


class PFB:
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

        self._profile_interp = None

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
        spec = np.zeros((nblock, self.lblock // 2), dtype=np.complex128)

        # Window function
        w = self.window(self.ntap, self.lblock)

        # Iterate over blocks and perform the PFB
        for bi in range(nblock):
            # Cut out the correct timestream section
            ts_sec = timestream[(bi * self.lblock) : ((bi + self.ntap) * self.lblock)]

            # Perform a real FFT (with applied window function)
            ft = np.fft.rfft(ts_sec * w)

            # Choose every n-th frequency
            spec[bi] = ft[: ((self.lblock // 2) * self.ntap) : self.ntap]

        return spec

    def compute_channel_profile(self, norm=True):
        """Compute the profile of a single frequency channel.

        This method computes the profile at a natural set of frequencies
        relative to the channel center. The output is suitable for
        input into separate code that constructs an interpolating function.
        If you plan to evaluate the same profile many times, use
        `evaluate_channel_profile` instead, since it will automatically
        construct an interpolating function and then evaluate it for
        subsequent calls.

        Note that this is the voltage profile; the absolute value of the
        output should be squared to obtain the profile corresponding to a
        visibility.

        Parameters
        ----------
        norm : bool, optional
            Normalize the profile to its peak (real-part) value.
            Default: True.

        Returns
        -------
        rel_freq : np.ndarray
            Array of frequencies at which the profile was computed, as
            fractions of the channel width and relative to the center of
            the channel. (For example, 0 is the center of the channel
            and [-0.5, 0.5] are the channel edges.)
        w : np.ndarray
            Channel profile evaluated at `rel_freq`.
        """
        N = self.ntap * self.lblock
        Nfft = N * self.oversample

        window = self.window(self.ntap, self.lblock).astype(np.complex128)
        w = np.fft.fftshift(np.fft.fft(window, n=Nfft))
        rel_freq = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1.0 / self.lblock))

        if norm:
            w /= w.real.max()

        return rel_freq, w

    def evaluate_channel_profile(self, channel_width_frac=None, norm=True):
        """Evaluate the profile of a single frequency channel.

        On the first call, this method computes the profile on a dense
        set of frequencies and constructs and interpolating function.
        This interpolating function is evaluated on subsequent calls.

        Note that this is the voltage profile; the absolute value of the
        output should be squared to obtain the profile corresponding to a
        visibility.

        Parameters
        ----------
        channel_width_frac : array_like
            Array of frequencies at which to evaluate channel profile, as
            a fraction of the channel width and centered at the center
            of the channel. (For example, 0 is the center of the channel
            and [-0.5, 0.5] are the channel edges.)
        norm : bool, optional
            Normalize the profile to its peak (real-part) value.
            Default: True.

        Returns
        -------
        profile : array_like
            The channel profile.
        """
        if self._profile_interp is None:
            rel_freq, w = self.compute_channel_profile(norm=norm)
            self._profile_interp = CubicSpline(rel_freq, w, extrapolate=False)

        return self._profile_interp(channel_width_frac)

    _decorr_interp = None

    def decorrelation_ratio(self, delay):
        """Calculate the decorrelation caused by a relative relay of two timestreams.

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
            wfpad = np.zeros(N * self.oversample**2 // 2 + 1, dtype=np.complex128)
            wfpad[: wf.size] = np.abs(wf) ** 2

            # Calculate the ratio and the effective delays it is available at
            decorrelation_ratio = np.fft.irfft(wfpad)
            tau = np.fft.fftfreq(
                N * self.oversample**2, d=(1.0 / (N * self.oversample))
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
