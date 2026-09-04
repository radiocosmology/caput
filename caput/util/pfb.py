"""Tools for calculating the effects of the CASPER tools PFB.

This module can:

- Evaluate the typical window functions used
- Evaluate a python model of the PFB
- Calculate the decorrelation effect for signals offset by a known time delay.
- Deconvolve the effects of the PFB from finely channelised data.

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.linalg as la
import scipy.sparse as ss
from scipy.interpolate import CubicSpline, interp1d

from ..algorithms import invert_no_zero

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt


def sinc_window(ntap: int, lblock: int) -> np.ndarray:
    """Sinc window function.

    Parameters
    ----------
    ntap : int
        Number of taps.
    lblock : int
        Length of block.

    Returns
    -------
    window : ndarray
        Array of length `ntap * lblock` with sinc window values.
    """
    # Sampling locations of sinc function
    X = np.linspace(-ntap / 2, ntap / 2, ntap * lblock, endpoint=False)

    # np.sinc function is sin(pi*x)/pi*x, not sin(x)/x, so we can just X
    return np.sinc(X)


def sinc_hann(ntap: int, lblock: int) -> np.ndarray:
    """Hann-sinc window function.

    Parameters
    ----------
    ntap : int
        Number of taps.
    lblock : int
        Length of block.

    Returns
    -------
    window : ndarray
        Array of length `ntap * lblock` with Hann-sinc window values.
    """
    return sinc_window(ntap, lblock) * np.hanning(ntap * lblock)


def sinc_hamming(ntap: int, lblock: int) -> np.ndarray:
    """Hamming-sinc window function.

    Parameters
    ----------
    ntap : int
        Number of taps.
    lblock : int
        Length of block.

    Returns
    -------
    window : ndarray
        Array of length `ntap * lblock` with Hamming-sinc window values.
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
    window : callable, optional
        The window function being used. If not set, use a Sinc-Hamming window.
    oversample : int, optional
        The amount to oversample when calculating the decorrelation ratio.
        This will improve accuracy. Default is 4.
    """

    def __init__(
        self,
        ntap: int,
        lblock: int,
        window: Callable | None = None,
        oversample: int = 16,
    ):
        """Set PFB parameters."""
        self.ntap = ntap
        self.lblock = lblock

        self.window = sinc_hamming if window is None else window
        self.oversample = oversample

        self._profile_interp = None

    def apply(self, timestream: npt.NDArray) -> np.ndarray[np.complex128]:
        """Apply the PFB to a timestream.

        Parameters
        ----------
        timestream : array_like
            Timestream to process.

        Returns
        -------
        freqs : complex ndarray
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

    def compute_channel_profile(
        self, norm: bool = True
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.complex128]]:
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

    def evaluate_channel_profile(
        self, channel_width_frac: npt.ArrayLike | None = None, norm: bool = True
    ) -> np.ndarray[np.complex128]:
        """Evaluate the profile of a single frequency channel.

        On the first call, this method computes the profile on a dense
        set of frequencies and constructs and interpolating function.
        This interpolating function is evaluated on subsequent calls.

        Note that this is the voltage profile; the absolute value of the
        output should be squared to obtain the profile corresponding to a
        visibility.

        Parameters
        ----------
        channel_width_frac : array_like | None, optional
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

    def decorrelation_ratio(self, delay: npt.ArrayLike) -> np.ndarray[np.float64]:
        """Calculate the decorrelation caused by a relative delay between two timestreams.

        This is caused by the fact that the PFB is generated from a finite time window
        of data.

        Parameters
        ----------
        delay : array_like
            The relative delay between the correlated streams in units of samples
            (not required to be an integer).

        Returns
        -------
        ratio : ndarray
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


class DeconvolvePFB:
    """Deconvolve the effects of the PFB from finely channelised data.

    Default parameters represent what is done within CHIME.

    Parameters
    ----------
    N : int, optional
        Length of a PFB tap.
    M : int, optional
        Number of PFB taps.
    Q : int, optional
        Number of subfrequencies, or equivalently the number of PFB outputs in the
        second FFT.
    window : callable, optional
        The window function applied in the PFB. If not set, use a Sinc-Hamming
        window.
    band : int, optional
        Number of neighbouring frequencies on either side of each output frequency
        to include in the deconvolution.
    nyquist : bool, optional
        Is the nyquist frequency included in the data? Default is False to
        match the CHIME/Caspertools PFB.
    """

    def __init__(
        self,
        N: int = 2048,
        M: int = 4,
        Q: int = 128,
        window: Callable[[int, int], np.ndarray] = sinc_hamming,
        band: int = 1,
        nyquist: bool = False,
    ):
        """Set the PFB parameters and generate the deconvolution matrices."""
        self.N = N
        self.M = M
        self.Q = Q
        self.band = band
        self.nyquist = nyquist

        if M > Q:
            raise ValueError(
                f"Number of subfreq ({Q=}) must be more than the number of PFB taps "
                f"({M=})."
            )

        w_pad = np.zeros(N * Q, dtype=np.float64)
        w_pad[: (M * N)] = window(M, N)

        self.Wt = np.fft.fft(w_pad).conj().reshape(N, Q).transpose(1, 0) / N
        self.Wt2 = np.abs(self.Wt) ** 2

        self._gen_matrices_sparse()

    def _gen_matrices_dense(self):
        """Generate the deconvolution matrices as dense arrays.

        This is slow, but is a good reference implementation that the sparse matrices
        are doing the correct thing.
        """
        Q = self.Q
        N = self.N
        band = self.band

        self.W = np.zeros((Q, N, N), dtype=np.float64)

        for ri in range(N):
            for alpha in range(-band, band):
                self.W[:, ri, (ri + alpha) % N] = self.Wt2[:, alpha]

        # Construct the matrix to project from the positive frequencies to the full
        # space
        self.Hf = np.zeros((N, N // 2 + 1), dtype=np.float64)
        for ri in range(N // 2 + 1):
            self.Hf[ri, ri] = 1.0
        for ri in range(N // 2 - 1):
            hN = N // 2
            self.Hf[hN + ri + 1, hN - ri - 1] = 1.0

        # Construct the matrix to go from the full frequencies to the positive
        # frequencies (excluding Nyquist)
        Nb = N // 2 + (1 if self.nyquist else 0)
        self.Hb = np.zeros((Nb, N), dtype=np.float64)
        for ri in range(Nb):
            self.Hb[ri, ri] = 1.0

        self.Wc = np.zeros((Q, Nb, N // 2 + 1), dtype=np.float64)

        for s in range(Q):
            self.Wc[s] = self.Hb @ self.W[s] @ self.Hf

    def _gen_matrices_sparse(self):
        """Generate the deconvolution matrices as sparse arrays."""
        Q = self.Q
        N = self.N
        band = self.band

        self.W = []

        # Pre-generate arrays for the row and col indices which are the same for each
        # subfreq
        row_ind = np.zeros((2 * band, N), dtype=np.int32)
        row_ind[:] = np.arange(N)[np.newaxis, :]
        band_ind = np.arange(-band, band, dtype=np.int32)
        col_ind = (row_ind + band_ind[:, np.newaxis]) % N

        data = np.zeros((2 * band, N), dtype=np.float64)

        for s in range(Q):
            data[:] = self.Wt2[s, band_ind][:, np.newaxis]
            self.W.append(
                ss.csr_array(
                    (data.ravel(), (row_ind.ravel(), col_ind.ravel())), shape=(N, N)
                )
            )

        # Construct the matrix to project from the positive frequencies to the full
        # space
        self.Hf = ss.lil_array((N, N // 2 + 1), dtype=np.float64)
        for ri in range(N // 2 + 1):
            self.Hf[ri, ri] = 1.0
        for ri in range(N // 2 - 1):
            hN = N // 2
            self.Hf[hN + ri + 1, hN - ri - 1] = 1.0
        self.Hf = self.Hf.tocsr()

        # Construct the matrix to go from the full frequencies to the positive
        # frequencies (excluding Nyquist)
        Nb = N // 2 + (1 if self.nyquist else 0)
        self.Hb = ss.lil_array((Nb, N), dtype=np.float64)
        for ri in range(Nb):
            self.Hb[ri, ri] = 1.0
        self.Hb = self.Hb.tocsr()

        self.Wc = [self.Hb @ self.W[s] @ self.Hf for s in range(Q)]

    def flatten(
        self, x: npt.NDArray, Ni: npt.NDArray, centered: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the data to remove the quantisation noise bias and the PFB shape.

        Parameters
        ----------
        x : np.ndarray
            Data array packed as [freq, subfreq, time].
        Ni : np.ndarray
            Weights (i.e. inverse noise variance) packed the same way as x.
        centered : bool, optional
            Is the data centered (i.e. as it is in the raw data) or shifted such that
            subfreq=0 is the DC bin.

        Returns
        -------
        fx : np.ndarray
            Flattened data with the bias subtracted out and divided by the bandpass and
            the average flux.
        fNi : np.ndarray
            Weights with the equivalent flattening correction applied.
        """
        # TODO: expose these parameters
        sig_a = 1e4
        sig_b = 1e4
        sig_d = 1e4

        # This is the subfreq template to fit
        w = self.Wt2.sum(axis=1)

        if centered:
            w = np.roll(w, self.Q // 2)

        # We need to construct all the products efficiently. This section tries to build
        # up all the needed combinations.
        wN = w[np.newaxis, :, np.newaxis] * Ni
        zN = Ni
        # Quadratic products
        wNw = (wN * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        zNw = (zN * w[np.newaxis, :, np.newaxis]).sum(axis=1)
        zNz = zN.sum(axis=1)
        # Products against the data
        wNd = (wN * x).sum(axis=1)
        zNd = (zN * x).sum(axis=1)

        # Construct the inverse covariance term via Sherman-Morrison using the products
        # above
        Ci = np.empty((x.shape[0], 2, 2), dtype=np.float64)
        Ci[:, 0, 0] = np.sum(wNw - sig_d**2 / (1 + sig_d**2 * wNw) * wNw**2, axis=-1)
        Ci[:, 0, 1] = np.sum(zNw - sig_d**2 / (1 + sig_d**2 * wNw) * zNw * wNw, axis=-1)
        Ci[:, 1, 0] = Ci[:, 0, 1]
        Ci[:, 1, 1] = np.sum(zNz - sig_d**2 / (1 + sig_d**2 * wNw) * zNw**2, axis=-1)
        # Add in the signal term
        Ci[:, 0, 0] += sig_a**-2
        Ci[:, 1, 1] += sig_b**-2

        # Construct the "dirty" estimator
        dirty = np.empty((x.shape[0], 2), dtype=np.float64)
        dirty[:, 0] = np.sum(wNd - sig_d**2 / (1 + sig_d**2 * wNw) * wNw * wNd, axis=-1)
        dirty[:, 1] = np.sum(zNd - sig_d**2 / (1 + sig_d**2 * wNw) * zNw * wNd, axis=-1)

        # Solve for a and b, and then apply to the data
        fx = np.empty_like(x)
        fNi = np.empty_like(Ni)
        for ii in range(x.shape[0]):
            a, b = la.solve(Ci[ii], dirty[ii], assume_a="pos")
            fx[ii] = (x[ii] - b) * invert_no_zero(a * w[:, np.newaxis])
            fNi[ii] = (a * w[:, np.newaxis]) ** 2 * Ni[ii]

        return fx, fNi
