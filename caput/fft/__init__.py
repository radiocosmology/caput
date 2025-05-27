"""caput.fft.

Utilities for FFTs.

Features
========
- Multi-threaded `scipy.fft` by default
- Helpers to use `pyFFTW`
"""

from ._scipy_fft import *  # noqa: F403
from . import _fftw as fftw
