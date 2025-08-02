"""caput.fft.

Utilities for FFTs.

Features
========
- Multi-threaded `scipy.fft` by default
- Helpers to use `pyFFTW`
"""

from ._scipy_fft import *
from . import fftw as fftw
