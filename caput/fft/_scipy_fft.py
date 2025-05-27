"""Scipy FFT routines.

This module just re-exports `scipy.fft`, but it uses
all cores by default (instead of one). Users can still
set the number of cores on a case-by-case basis using
the standard `scipy` syntax.

For small arrays, it will generally be faster to use a
single threaded version from `scipy.fft` directly.
"""

import scipy.fft
import scipy.fft._pocketfft
from scipy.fft import *  # noqa: F403

from .. import mpiutil

# Re-export only from `scipy.fft`
__all__ = scipy.fft.__all__

# Overwrite any existing backends on import
scipy.fft.set_global_backend("scipy", only=True)
# Set the default number of workers
# NOTE: I'm not sure if this could break anything. Be careful
scipy.fft._pocketfft.helper._config.default_workers = mpiutil.cpu_count()
