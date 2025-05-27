"""Scipy FFT routines.

This module just re-exports `scipy.fft`, but it uses
all cores by default (instead of one). Users can still
set the number of cores on a case-by-case basis using
the standard `scipy` syntax.

For small arrays, it will generally be faster to use a
single threaded version from `scipy.fft` directly.
"""

import scipy.fft

from .. import mpiutil

# Overwrite any existing backends on import
scipy.fft.set_global_backend("scipy", only=True)

# Re-export only from `scipy.fft`
__all__ = scipy.fft.__all__

# Only use this lookup once
_nworkers = mpiutil.cpu_count()


def _set_workers(func):
    def _inner(*args, **kwargs):
        with scipy.fft.set_workers(kwargs.pop("workers", _nworkers)):
            return func(*args, **kwargs)

    return _inner


# Re-export all scipy symbols with multiple workers set
def __getattr__(name):
    return _set_workers(getattr(scipy.fft, name))
