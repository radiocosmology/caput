r"""Efficient FFT implementations.

Features
~~~~~~~~
- Uses multi-core `scipy.fft`_ by default \- all `scipy.fft`_ functions
  are re-exported.
- Provides helpers to use `pyfftw`_ for faster FFTs when available.

.. _`scipy.fft`: https://docs.scipy.org/doc/scipy/reference/fft.html
.. _`pyfftw`: https://pyfftw.readthedocs.io/en/latest/
"""

import scipy.fft

from ...util import mpitools

from . import fftw as fftw

# Overwrite any existing backends on import
scipy.fft.set_global_backend("scipy", only=True)

# Re-export only from `scipy.fft`
# __all__ = scipy.fft.__all__

# Only use this lookup once
_nworkers = mpitools.cpu_count()


def _set_workers(func):
    def _inner(*args, **kwargs):
        with scipy.fft.set_workers(kwargs.pop("workers", _nworkers)):
            return func(*args, **kwargs)

    return _inner


_non_scipy_symboles = {"fftw"}


# Re-export all scipy symbols with multiple workers set
def __getattr__(name):
    if name in _non_scipy_symboles:
        return globals()[name]

    return _set_workers(getattr(scipy.fft, name))
