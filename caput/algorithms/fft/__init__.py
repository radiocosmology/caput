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

try:
    from . import fftw as fftw
except ImportError:
    pass

# Overwrite any existing backends on import
scipy.fft.set_global_backend("scipy", only=True)

# Only use this lookup once
_nworkers = mpitools.cpu_count()


def _set_workers(func):
    def _inner(*args, **kwargs):
        with scipy.fft.set_workers(kwargs.pop("workers", _nworkers)):
            return func(*args, **kwargs)

    return _inner


# Importable symbols which are independent of `scipy.fft`
_non_scipy_symbols = {"fftw"}


# Re-export all scipy symbols with multiple workers set
def __getattr__(name):
    if name in _non_scipy_symbols:
        try:
            return globals()[name]
        except (KeyError, ValueError):
            raise AttributeError(
                f"Unable to find symbol {name}. "
                "This is likely due to a missing optional dependency."
            )

    return _set_workers(getattr(scipy.fft, name))
