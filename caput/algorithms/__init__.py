"""Fast implementations of common algorithms."""

# NOTE: this must be imported before the subpackages below. `caput.algorithms.fft`
# pulls in `caput.util`, which imports `caput.util.pfb`, which needs this name to
# already be bound on this package.
from ._invert_no_zero import invert_no_zero as invert_no_zero

from . import (
    fft as fft,
    median as median,
    random as random,
)
