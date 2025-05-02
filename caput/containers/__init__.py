"""caput.containers.

A high-level in-memory data container format for `caput.pipeline`.
"""

from . import _basic, _core, _util
from ._basic import (
    DataWeightContainer as DataWeightContainer,
    FreqContainer as FreqContainer,
)
from ._core import (
    ContainerBase as ContainerBase,
    TableBase as TableBase,
)
from ._util import (
    empty_like as empty_like,
    copy_datasets_filter as copy_datasets_filter,
)

# Try to import bitshuffle to set the default compression options
try:
    import bitshuffle.h5

    COMPRESSION = bitshuffle.h5.H5FILTER
    COMPRESSION_OPTS = (0, bitshuffle.h5.H5_COMPRESS_LZ4)
except ImportError:
    COMPRESSION = None
    COMPRESSION_OPTS = None
