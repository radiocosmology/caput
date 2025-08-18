"""caput.containers.

A high-level in-memory data container format for `caput.pipeline`.
"""

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
from . import tod as tod

# Try to import bitshuffle to set the default compression options
try:
    import bitshuffle.h5

    COMPRESSION = bitshuffle.h5.H5FILTER
    COMPRESSION_OPTS = (0, bitshuffle.h5.H5_COMPRESS_LZ4)
except ImportError:
    COMPRESSION = None
    COMPRESSION_OPTS = None

__all__ = [
    "ContainerBase",
    "DataWeightContainer",
    "FreqContainer",
    "TableBase",
    "copy_datasets_filter",
    "empty_like",
]
