from typing import Literal

from ...config import Property
from ...memdata.fileformats import FileFormat

__all__ = ["file_format"]

def file_format(
    default: Literal["hdf5", "zarr"] | FileFormat | None = None,
) -> Property: ...
