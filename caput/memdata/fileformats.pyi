import abc
import types
from os import PathLike
from types import ModuleType
from typing import Any

import h5py
import zarr
from mpi4py.MPI import Comm
from numcodecs.abc import Codec

BACKENDS: dict[str, ModuleType]
_compression_enabled: bool

#
class FileFormat(metaclass=abc.ABCMeta):
    module: ModuleType | None
    compression: bool

    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> Any: ...
    @classmethod
    def compression_kwargs(
        cls,
        compression: str | int | None = None,
        compression_opts: tuple[int, int] | None = None,
        compressor: Codec | None = None,
    ) -> dict: ...

#
class HDF5(FileFormat):
    module: ModuleType | None
    compression: bool

    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> h5py.File: ...
    @classmethod
    def compression_kwargs(
        cls,
        compression: str | int | None = None,
        compression_opts: tuple[int, int] | None = None,
        compressor: Codec | None = None,
    ) -> dict: ...

#
class Zarr(FileFormat):
    module: ModuleType | None
    compression: bool

    @classmethod
    def open(cls, *args: Any, **kwargs: Any) -> zarr.Group: ...
    @classmethod
    def compression_kwargs(
        cls,
        compression: str | int | None = None,
        compression_opts: tuple[int, int] | None = None,
        compressor: Codec | None = None,
    ) -> dict: ...

#
class ZarrProcessSynchronizer:
    name: str
    _comm: Comm | None
    def __init__(self, name: str, comm: Comm | None = None) -> None: ...
    def __enter__(self) -> zarr.ProcessSynchronizer: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None: ...

#
def remove_file_or_dir(name: str) -> None: ...
def guess_file_format(
    name: PathLike,
    default: type[FileFormat] = HDF5,
) -> FileFormat: ...
def check_file_format(
    filename: str, file_format: FileFormat | None, data: Any
) -> HDF5 | Zarr: ...
