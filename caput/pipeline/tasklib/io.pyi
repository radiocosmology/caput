import subprocess
from collections.abc import Sequence
from os import PathLike
from typing import Any, ClassVar

import numpy.typing as npt

from ...containers import Container, ContainerPrototype
from ...memdata import MemDiskGroup, MemGroup
from ...memdata._memh5 import FileLike
from ...memdata.fileformats import FileFormat
from ...mpiarray import SelectionLike
from .base import ContainerTask, MPILoggedTask

def list_of_filelists(
    files: Sequence[PathLike] | Sequence[list[PathLike]],
) -> list[PathLike]: ...
def list_or_glob(files: PathLike | list[PathLike]) -> list[PathLike]: ...
def list_of_filegroups(
    groups: dict[str, PathLike] | list[dict[str, PathLike]],
) -> list[dict[str, PathLike]]: ...

class FindFiles(MPILoggedTask):
    files: PathLike | list[PathLike]
    def setup(self) -> list[PathLike]: ...

class SelectionsMixin:
    selections: dict[str, SelectionLike]
    allow_index_map: bool
    _sel: dict[str, SelectionLike]
    def setup(self) -> None: ...
    def _resolve_sel(self): ...
    def _parse_range(self, x: list | tuple) -> slice: ...
    def _parse_index(self, x: list | tuple, type_=...) -> list: ...

class BaseLoadFiles(SelectionsMixin, ContainerTask):
    distributed: bool
    convert_strings: bool
    redistribute: str | None
    def _load_file(
        self, filename: str, extra_message: str = ""
    ) -> Container | MemDiskGroup: ...

class LoadFilesFromParams(BaseLoadFiles):
    files: Sequence[PathLike]
    _file_ind: int
    def process(self) -> Container: ...

class LoadFilesFromAttrs(BaseLoadFiles):
    filename: PathLike
    def process(self, incont: Container) -> Container: ...

class LoadFilesAndSelect(BaseLoadFiles):
    files: Sequence[PathLike]
    key_format: str
    collection: dict[int | str, FileLike]
    def setup(self) -> None: ...
    def process(self, incont: Container) -> Container | None: ...

class LoadFilesFromPathAndTag(LoadFilesFromParams):
    paths: list[PathLike]
    tags: list[str]
    files: ClassVar[list[str] | None]
    def setup(self) -> None: ...

class LoadFiles(LoadFilesFromParams):
    files: ClassVar[list[PathLike] | None]
    def setup(self, files: Sequence[PathLike]): ...

class Save(ContainerTask):
    root: PathLike
    count: int
    def next(self, data: MemGroup) -> MemGroup: ...

class Truncate(ContainerTask):
    dataset: dict[str, bool | float | dict[str, float]] | None
    ensure_chunked: bool
    default_params: ClassVar[dict]
    def _get_params(self, container: ContainerPrototype, dset: str) -> dict | None: ...
    def _get_weights(
        self, container: ContainerPrototype, dset: str, wdset: str
    ) -> npt.ArrayLike: ...
    def process(self, data: ContainerPrototype) -> ContainerPrototype: ...

class ZipZarrContainers(ContainerTask):
    containers: list[str]
    remove: bool
    _host_rank: int | None
    _path_7z: str
    _num_hosts: int
    def setup(self, _: Any | None = None) -> None: ...
    def process(self) -> None: ...

class ZarrZipHandle:
    filename: str
    handle: subprocess.Popen | None
    def __init__(self, filename: str, handle: subprocess.Popen | None) -> None: ...

class SaveZarrZip(ZipZarrContainers):
    _operation_counter: int
    output_name: str
    output_format: FileFormat
    save: bool
    def setup(self) -> None: ...
    def next(self, container: Container) -> ZarrZipHandle: ...

class WaitZarrZip(MPILoggedTask):
    _handles: list[ZarrZipHandle] | None
    def next(self, handle: ZarrZipHandle) -> None: ...
    def finish(self) -> None: ...
