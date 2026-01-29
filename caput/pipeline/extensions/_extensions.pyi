from os import PathLike
from typing import Any

from _typeshed import Incomplete
from mpi4py import MPI

from ...containers import Container
from ...memdata import MemGroup, fileformats
from ...memdata._memh5 import GroupLike
from .._pipeline import Task

__all__ = ["ContainerIOMixin", "GroupIOMixin"]

class _OneAndOne(Task):
    input_root: str
    output_root: str
    output_format: fileformats.FileFormat
    _no_input: bool
    def __init__(self) -> None: ...
    def process(self, input: Any) -> Any: ...
    def validate(self) -> None: ...
    def read_process_write(self, input, input_filename, output_filename): ...
    def read_input(self, filename) -> None: ...
    def cast_input(self, input): ...
    def read_output(self, filename) -> None: ...
    @staticmethod
    def write_output(filename, output, file_format=None, **kwargs) -> None: ...

class SingleBase(_OneAndOne):
    input_filename: str
    output_filename: str
    output_format: Incomplete
    output_compression: str
    output_compression_opts: dict | str | None
    done: bool
    def next(self, input=None): ...

class IterBase(_OneAndOne):
    file_middles: Incomplete
    input_ext: Incomplete
    output_ext: Incomplete
    iteration: int
    def __init__(self) -> None: ...
    def next(self, input=None): ...

class GroupIOMixin:
    @staticmethod
    def read_input(filename: str) -> MemGroup: ...
    @staticmethod
    def read_output(filename: str) -> MemGroup: ...
    @staticmethod
    def write_output(
        filename: str,
        output: GroupLike,
        file_format: fileformats.FileFormat | None = None,
        **kwargs: dict,
    ) -> None: ...

class ContainerIOMixin:
    _distributed: bool
    _comm: MPI.Comm | None
    #
    def read_input(self, filename: PathLike) -> Container: ...
    def read_output(self, filename: PathLike) -> Container: ...
    #
    @staticmethod
    def write_output(
        filename: PathLike,
        output: Container,
        file_format: fileformats.FileFormat | None = None,
        **kwargs: Any,
    ) -> None: ...

class SingleH5Base(GroupIOMixin, SingleBase): ...
class IterH5Base(GroupIOMixin, IterBase): ...
