import logging
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from mpi4py import MPI

from ...containers import Container
from ...memdata.fileformats import FileFormat
from .. import extensions
from .._pipeline import Task

__all__ = [
    "ContainerTask",
    "LoggedTask",
    "MPILogFilter",
    "MPILoggedTask",
    "MPITask",
    "SetMPILogging",
    "group_tasks",
]

class MPILogFilter(logging.Filter):
    add_mpi_info: bool
    level_rank0: int
    level_all: int
    comm: MPI.Comm
    def __init__(
        self,
        add_mpi_info: bool = True,
        level_rank0: int = logging.INFO,
        level_all: int = logging.WARN,
    ) -> None: ...
    def filter(self, record: logging.LogRecord) -> bool: ...

def _log_level(
    x: int | Literal["DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"],
) -> int: ...

class SetMPILogging(Task):
    level_rank0: int
    level_all: int
    def __init__(self) -> None: ...

class LoggedTask(Task):
    log_level: int
    _log: logging.Logger
    def __init__(self) -> None: ...
    @property
    def log(self) -> logging.Logger: ...

class MPITask(Task):
    comm: MPI.Comm | None
    def __init__(self) -> None: ...

class _AddRankLogAdapter(logging.LoggerAdapter):
    calling_obj: Any
    def process(
        self, msg: Any, kwargs: dict[str, Any]
    ) -> tuple[Any, dict[str, Any]]: ...

class MPILoggedTask(MPITask, LoggedTask):
    _log: _AddRankLogAdapter
    def __init__(self) -> None: ...

class ContainerTask(MPILoggedTask, extensions.ContainerIOMixin):
    save: bool | list[bool]
    output_root: str
    output_name: str | list[str]
    output_format: FileFormat
    compression: dict | bool
    nan_check: bool
    nan_skip: bool
    nan_dump: bool
    versions: dict
    pipeline_config: dict
    tag: str
    attrs: dict | None
    _count: int
    done: bool
    _no_input: bool
    #
    def __init__(self) -> None: ...
    def next(
        self, *input: Iterable[Container]
    ) -> tuple[Container, ...] | Container | None: ...
    def finish(self) -> tuple[Container, ...] | Container | None: ...
    def _process_output(self, output: Container, ii: int = 0) -> Container: ...
    def _save_output(self, output: Container, ii: int = 0) -> str | None: ...
    def _nan_process_output(self, output: Container) -> Container | None: ...
    def _interpolation_dict(self, output: Container, ii: int = 0) -> dict[str, Any]: ...
    def _nan_check_walk(self, cont: Container) -> bool: ...

class SingleTask(ContainerTask):
    def __init__(self) -> None: ...

def group_tasks(*tasks: Sequence[Task]) -> Task: ...

class _ReturnLastInputOnFinish(SingleTask):
    x: Any
    def process(self, x: Any) -> None: ...
    def process_finish(self): ...

class _ReturnFirstInputOnFinish(SingleTask):
    x: Any
    def process(self, x: Any) -> None: ...
    def process_finish(self): ...
