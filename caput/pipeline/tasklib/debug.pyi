from typing import Any

from .._pipeline import Task
from .base import ContainerTask, MPILoggedTask, SetMPILogging

class CheckMPIEnvironment(MPILoggedTask):
    timeout: int
    def setup(self) -> None: ...

class DebugInfo(MPILoggedTask, SetMPILogging):
    level_rank0: int
    level_all: int
    def __init__(self) -> None: ...
    def _get_external_ip(self) -> str: ...
    def _get_package_versions(self) -> list[tuple[str, str]]: ...

class Print(Task):
    def next(self, input_: Any) -> Any: ...

class SaveModuleVersions(ContainerTask):
    root: str
    done: bool
    def setup(self) -> None: ...
    def process(self) -> None: ...

class SaveConfig(ContainerTask):
    root: str
    done: bool
    def setup(self) -> None: ...
    def process(self) -> None: ...
