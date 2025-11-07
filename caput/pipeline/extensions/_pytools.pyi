from collections.abc import Callable, Iterable
from typing import Any

from .._pipeline import Task

__all__ = ["Input", "Output"]

class Input(Task):
    inputs: Iterable
    _iter: Any | None
    def __init__(self, inputs: Iterable | None = None) -> None: ...
    def next(self) -> Any | None: ...

class Output(Task):
    outputs: list
    callback: Callable
    def __init__(self, callback: Callable | None = None) -> None: ...
    def next(self, in_: Any) -> None: ...
