"""Extensions to port pipelines to an interactive python environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import exceptions
from .._pipeline import Task

if TYPE_CHECKING:
    from typing import Any


__all__ = ["Input", "Output"]


class Input(Task):
    """Pass inputs into the pipeline from outside."""

    def __init__(self, inputs=None):
        super().__init__()
        self.inputs = inputs or []
        self._iter = None

    def next(self):
        """Pop and return successive inputs.

        Raises
        ------
        :py:exc:`~..exceptions.PipelineStopIteration`
            No more inputs to provide.
        """
        if self._iter is None:
            self._iter = iter(self.inputs)

        try:
            return next(self._iter)
        except StopIteration as e:
            raise exceptions.PipelineStopIteration from e


class Output(Task):
    """Take outputs from the pipeline and place them in a list.

    To apply some processing to pipeline output (i.e. this tasks input), use the
    `callback` argument which will get passed the item. The return value of the
    callback is placed in the `outputs` attribute. Note that this need not be the
    input, so if pipeline output should be deleted to save memory you can simply
    return `None`.

    Parameters
    ----------
    callback : callable | None
        A function which can apply some processing to the pipeline output.
    """

    def __init__(self, callback=None):
        super().__init__()
        self.outputs = []
        self.callback = callback

    def next(self, in_: Any) -> None:
        """Append `in_` to the outputs.

        If `callback` is not ``None``, that is called first to
        process `in_`.
        """
        if self.callback:
            in_ = self.callback(in_)

        self.outputs.append(in_)
